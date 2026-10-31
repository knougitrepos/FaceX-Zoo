"""Spawn-safe, bounded-memory input preparation and deterministic epoch sampling."""
import hashlib
import json
import math
import os
import random
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Sampler


def initialize_worker(worker_id):
    # Keep decoding workers CPU-only and prevent nested thread oversubscription.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class EpochBatchSampler(Sampler):
    """Epoch ordering independent of worker count, with explicit resume offset."""
    def __init__(self, size, batch_size, seed, shuffle=True, drop_last=False):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.shuffle, self.drop_last = shuffle, drop_last
        self.epoch, self.start_batch = 1, 0

    @property
    def total_batches(self):
        return self.size // self.batch_size if self.drop_last else math.ceil(self.size / self.batch_size)

    def set_epoch(self, epoch, start_batch=0):
        if not 0 <= start_batch <= self.total_batches:
            raise ValueError("Resume batch is outside this epoch")
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self):
        return self.total_batches - self.start_batch

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.size, generator=generator)
        else:
            order = range(self.size)
        for batch in range(self.start_batch, self.total_batches):
            indexes = order[batch * self.batch_size:min((batch + 1) * self.batch_size, self.size)]
            yield [(int(index), self.epoch) for index in indexes]


class MappedSamples:
    """Reopen a compact index in spawned workers instead of pickling millions of strings."""
    def __init__(self, path):
        self.path = str(path)
        self._data = np.load(self.path, mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        row = self._data[index]
        return bytes(row["path"]).decode("utf-8"), int(row["label"])

    @property
    def labels(self):
        return self._data["label"]

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.__init__(state["path"])


_INDEX_SCHEMA = "path_index_v2"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_rows(index_file, root):
    """Resolve one source row at a time, with the same root policy as v1."""
    base = None
    with Path(index_file).open(encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            rel, label = line.strip().rsplit(" ", 1)
            rel, label = rel.replace("\\", "/"), int(label)
            if not 0 <= label <= np.iinfo(np.int32).max:
                raise ValueError("Image index label must be a nonnegative int32")
            if base is None:
                candidates = [root / rel, root / "train" / rel]
                if rel.startswith("train/"):
                    candidates.append(root / rel[6:])
                actual = next((p for p in candidates if p.is_file()), None)
                if actual is None:
                    raise FileNotFoundError(f"First indexed image missing: {root / rel}")
                base = root / "train" if actual == candidates[1] else root
                strip_train = rel.startswith("train/") and actual == candidates[-1] and len(candidates) == 3
            clean = rel[6:] if strip_train and rel.startswith("train/") else rel
            yield str(base / clean).encode("utf-8"), label


def _verify_mapped_cache(cache, contract):
    """Check receipt, bytes, schema and every path/label against the source."""
    data = None
    try:
        receipt = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        if any(receipt.get(k) != v for k, v in contract.items()):
            raise ValueError("cache source/root/schema differs")
        if (type(receipt.get("rows")) is not int or receipt["rows"] < 1
                or receipt.get("bytes") != cache.stat().st_size
                or receipt.get("cache_sha256") != _file_sha256(cache)):
            raise ValueError("cache size/hash/row receipt differs")
        data = np.load(cache, mmap_mode="r", allow_pickle=False)
        if (data.shape != (receipt["rows"],) or data.dtype.names != ("path", "label")
                or data.dtype["path"].kind != "S"
                or data.dtype != np.dtype([("path", f"S{data.dtype['path'].itemsize}"), ("label", "<i4")])
                or json.loads(json.dumps(data.dtype.descr)) != receipt.get("dtype")):
            raise ValueError("cache shape/dtype differs")
        count = 0
        for count, (path, label) in enumerate(
                _index_rows(Path(contract["source"]), Path(contract["root"])), start=1):
            if count > len(data) or bytes(data[count - 1]["path"]) != path or int(data[count - 1]["label"]) != label:
                raise ValueError("cache path/label does not match source list")
        if count != len(data):
            raise ValueError("cache row count does not match source list")
        if _file_sha256(contract["source"]) != contract["source_sha256"]:
            raise ValueError("source list changed during cache verification")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Mapped index cache is incomplete or invalid: {cache}. "
            "Preserve it for diagnosis and select a new cache directory; "
            "the source images and source list were not modified."
        ) from exc
    finally:
        if data is not None:
            data._mmap.close()


def prepare_mapped_index(data_root, index_file, cache_dir):
    """Build/reuse a verified v2 cache without overwriting legacy or bad caches.

    Validation streams the source list and memory-maps the NPY, so the checks do
    not load VGGFace2's full Python path list into RAM. V1 files remain untouched.
    Missing receipts or interrupted publications fail closed; a fresh cache
    directory is the explicit recovery path. Input files must stay immutable
    for the duration of training, including after workers reopen the index.
    """
    from .shared_storage import artifact_lock

    index_file, root = Path(index_file).resolve(), Path(data_root).resolve()
    contract = dict(schema=_INDEX_SCHEMA, source=str(index_file), root=str(root),
                    source_sha256=_file_sha256(index_file))
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / (key + ".npy")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with artifact_lock(cache):
        if cache.exists() or cache.with_suffix(".json").exists():
            _verify_mapped_cache(cache, contract)
            return MappedSamples(cache)
        return _build_mapped_index(cache, contract)


def _build_mapped_index(cache, contract):
    # Parse an immutable local snapshot so both passes use the exact bytes hashed
    # into the cache identity. Detect concurrent source edits before publication.
    token = uuid.uuid4().hex
    snapshot = cache.with_name(cache.name + f".{token}.source.tmp")
    temp = cache.with_name(cache.name + f".{token}.tmp")
    meta_temp = cache.with_name(cache.name + f".{token}.json.tmp")
    data = None
    try:
        digest = hashlib.sha256()
        with Path(contract["source"]).open("rb") as src, snapshot.open("xb") as dst:
            for chunk in iter(lambda: src.read(1024**2), b""):
                dst.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != contract["source_sha256"]:
            raise ValueError("Source list changed before cache construction")
        root = Path(contract["root"])
        count, max_bytes = 0, 1
        for path, _ in _index_rows(snapshot, root):
            count += 1
            max_bytes = max(max_bytes, len(path))
        if not count:
            raise ValueError("Empty image index")
        dtype = np.dtype([("path", f"S{max_bytes}"), ("label", "<i4")])
        data = np.lib.format.open_memmap(temp, mode="w+", dtype=dtype, shape=(count,))
        for i, row in enumerate(_index_rows(snapshot, root)):
            data[i] = row
        data.flush()
        data._mmap.close()
        data = None
        if _file_sha256(contract["source"]) != contract["source_sha256"]:
            raise ValueError("Source list changed during cache construction")
        receipt = dict(contract, rows=count, bytes=temp.stat().st_size,
                       dtype=dtype.descr, cache_sha256=_file_sha256(temp))
        meta_temp.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        # A crash between these writes leaves an invalid pair, never a reusable
        # cache. The next call rejects it instead of silently loading the NPY.
        os.replace(temp, cache)
        os.replace(meta_temp, cache.with_suffix(".json"))
        _verify_mapped_cache(cache, contract)
        return MappedSamples(cache)
    finally:
        if data is not None:
            data._mmap.close()
        snapshot.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
        meta_temp.unlink(missing_ok=True)
