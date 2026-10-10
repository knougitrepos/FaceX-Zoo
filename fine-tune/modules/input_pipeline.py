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
    def __init__(self, size, batch_size, seed, shuffle=True, drop_last=True):
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


def prepare_mapped_index(data_root, index_file, cache_dir):
    """Cache only paths/labels, preserving image bytes and original index files."""
    index_file, root = Path(index_file).resolve(), Path(data_root).resolve()
    digest = hashlib.sha256()
    with index_file.open("rb") as src:
        for chunk in iter(lambda: src.read(1024**2), b""):
            digest.update(chunk)
    key = hashlib.sha256((digest.hexdigest() + str(root) + "path_index_v1").encode()).hexdigest()
    cache = Path(cache_dir) / (key + ".npy")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        return MappedSamples(cache)
    count, max_bytes, base = 0, 1, None
    def rows():
        with index_file.open(encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    rel, label = line.strip().rsplit(" ", 1)
                    yield rel.replace("\\", "/"), int(label)
    for rel, label in rows():
        if base is None:
            candidates = [root / rel, root / "train" / rel]
            if rel.startswith("train/"):
                candidates.append(root / rel[6:])
            actual = next((p for p in candidates if p.is_file()), None)
            if actual is None:
                raise FileNotFoundError(f"First indexed image missing: {root / rel}")
            # absolute indexes are permitted; the root is not prefixed to them.
            base = root / "train" if actual == candidates[1] else root
            strip_train = rel.startswith("train/") and actual == candidates[-1] and len(candidates) == 3
        clean = rel[6:] if strip_train and rel.startswith("train/") else rel
        path = str(base / clean)
        max_bytes = max(max_bytes, len(path.encode("utf-8")))
        count += 1
    if count == 0:
        raise ValueError("Empty image index")
    temp = cache.with_name(cache.name + f".{uuid.uuid4().hex}.tmp")
    data = None
    try:
        data = np.lib.format.open_memmap(temp, mode="w+", dtype=[("path", f"S{max_bytes}"), ("label", "<i4")], shape=(count,))
        for i, (rel, label) in enumerate(rows()):
            clean = rel[6:] if strip_train and rel.startswith("train/") else rel
            data[i] = (str(base / clean).encode("utf-8"), label)
        data.flush()
        del data
        data = None
        os.replace(temp, cache)
        cache.with_suffix(".json").write_text(json.dumps({"source": str(index_file), "source_sha256": digest.hexdigest(),
                                                        "root": str(root), "rows": count, "bytes": cache.stat().st_size}, indent=2), encoding="utf-8")
    finally:
        if data is not None:
            del data
        temp.unlink(missing_ok=True)
    return MappedSamples(cache)
