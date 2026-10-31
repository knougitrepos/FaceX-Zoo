"""Reject damaged caches and invalid batches before any training step."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import dataset, input_pipeline as pipeline
from modules.shared_storage import artifact_lock


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.png").write_bytes(b"unused: no image decode in index preparation")
    index = tmp_path / "source.txt"
    index.write_text("a.png 0\na.png 1\n", encoding="utf-8")
    return root, index, tmp_path / "cache"


def prepared(source):
    samples = pipeline.prepare_mapped_index(*source)
    path = Path(samples.path)
    assert list(samples.labels) == [0, 1]
    samples._data._mmap.close()
    return path


def test_verified_cache_reuses_identical_bytes_and_preserves_legacy(source):
    source[2].mkdir()
    legacy = source[2] / "legacy.npy"
    legacy.write_bytes(b"untouched old v1 cache")
    path = prepared(source)
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert prepared(source) == path
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert legacy.read_bytes() == b"untouched old v1 cache"


@pytest.mark.parametrize("damage", ["label", "path", "missing_receipt", "root", "rows", "dtype", "hash", "truncated", "rehash_label"])
def test_corrupt_cache_never_silently_reused(source, damage):
    path = prepared(source)
    receipt = path.with_suffix(".json")
    if damage in ("label", "path", "rehash_label"):
        data = np.load(path, mmap_mode="r+")
        data[0]["path" if damage == "path" else "label"] = b"other.png" if damage == "path" else 1
        data.flush()
        data._mmap.close()
        if damage == "rehash_label":
            meta = json.loads(receipt.read_text())
            meta["cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt.write_text(json.dumps(meta))
    elif damage == "missing_receipt":
        receipt.unlink()
    elif damage == "truncated":
        path.write_bytes(b"broken npy")
    else:
        meta = json.loads(receipt.read_text())
        meta[{"hash": "cache_sha256"}.get(damage, damage)] = "invalid"
        receipt.write_text(json.dumps(meta))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete or invalid"):
        pipeline.prepare_mapped_index(*source)
    assert path.read_bytes() == before


def test_source_change_gets_new_cache_without_rewriting_old(source):
    old = prepared(source)
    before = old.read_bytes()
    source[1].write_text("a.png 1\na.png 0\n", encoding="utf-8")
    samples = pipeline.prepare_mapped_index(*source)
    assert samples.path != str(old)
    assert list(samples.labels) == [1, 0]
    samples._data._mmap.close()
    assert old.read_bytes() == before


def test_cache_lock_blocks_a_second_writer(source):
    path = prepared(source)
    with artifact_lock(path), pytest.raises(FileExistsError):
        pipeline.prepare_mapped_index(*source)


def test_interrupted_cache_publication_is_rejected(source, monkeypatch):
    replace = pipeline.os.replace
    def fail_receipt(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("injected receipt failure")
        return replace(src, dst)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline.os, "replace", fail_receipt)
        with pytest.raises(OSError, match="injected"):
            pipeline.prepare_mapped_index(*source)
    with pytest.raises(ValueError, match="incomplete or invalid"):
        pipeline.prepare_mapped_index(*source)
    assert not list(source[2].glob("*.tmp"))


def test_source_edit_during_build_does_not_publish(source, monkeypatch):
    rows = pipeline._index_rows
    def changing_rows(index, root):
        yield from rows(index, root)
        source[1].write_text("a.png 1\na.png 0\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_index_rows", changing_rows)
    with pytest.raises(ValueError, match="changed during cache construction"):
        pipeline.prepare_mapped_index(*source)
    assert not list(source[2].glob("*.npy"))
    assert not list(source[2].glob("*.tmp"))


@pytest.mark.parametrize("batch", [0, -1, 1, True, 2.5])
def test_bad_training_batch_size_rejected_before_dataset_access(batch):
    with pytest.raises(ValueError, match="batch_size"):
        dataset.build_dataloader("missing", "missing", batch_size=batch)


@pytest.mark.parametrize("size,batch", [(176205, 4), (1, 64), (5, 4)])
def test_existing_singleton_guard_prevents_training(size, batch, monkeypatch):
    monkeypatch.setattr(dataset, "FaceImageDataset", lambda **kw: range(size))
    with pytest.raises(ValueError, match="Final singleton.*BatchNorm"):
        dataset.build_dataloader("unused", "unused", batch_size=batch)


@pytest.mark.parametrize("size,batch,train", [(176205, 16, True), (176205, 64, True), (5, 4, False), (1, 1, False)])
def test_compatible_batches_keep_all_rows(size, batch, train, monkeypatch):
    monkeypatch.setattr(dataset, "FaceImageDataset", lambda **kw: range(size))
    loader = dataset.build_dataloader("unused", "unused", batch_size=batch, is_train=train)
    batches = list(loader.batch_sampler)
    indices = [i for part in batches for i, _ in part]
    assert len(indices) == size and len(set(indices)) == size
    if train:
        assert min(map(len, batches)) >= 2
