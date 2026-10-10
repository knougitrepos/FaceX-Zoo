import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.archive_preparation import extract_train_archive


def make_archive(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))


def test_extract_resume_and_preserve_conflict(tmp_path):
    archive, root = tmp_path / "train.tar.gz", tmp_path / "images"
    make_archive(archive, [("train/n001/1.jpg", b"first"), ("train/n001/2.jpg", b"second")])
    result = extract_train_archive(archive, root)
    assert result["written"] == 2
    first = root / "train/n001/1.jpg"
    stamp = first.stat().st_mtime_ns
    result = extract_train_archive(archive, root)
    assert result["written"] == 0 and result["existing_verified"] == 2
    assert first.stat().st_mtime_ns == stamp
    (root / "train/n001/2.jpg").unlink()
    result = extract_train_archive(archive, root)
    assert result["written"] == 1 and result["existing_verified"] == 1
    first.write_bytes(b"other")  # Same size must not count as equal content.
    with pytest.raises(FileExistsError):
        extract_train_archive(archive, root)
    assert first.read_bytes() == b"other"
    assert not list(root.rglob("*.partial"))


@pytest.mark.parametrize("name", ["../outside.jpg", "train/../../outside.jpg", "/train/n/1.jpg",
                                 "train/n/1.jpg:stream", "train/NUL/1.jpg", "test/n/1.jpg"])
def test_reject_unsafe_or_wrong_split(tmp_path, name):
    archive = tmp_path / "train.tar.gz"
    make_archive(archive, [(name, b"x")])
    with pytest.raises(ValueError):
        extract_train_archive(archive, tmp_path / "images")


def test_reject_links(tmp_path):
    archive = tmp_path / "train.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("train/n/link.jpg")
        member.type, member.linkname = tarfile.SYMTYPE, "../../outside"
        tar.addfile(member)
    with pytest.raises(ValueError):
        extract_train_archive(archive, tmp_path / "images")


def test_extracted_images_reach_loader(tmp_path):
    import cv2
    import numpy as np
    from modules.notebook_workflow import make_loaders

    _, image = cv2.imencode(".jpg", np.full((120, 130, 3), 80, dtype=np.uint8))
    archive, root = tmp_path / "train.tar.gz", tmp_path / "images"
    make_archive(archive, [(f"train/n001/{i}.jpg", image.tobytes()) for i in range(4)])
    index = tmp_path / "index.txt"
    index.write_text("".join(f"n001/{i}.jpg 0\n" for i in range(4)))
    cfg = {"dataset": {"raw_root": str(root), "train_list": str(index), "val_list": str(index),
                       "input_size": [112, 112], "pixel_mean": 127.5, "pixel_std": 128.0, "num_classes": 1},
           "training": {"batch_size": 2, "num_workers": 0}}
    with pytest.raises(FileNotFoundError, match="EXTRACT_TRAIN_ARCHIVE"):
        make_loaders(cfg)
    extract_train_archive(archive, root)
    _, val = make_loaders(cfg)
    images, labels = next(iter(val))
    assert tuple(images.shape) == (2, 3, 112, 112)
    assert labels.tolist() == [0, 0]
