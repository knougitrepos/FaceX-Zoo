"""Dataset transforms, name mapping, coverage, and parent/target policy separation."""
import copy
import io
import json
import sqlite3
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fine-tune"))
from modules import alignment as prep
from modules import notebook_workflow as flow
from test_notebook_workflow import cfg, aligned_cfg


@pytest.mark.parametrize("name", ["tinyface", "survface", "scface"])
def test_lowres_preserves_full_crop_channels_order_and_labels(name, tmp_path, monkeypatch):
    cfg = flow.resolve_settings(ROOT)
    # A resize path must work without any landmark archive or SDK detector.
    cfg["alignment"]["vggface2"]["archive"] = str(tmp_path / "absent.tar.gz")
    monkeypatch.setattr(prep, "ProvidedLandmarkAligner", lambda *a: pytest.fail("Landmarker called"))
    rng = np.random.default_rng(14)
    original = rng.integers(0, 256, (15, 13, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "face.png"), original)
    index = tmp_path / "source.txt"
    index.write_text("face.png 7\n", encoding="utf-8")
    ds = {"name": name, "raw_root": str(tmp_path)}
    provider = prep.create_preprocessor(ROOT, cfg, ds)
    report = prep.prepare_alignment(index, tmp_path, tmp_path / "out", provider)
    assert report["completed"] and report["prepared"] == 1 and report["failed"] == 0
    record = json.loads(Path(report["manifest"]).read_text())
    assert record["label"] == 7 and record["transform"] == "resize_only"
    assert record["source_size"] == [15, 13] and "landmarks" not in record
    expected = cv2.resize(original, (112, 112), interpolation=cv2.INTER_CUBIC)
    np.testing.assert_array_equal(cv2.imread(record["output"]), expected)
    prep.verify_alignment(tmp_path / "out/report.json", index, provider.policy)
    changed = copy.deepcopy(provider.policy)
    changed["interpolation"] = "bilinear"
    with pytest.raises(ValueError, match="policy"):
        prep.verify_alignment(tmp_path / "out/report.json", index, changed)


def test_decode_failure_retains_manifest_row_and_blocks_training(tmp_path):
    cfg = flow.resolve_settings(ROOT)
    index = tmp_path / "source.txt"
    index.write_text("bad.jpg 0\ngood.png 1\n")
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    cv2.imwrite(str(tmp_path / "good.png"), np.zeros((12, 11, 3), np.uint8))
    provider = prep.create_preprocessor(ROOT, cfg, {"name": "tinyface"})
    report = prep.prepare_alignment(index, tmp_path, tmp_path / "out", provider)
    records = [json.loads(line) for line in Path(report["manifest"]).read_text().splitlines()]
    assert [r["status"] for r in records] == ["failed", "prepared"]
    assert report["processed"] == 2 and not report["completed"]
    with pytest.raises(ValueError, match="partial/failed"):
        prep.verify_alignment(tmp_path / "out/report.json", index, provider.policy)


def archive_with_rows(path, rows):
    member = "bb_landmark/loose_landmark_train.csv"
    header = "NAME_ID," + ",".join(f"P{i}{a}" for i in range(1, 6) for a in "XY") + "\n"
    data = (header + "\n".join(rows) + "\n").encode()
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return member


def test_landmarks_are_looked_up_by_exact_image_key(tmp_path):
    path = tmp_path / "meta.tar.gz"
    member = archive_with_rows(path, ['"n000002/0002_01",2,3,4,5,6,7,8,9,10,11',
                                     '"n000002/0001_01",1,2,3,4,5,6,7,8,9,10'])
    index = prep.VGGFace2Landmarks(path, member, tmp_path / "cache")
    try:
        assert index.count == 2
        assert index.lookup("train/n000002/0001_01.jpg") == list(range(1, 11))
        assert index.lookup("n000002/0002_01") == list(range(2, 12))
        with pytest.raises(ValueError, match="Missing provided"):
            index.lookup("n000002/absent.jpg")
    finally:
        index.close()
    assert not list((tmp_path / "cache").iterdir())


def test_duplicate_landmark_keys_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "meta.tar.gz"
    row = '"n000002/0001_01",1,2,3,4,5,6,7,8,9,10'
    member = archive_with_rows(path, [row, row])
    with pytest.raises(sqlite3.IntegrityError):
        prep.VGGFace2Landmarks(path, member, tmp_path / "cache")


def test_parent_policy_is_distinct_from_downstream_and_both_are_bound(aligned_cfg):
    cfg = aligned_cfg
    parent = prep.bind_aligned_dataset(cfg)["dataset"]
    selected = copy.deepcopy(cfg)
    selected["dataset"] = flow.merge_settings(cfg["dataset"], cfg["finetune_profiles"]["tinyface"])
    target = prep.bind_aligned_dataset(selected)["dataset"]
    assert parent["alignment"] == "vggface2_official5_arcface112_v2"
    assert target["alignment"] == "resize_only"
    fields = ("input_size", "pixel_mean", "pixel_std", "alignment", "alignment_policy")
    parent_config = {"preprocessing": {key: parent[key] for key in fields}}
    flow.validate_parent_preprocessing(parent_config, cfg["dataset"], cfg["alignment"])
    parent_config["preprocessing"]["alignment_policy"] = target["alignment_policy"]
    with pytest.raises(ValueError, match="Common parent preprocessing.alignment_policy"):
        flow.validate_parent_preprocessing(parent_config, cfg["dataset"], cfg["alignment"])
