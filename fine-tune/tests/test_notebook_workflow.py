"""Notebook settings and actual small-image campaign contracts, without VGGFace2 training."""
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fine-tune"))
from modules import notebook_workflow as flow
from modules.dataset import FaceImageDataset
from modules.checkpoint_manager import load_checkpoint


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / "data"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/identity_meta.csv").write_text("id,name,count,flag\na,A,4,1\nb,B,4,1\n", encoding="utf-8")
    lines = []
    for identity in ("a", "b"):
        (root / identity).mkdir()
        for i in range(4):
            cv2.imwrite(str(root / identity / f"{i}.png"), np.full((120, 120, 3), 60 + i * 30, dtype=np.uint8))
            lines.append(f"{identity}/{i}.png")
    (root / "data/train_list.txt").write_text("\n".join(lines), encoding="utf-8")
    result = flow.resolve_settings(ROOT, overrides={
        "project": {"seed_list": [8972, 8973], "keep_raw_results": False},
        "dataset": {"raw_root": str(root), "num_classes": 2,
                    "meta_csv": str(root / "meta/identity_meta.csv"),
                    "train_archive": str(root / "missing_train.tar.gz"),
                    "test_archive": str(root / "missing_test.tar.gz"),
                    "train_list": str(tmp_path / "train.txt"), "val_list": str(tmp_path / "val.txt"),
                    "pixel_mean": 0.0, "pixel_std": 255.0},
        "backbone": {"feat_dim": 8},
        "training": {"epoches": 1, "batch_size": 2, "num_workers": 0},
        "paths": {"checkpoint_dir": str(tmp_path / "checkpoints"), "export_dir": str(tmp_path / "exports"),
                  "report_dir": str(tmp_path / "reports")},
    })
    flow.prepare_indices(result, val_every=2)
    return result


def test_current_notebook_variables_override_saved_and_yaml(cfg, tmp_path):
    path = tmp_path / "settings.json"
    flow.save_settings(path, cfg)
    current = flow.resolve_settings(ROOT, path, {"project": {"keep_raw_results": True}, "training": {"batch_size": 4}})
    assert current["project"]["keep_raw_results"] is True
    assert current["training"]["batch_size"] == 4
    assert current["dataset"]["pixel_std"] == 255


def test_preprocessing_and_workers_reach_train_and_val(cfg):
    train, val = flow.make_loaders(cfg)
    for loader in (train, val):
        assert loader.num_workers == 0
        assert loader.dataset.pixel_mean == 0
        assert loader.dataset.pixel_std == 255
    x, _ = next(iter(val))
    assert x.shape[1:] == (3, 112, 112)
    assert 0 <= x.min() <= x.max() <= 1
    ds = FaceImageDataset(cfg["dataset"]["raw_root"], cfg["dataset"]["val_list"],
                          target_size=(96, 80), pixel_mean=0, pixel_std=255, is_train=False)
    assert ds[0][0].shape == (3, 96, 80)  # generic loader honors height and width


def test_unsupported_notebook_input_and_workers_fail_explicitly(cfg):
    for workers in (1, 2, 4):
        altered = flow.merge_settings(cfg, {"training": {"num_workers": workers}})
        with pytest.raises(ValueError, match="NUM_WORKERS=0"):
            flow.validate_settings(altered)
    with pytest.raises(ValueError, match="INPUT_SIZE"):
        flow.validate_settings(flow.merge_settings(cfg, {"dataset": {"input_size": [96, 96]}}))


def test_seed_head_matrix_and_unchanged_lists(cfg):
    assert len(flow.plan_matrix(cfg, "pretrain")) == 2
    plan = flow.plan_matrix(cfg, "finetune")
    assert [(r["seed"], r["head"]) for r in plan] == [
        (seed, head) for seed in (8972, 8973) for head in flow.HEAD_NAMES]
    original = Path(cfg["dataset"]["train_list"]).read_bytes()
    flow.prepare_indices(cfg, 2)
    assert Path(cfg["dataset"]["train_list"]).read_bytes() == original
    with pytest.raises(FileExistsError):
        flow.prepare_indices(cfg, 3)
    assert Path(cfg["dataset"]["train_list"]).read_bytes() == original


def test_environment_never_claims_unchecked_integrity(cfg):
    state = flow.inspect_environment(ROOT, cfg)
    assert state["train_archive"]["exists"] is False
    assert state["train_archive"]["bytes"] is None
    assert state["archive_integrity"] == "not_checked"
    assert state["all_images_decode"] == "not_checked"
    assert state["identity_count_matches"] is True


def test_real_small_image_matrix_and_export(cfg, monkeypatch):
    # CPU only for this bounded test; production formal path still requires CUDA.
    monkeypatch.setattr(flow, "_formal_device", lambda config: torch.device("cpu"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        parents = flow.run_training_matrix(cfg, "pretrain")
        assert len(parents) == 2 and all(r["completed"] for r in parents)
        runs = flow.run_training_matrix(cfg, "finetune")
        assert len(runs) == 6 and all(r["completed"] for r in runs)
        for row in runs:
            checkpoint = load_checkpoint(row["checkpoint"])
            assert checkpoint["config"]["preprocessing"]["pixel_std"] == 255
            assert checkpoint["extra_meta"]["execution_settings"]["project"]["keep_raw_results"] is False
        again = flow.run_training_matrix(cfg, "finetune")
        assert all(r["status"] == "reused_completed" for r in again)
        assert [r["sha256"] for r in again] == [r["sha256"] for r in runs]
        exports = flow.export_matrix(cfg)
        assert len(exports) == 6 and all(r["completed"] for r in exports)
        assert all(r["status"] == "reused_completed" for r in flow.export_matrix(cfg))
        for out in exports:
            meta = json.loads(Path(out["export"]).with_name(Path(out["export"]).stem + "_meta.json").read_text())
            assert meta["preprocessing"]["pixel_std"] == 255
            assert meta["fiqa_evaluation"] == "not_executed"
        changed = flow.merge_settings(cfg, {"dataset": {"pixel_std": 128.0}})
        with pytest.raises(ValueError):
            flow.run_training_matrix(changed, "finetune")
    finally:
        torch.set_num_threads(old_threads)


def test_campaign_failure_records_remaining_work(cfg, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(flow, "_run_one", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        flow.run_training_matrix(cfg, "finetune")
    summary = json.loads(next(Path(cfg["paths"]["report_dir"]).glob("*.json")).read_text())
    assert summary["completed"] == 0 and len(summary["not_started"]) == 5
    assert summary["rows"][0]["status"] == "failed"


def test_interrupted_resume_allows_subset_and_retention_override(cfg, monkeypatch):
    from modules.trainer import Trainer
    monkeypatch.setattr(flow, "_formal_device", lambda config: torch.device("cpu"))
    original = Trainer.train_one_epoch
    cfg = flow.merge_settings(cfg, {"training": {"epoches": 2, "save_freq": 1},
                                     "project": {"keep_raw_results": True}})
    def interrupted(self, epoch):
        if epoch == 2:
            raise RuntimeError("interrupt after checkpoint")
        return original(self, epoch)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setattr(Trainer, "train_one_epoch", interrupted)
        with pytest.raises(RuntimeError, match="interrupt"):
            flow.run_training_matrix(cfg, "pretrain")
        run = Path(cfg["paths"]["checkpoint_dir"]) / "pretrain_arcface_seed8972"
        before = (run / "notebook_settings.json").read_bytes()
        checkpoint = run / "pretrain_mobilefacenet_8d_epoch_1.pt"
        monkeypatch.setattr(Trainer, "train_one_epoch", original)
        selected = flow.merge_settings(cfg, {"project": {"seed_list": [8972], "keep_raw_results": False}})
        rows = flow.run_training_matrix(selected, "pretrain", {"pretrain_arcface_seed8972": str(checkpoint)})
        assert rows[0]["completed"] and rows[0]["keep_raw_results"] is False
        assert (run / "notebook_settings.json").read_bytes() == before
        assert len(list(run.glob("resume_settings_*.json"))) == 1
    finally:
        torch.set_num_threads(threads)
