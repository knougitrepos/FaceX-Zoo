"""
Targeted Verification Suite for Audit Bug Fixes (P1 & P2).
Explicitly verifies:
1. Relative import resolution in modules.trainer
2. Fail-closed behavior on missing images (no zero vector fallback)
3. Dataset image existence validation
4. Rejection of smoke checkpoints as formal parents
5. Enforced parent hash verification in formal fine-tuning
6. Live optimizer state & parent_hash preservation in checkpoints
7. Renamed Feature Norm Diagnostics and unmeasured Rank-1 reporting
8. Reproducibility seed setting
"""

import sys
import os
import shutil
import tempfile
import pytest
import torch
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINETUNE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULES_DIR = os.path.join(FINETUNE_DIR, "modules")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if FINETUNE_DIR not in sys.path:
    sys.path.insert(0, FINETUNE_DIR)
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from modules.trainer import Trainer, set_seed
from modules.dataset import FaceImageDataset, validate_dataset_files, resolve_image_path
from modules.checkpoint_manager import (
    CheckpointManager,
    verify_parent_checkpoint,
    validate_checkpoint_for_export,
    compute_sha256,
    compute_config_hash,
    extract_train_config,
    extract_model_and_train_config,
    load_checkpoint,
)
from modules.evaluator import FeatureEvaluator
from modules.summary_reporter import ExperimentSummaryReporter
from modules.model_builder import build_model
from backbone.MobileFaceNets import MobileFaceNet


def test_import_and_seed():
    """Verify modules.trainer can be imported cleanly and set_seed works with goal.md seed (8972)."""
    assert callable(Trainer)
    assert callable(set_seed)
    # Test default seed (8972)
    set_seed()
    t1 = torch.randn(5)
    set_seed(8972)
    t2 = torch.randn(5)
    assert torch.allclose(t1, t2)
    # Test reproducibility
    set_seed(8972)
    t3 = torch.randn(5)
    assert torch.allclose(t1, t3)


def test_fail_closed_missing_image():
    """P1: Verify dataset raises FileNotFoundError on missing image (no zero vector fallback)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_file = os.path.join(tmp_dir, "test_idx.txt")
        with open(idx_file, "w") as f:
            f.write("does_not_exist/img_001.jpg 10\n")

        ds = FaceImageDataset(data_root=tmp_dir, index_file=idx_file, is_train=False)
        with pytest.raises(FileNotFoundError, match="Image file does not exist"):
            _ = ds[0]


def test_validate_dataset_files():
    """P1: Verify validate_dataset_files detects missing paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_file = os.path.join(tmp_dir, "test_idx.txt")
        with open(idx_file, "w") as f:
            f.write("missing1.jpg 0\nmissing2.jpg 1\n")

        checked, missing = validate_dataset_files(data_root=tmp_dir, index_file=idx_file)
        assert checked == 2
        assert len(missing) == 2


def test_smoke_checkpoint_rejection():
    """P1: Verify verify_parent_checkpoint strictly rejects smoke checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        smoke_ckpt = os.path.join(tmp_dir, "smoke_pretrain_mobilefacenet_1024d.pt")
        torch.save({"backbone_state_dict": {}}, smoke_ckpt)
        smoke_hash = compute_sha256(smoke_ckpt)

        with pytest.raises(ValueError, match="Smoke-test checkpoint cannot be used"):
            verify_parent_checkpoint(smoke_ckpt, expected_hash=smoke_hash, require_hash=True)


def test_enforce_expected_hash_requirement():
    """P1: Verify verify_parent_checkpoint requires expected_hash when require_hash=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        formal_ckpt = os.path.join(tmp_dir, "pretrain_mobilefacenet_1024d.pt")
        torch.save({"backbone_state_dict": {}}, formal_ckpt)

        with pytest.raises(ValueError, match="expected_hash is required"):
            verify_parent_checkpoint(formal_ckpt, expected_hash=None, require_hash=True)


def test_save_state_checkpoint_preserves_live_state():
    """P1: Verify save_state_checkpoint preserves actual optimizer state and parent_hash."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        head_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        device = torch.device("cpu")

        # Fake empty loader
        dummy_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=2)

        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20], "lr_decay_ratio": 0.1},
        }
        ckpt_mgr = CheckpointManager(tmp_dir)
        parent_hash_val = "abc123parent_hash"
        trainer = Trainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=device,
            parent_hash=parent_hash_val,
        )

        # Run 1 step to populate optimizer state
        loss_val = trainer.train_one_epoch(epoch=1)
        trainer.cur_epoch = 1
        trainer.history = {"epochs": [1], "train_loss": [loss_val], "lr": [0.001]}

        saved_file = trainer.save_state_checkpoint(
            filename="test_trained_state.pt",
            epoch=1,
            loss=loss_val,
            extra_meta={"head_type": "ArcFace"},
        )
        assert os.path.exists(saved_file)

        loaded = load_checkpoint(saved_file)
        assert loaded["epoch"] == 1
        assert "optimizer_state_dict" in loaded
        assert len(loaded["optimizer_state_dict"]["state"]) > 0  # optimizer state is populated, NOT empty!
        assert loaded["extra_meta"]["parent_hash"] == parent_hash_val
        assert loaded["extra_meta"]["total_epochs"] == 40


def test_feature_norm_diagnostics_and_summary():
    """P2: Verify feature norm diagnostics and summary report without fake 100% accuracy."""
    norms = np.array([[15.2], [18.4], [12.1], [22.0]], dtype=np.float32)
    diag = FeatureEvaluator.compute_feature_norm_diagnostics(norms)
    assert diag["mean_norm"] > 0
    assert diag["std_norm"] > 0
    assert diag["min_norm"] == float(np.min(norms))
    assert diag["max_norm"] == float(np.max(norms))

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "test_summary.md")
        report = ExperimentSummaryReporter.generate_chatgpt_summary(
            experiment_meta={"model_name": "TestModel", "head_type": "AdaFace", "feat_dim": 1024, "total_epochs": 40},
            train_history=None,
            eval_metrics={"norm_diagnostics": diag, "rank1_acc": None},
            output_md_path=report_path,
        )
        assert "100.00%" not in report
        assert "미실행" in report
        assert "특징 Norm 통계 진단 (Feature Norm Diagnostics)" in report


def test_resolve_image_path_train_subfolder():
    """P1: Verify resolve_image_path seamlessly handles root vs train/ subfolder paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Case 1: direct under root
        direct_dir = os.path.join(tmp_dir, "n000002")
        os.makedirs(direct_dir, exist_ok=True)
        img1 = os.path.join(direct_dir, "0001_01.jpg")
        with open(img1, "w") as f:
            f.write("fake")

        resolved_1 = resolve_image_path(tmp_dir, "n000002/0001_01.jpg")
        assert resolved_1 == img1

        # Case 2: unpacked under train/ subfolder
        train_sub = os.path.join(tmp_dir, "train", "n000003")
        os.makedirs(train_sub, exist_ok=True)
        img2 = os.path.join(train_sub, "0002_01.jpg")
        with open(img2, "w") as f:
            f.write("fake")

        resolved_2 = resolve_image_path(tmp_dir, "n000003/0002_01.jpg")
        assert resolved_2 == img2

        # Case 3: data_root is already tmp_dir/train, rel_path has train/ prefix
        resolved_3 = resolve_image_path(os.path.join(tmp_dir, "train"), "train/n000003/0002_01.jpg")
        assert resolved_3 == img2


def test_verify_parent_checkpoint_deep_payload_smoke_rejection():
    """P1: Verify verify_parent_checkpoint rejects smoke checkpoint even if filename has no 'smoke'."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        innocent_name = os.path.join(tmp_dir, "parent_mobilefacenet.pt")
        # Payload explicitly marked as smoke
        torch.save({"backbone_state_dict": {}, "is_smoke": True}, innocent_name)
        file_hash = compute_sha256(innocent_name)

        with pytest.raises(ValueError, match="Smoke-test checkpoint cannot be used"):
            verify_parent_checkpoint(innocent_name, expected_hash=file_hash, require_hash=True)


def test_validate_checkpoint_for_export_smoke_rejection():
    """P1: Verify validate_checkpoint_for_export strictly blocks smoke checkpoints."""
    # Smoke payload
    smoke_payload = {"state_dict": {}, "is_smoke": True, "epoch": 1}
    with pytest.raises(ValueError, match="marked as a smoke test"):
        validate_checkpoint_for_export(smoke_payload, target_epoch=40, allow_smoke=False)

    # Valid payload
    valid_payload = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "completed": True,
        "is_smoke": False,
        "stage": "finetune",
        "extra_meta": {"epoch": 40, "completed": True, "parent_hash": "hash123", "is_smoke": False, "stage": "finetune"},
    }
    meta = validate_checkpoint_for_export(valid_payload, target_epoch=40, allow_smoke=False)
    assert meta["epoch"] == 40
    assert meta["is_smoke"] is False


def test_export_backbone_only_atomic_and_filter_head():
    """P1: Verify export_backbone_only strips head layers and validates strict loading into MobileFaceNet."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Build real MobileFaceNet backbone
        real_backbone = MobileFaceNet(embedding_size=1024, out_h=7, out_w=7, dropout_rate=0.0)
        state = real_backbone.state_dict()
        # Pollute state dict with classifier head layers (e.g. ArcFace / AdaFace weights)
        polluted_state = {k: v.clone() for k, v in state.items()}
        polluted_state["head.kernel"] = torch.randn(1024, 8631)
        polluted_state["head.bias"] = torch.randn(8631)
        polluted_state["classifier.weight"] = torch.randn(8631, 1024)

        ckpt_mgr = CheckpointManager(tmp_dir)
        metadata = {
            "model_name": "MobileFaceNet_1024D_Test",
            "feat_dim": 1024,
            "head_type": "adaface",
            "is_smoke": False,
        }

        export_pt = ckpt_mgr.export_backbone_only(
            model=polluted_state,
            output_filename="exported_clean_backbone.pt",
            metadata=metadata,
            target_feat_dim=1024,
        )
        assert os.path.exists(export_pt)
        assert not os.path.exists(export_pt + ".tmp")  # temporary file was atomically cleaned up

        # Verify exported checkpoint contains NO classifier keys
        exported_data = load_checkpoint(export_pt, map_location="cpu")
        clean_keys = set(exported_data["state_dict"].keys())
        for k in clean_keys:
            assert not k.startswith("head.")
            assert not k.startswith("classifier.")

        # Re-verify strict load
        test_model = MobileFaceNet(embedding_size=1024, out_h=7, out_w=7, dropout_rate=0.0)
        test_model.load_state_dict(exported_data["state_dict"], strict=True)


def test_resume_checkpoint_restores_state_and_start_epoch():
    """P1: Verify resume_checkpoint restores model, optimizer, scheduler, RNG, and returns start_epoch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        ckpt_mgr = CheckpointManager(tmp_dir)
        ckpt_file = ckpt_mgr.save_checkpoint(
            filename="checkpoint_epoch_15.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=15,
        )

        # Create fresh model and resume
        resumed_model = build_model(backbone_conf, head_conf)
        resumed_opt = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
        resumed_sched = torch.optim.lr_scheduler.MultiStepLR(resumed_opt, milestones=[10, 20], gamma=0.1)

        start_epoch = ckpt_mgr.resume_checkpoint(
            checkpoint_path=ckpt_file,
            model=resumed_model,
            optimizer=resumed_opt,
            scheduler=resumed_sched,
        )
        assert start_epoch == 16


def test_keep_raw_results_pruning():
    """P2: Verify KEEP_RAW_RESULTS=False prunes older intermediate periodic checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        device = torch.device("cpu")

        dummy_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=2)

        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 6, "milestones": [3], "lr_decay_ratio": 0.1, "save_freq": 2},
            "project": {"keep_raw_results": False, "seed": 8972},
        }
        ckpt_mgr = CheckpointManager(tmp_dir)
        trainer = Trainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=device,
        )
        assert trainer.keep_raw_results is False

        trainer.train(start_epoch=1, model_prefix="test_prune")
        # Periodic saving happens at epoch 2, 4, 6
        # With keep_raw_results=False, epoch 2 should be pruned when epoch 4 is saved, and final (epoch 6) remains
        assert not os.path.exists(os.path.join(tmp_dir, "test_prune_epoch_2.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "test_prune_epoch_6.pt"))


def test_validate_checkpoint_for_export_rejects_incomplete_runs():
    """P1: Verify validate_checkpoint_for_export strictly rejects incomplete checkpoints."""
    # Case 1: epoch=1, completed=False, target_epoch=40 -> MUST FAIL
    incomplete_payload_1 = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 1,
        "completed": False,
        "is_smoke": False,
        "extra_meta": {"epoch": 1, "completed": False, "is_smoke": False},
    }
    with pytest.raises(ValueError, match="completed=False|less than target_epoch"):
        validate_checkpoint_for_export(incomplete_payload_1, target_epoch=40, allow_partial=False)

    # Case 2: epoch=40, but completed=False -> MUST FAIL
    incomplete_payload_2 = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "completed": False,
        "is_smoke": False,
        "extra_meta": {"epoch": 40, "completed": False, "is_smoke": False},
    }
    with pytest.raises(ValueError, match="completed=False"):
        validate_checkpoint_for_export(incomplete_payload_2, target_epoch=40, allow_partial=False)

    # Case 3: completed=True, but epoch=1 (< 40) -> MUST FAIL
    incomplete_payload_3 = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 1,
        "completed": True,
        "is_smoke": False,
        "extra_meta": {"epoch": 1, "completed": True, "is_smoke": False},
    }
    with pytest.raises(ValueError, match="less than target_epoch"):
        validate_checkpoint_for_export(incomplete_payload_3, target_epoch=40, allow_partial=False)

    # Case 4: allow_partial=True explicitly allows intermediate inspection
    partial_meta = validate_checkpoint_for_export(incomplete_payload_1, target_epoch=40, allow_partial=True)
    assert partial_meta["epoch"] == 1


def test_validate_checkpoint_for_export_rejects_seed_and_head_mismatch():
    """P1: Verify validate_checkpoint_for_export rejects seed, head, and stage mismatches."""
    valid_payload = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "completed": True,
        "is_smoke": False,
        "extra_meta": {
            "epoch": 40,
            "seed": 8973,
            "head": "arcface",
            "stage": "finetune",
            "completed": True,
            "is_smoke": False,
        },
    }

    # Requesting seed 8972, but checkpoint has 8973 -> MUST FAIL
    with pytest.raises(ValueError, match="Seed mismatch"):
        validate_checkpoint_for_export(
            valid_payload,
            target_epoch=40,
            expected_seed=8972,
            expected_head="arcface",
        )

    # Requesting head adaface, but checkpoint has arcface -> MUST FAIL
    with pytest.raises(ValueError, match="Head mismatch"):
        validate_checkpoint_for_export(
            valid_payload,
            target_epoch=40,
            expected_seed=8973,
            expected_head="adaface",
        )

    # Matching seed & head -> PASS
    meta = validate_checkpoint_for_export(
        valid_payload,
        target_epoch=40,
        expected_seed=8973,
        expected_head="arcface",
        expected_stage="finetune",
    )
    assert meta["seed"] == 8973


def test_verify_parent_checkpoint_rejects_incomplete_and_mismatch():
    """P1: Verify verify_parent_checkpoint rejects incomplete parents and seed mismatches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Incomplete parent: epoch=1, completed=False
        incomplete_path = os.path.join(tmp_dir, "parent_incomplete.pt")
        torch.save(
            {
                "backbone_state_dict": {"conv.weight": torch.randn(2)},
                "epoch": 1,
                "completed": False,
                "extra_meta": {"epoch": 1, "completed": False, "seed": 8972, "stage": "pretrain"},
            },
            incomplete_path,
        )
        h_inc = compute_sha256(incomplete_path)

        # 1. Rejected due to completed=False
        with pytest.raises(ValueError, match="incomplete \\(completed=False\\)|less than target epoch"):
            verify_parent_checkpoint(
                incomplete_path,
                expected_hash=h_inc,
                require_hash=True,
                target_epoch=40,
                allow_partial=False,
            )

        # Complete parent but wrong seed
        wrong_seed_path = os.path.join(tmp_dir, "parent_wrong_seed.pt")
        torch.save(
            {
                "backbone_state_dict": {"conv.weight": torch.randn(2)},
                "epoch": 40,
                "completed": True,
                "extra_meta": {"epoch": 40, "completed": True, "seed": 8973, "stage": "pretrain"},
            },
            wrong_seed_path,
        )
        h_ws = compute_sha256(wrong_seed_path)

        # 2. Rejected due to seed mismatch
        with pytest.raises(ValueError, match="Seed mismatch"):
            verify_parent_checkpoint(
                wrong_seed_path,
                expected_hash=h_ws,
                require_hash=True,
                target_epoch=40,
                expected_seed=8972,
            )

        # Valid parent: epoch=40, completed=True, seed=8972
        valid_parent_path = os.path.join(tmp_dir, "parent_valid.pt")
        torch.save(
            {
                "backbone_state_dict": {"conv.weight": torch.randn(2)},
                "epoch": 40,
                "completed": True,
                "extra_meta": {"epoch": 40, "completed": True, "seed": 8972, "stage": "pretrain"},
            },
            valid_parent_path,
        )
        h_val = compute_sha256(valid_parent_path)

        # 3. Passed verification
        data, h_res = verify_parent_checkpoint(
            valid_parent_path,
            expected_hash=h_val,
            require_hash=True,
            target_epoch=40,
            expected_seed=8972,
            require_completed=True,
        )
        assert h_res == h_val


def test_trainer_prevents_fake_completed_status():
    """P1: Verify Trainer.save_state_checkpoint refuses to mark completed=True when actual epochs are incomplete."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        device = torch.device("cpu")

        dummy_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=2)

        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20], "lr_decay_ratio": 0.1},
            "project": {"seed": 8972},
        }
        ckpt_mgr = CheckpointManager(tmp_dir)
        trainer = Trainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=device,
        )
        # Trainer cur_epoch is 5 (< total_epochs 40)
        trainer.cur_epoch = 5
        trainer.history = {"epochs": [1, 2, 3, 4, 5], "train_loss": [1.0] * 5, "lr": [0.001] * 5}

        # Attempt to save with extra_meta={"completed": True}
        saved_file = trainer.save_state_checkpoint(
            filename="test_override.pt",
            epoch=5,
            loss=0.5,
            extra_meta={"completed": True},
        )
        ckpt_data = load_checkpoint(saved_file)
        # MUST be overridden to False because 5 < 40
        assert ckpt_data["completed"] is False
        assert ckpt_data["extra_meta"]["completed"] is False


def test_export_backbone_only_immutability_and_overwrite_protection():
    """P1: Verify export_backbone_only blocks silent overwrites unless allow_overwrite=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        real_backbone = MobileFaceNet(embedding_size=1024, out_h=7, out_w=7, dropout_rate=0.0)
        state = real_backbone.state_dict()
        ckpt_mgr = CheckpointManager(tmp_dir)
        metadata = {
            "model_name": "MobileFaceNet_1024D_Test",
            "feat_dim": 1024,
            "head_type": "adaface",
            "seed": 8972,
            "is_smoke": False,
        }

        # First export succeeds
        export_file = ckpt_mgr.export_backbone_only(
            model=state,
            output_filename="exported_model_seed8972.pt",
            metadata=metadata,
            allow_overwrite=False,
        )
        assert os.path.exists(export_file)

        # Second export with same filename and allow_overwrite=False MUST raise FileExistsError
        with pytest.raises(FileExistsError, match="Exported model already exists"):
            ckpt_mgr.export_backbone_only(
                model=state,
                output_filename="exported_model_seed8972.pt",
                metadata=metadata,
                allow_overwrite=False,
            )

        # Export with allow_overwrite=True succeeds
        re_exported = ckpt_mgr.export_backbone_only(
            model=state,
            output_filename="exported_model_seed8972.pt",
            metadata=metadata,
            allow_overwrite=True,
        )
        assert os.path.exists(re_exported)


def test_resume_validation_and_history_accumulation():
    """P2: Verify resume validates seed/head/parent_hash and Trainer restores and accumulates history."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "AdaFace", "num_class": 10, "m": 0.4, "h": 0.333, "scale": 64.0, "t_alpha": 0.01}
        model = build_model(backbone_conf, head_conf)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "weight_decay": 0.0, "epoches": 3, "milestones": [10, 20], "lr_decay_ratio": 0.1, "batch_size": 2},
            "project": {"seed": 8972},
        }

        ckpt_mgr = CheckpointManager(tmp_dir)
        saved_file = ckpt_mgr.save_checkpoint(
            filename="ckpt_epoch_2.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            extra_meta={
                "seed": 8972,
                "head": "adaface",
                "stage": "finetune",
                "parent_hash": "parent_hash_123",
                "history": {"epochs": [1, 2], "train_loss": [2.5, 2.1], "lr": [0.001, 0.001]},
                "is_smoke": False,
                "raw_config": config,
            },
        )

        # 1. Validation failure: seed mismatch
        with pytest.raises(ValueError, match="Seed mismatch"):
            ckpt_mgr.resume_checkpoint(
                checkpoint_path=saved_file,
                model=model,
                expected_seed=9999,
            )

        # 2. Validation failure: head mismatch
        with pytest.raises(ValueError, match="Head mismatch"):
            ckpt_mgr.resume_checkpoint(
                checkpoint_path=saved_file,
                model=model,
                expected_head="arcface",
            )

        # 3. Validation failure: parent_hash mismatch
        with pytest.raises(ValueError, match="Parent hash mismatch"):
            ckpt_mgr.resume_checkpoint(
                checkpoint_path=saved_file,
                model=model,
                expected_parent_hash="wrong_hash",
            )

        # 4. Successful resume and history continuation in Trainer
        dummy_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=2)
        resumed_model = build_model(backbone_conf, head_conf)
        trainer = Trainer(
            model=resumed_model,
            train_loader=dummy_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            parent_hash="parent_hash_123",
        )

        start_epoch = trainer.resume_from_checkpoint(saved_file, expected_head="adaface")
        assert start_epoch == 3
        assert trainer.history["epochs"] == [1, 2]

        # Train epoch 3: history should now contain epochs [1, 2, 3] (accumulated!)
        final_history = trainer.train(start_epoch=start_epoch, model_prefix="accumulated")
        assert final_history["epochs"] == [1, 2, 3]
        assert len(final_history["train_loss"]) == 3


def test_summary_reporter_annotates_missing_epochs():
    """P2: Verify ExperimentSummaryReporter notes missing resumed epochs in convergence table."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "test_resumed_summary.md")
        resumed_history = {
            "epochs": [11, 12],
            "train_loss": [1.5, 1.4],
            "lr": [0.0001, 0.0001],
            "missing_epochs": list(range(1, 11)),
        }
        ExperimentSummaryReporter.generate_chatgpt_summary(
            experiment_meta={"model_name": "TestResumed", "head_type": "AdaFace", "feat_dim": 1024, "total_epochs": 40},
            train_history=resumed_history,
            eval_metrics={},
            output_md_path=report_path,
        )
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Epoch 1 ~ 10" in content
        assert "누락" in content or "미기록" in content


def test_validate_checkpoint_for_export_rejects_missing_metadata_fields():
    """P1: Verify validate_checkpoint_for_export strictly rejects checkpoints missing completed/seed/head/stage."""
    # 1. Missing completed flag (completed=None)
    payload_no_completed = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "seed": 8972,
        "head": "adaface",
        "stage": "finetune",
    }
    with pytest.raises(ValueError, match="not marked completed"):
        validate_checkpoint_for_export(payload_no_completed, target_epoch=40, expected_seed=8972, expected_head="adaface")

    # 2. Missing seed when expected_seed requested
    payload_no_seed = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "completed": True,
        "head": "adaface",
        "stage": "finetune",
    }
    with pytest.raises(ValueError, match="lacks required seed metadata"):
        validate_checkpoint_for_export(payload_no_seed, target_epoch=40, expected_seed=8972, expected_head="adaface")

    # 3. Missing head when expected_head requested
    payload_no_head = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "completed": True,
        "seed": 8972,
        "stage": "finetune",
    }
    with pytest.raises(ValueError, match="lacks required head metadata"):
        validate_checkpoint_for_export(payload_no_head, target_epoch=40, expected_seed=8972, expected_head="adaface")

    # 4. Missing stage when expected_stage requested
    payload_no_stage = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "completed": True,
        "seed": 8972,
        "head": "adaface",
    }
    with pytest.raises(ValueError, match="lacks required stage metadata"):
        validate_checkpoint_for_export(payload_no_stage, target_epoch=40, expected_seed=8972, expected_head="adaface", expected_stage="finetune")

    # 5. Mismatched parent_hash
    payload_wrong_parent = {
        "backbone_state_dict": {"conv.weight": torch.randn(2)},
        "epoch": 40,
        "completed": True,
        "seed": 8972,
        "head": "adaface",
        "stage": "finetune",
        "parent_hash": "correct_parent_hash",
    }
    with pytest.raises(ValueError, match="Parent hash mismatch"):
        validate_checkpoint_for_export(
            payload_wrong_parent,
            target_epoch=40,
            expected_seed=8972,
            expected_head="adaface",
            expected_stage="finetune",
            expected_parent_hash="different_parent_hash",
        )


def test_verify_parent_checkpoint_rejects_missing_metadata():
    """P1: Verify verify_parent_checkpoint rejects parents missing completed, epoch, seed, or stage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Checkpoint without completed flag
        p_no_comp = os.path.join(tmp_dir, "parent_no_comp.pt")
        torch.save({"backbone_state_dict": {"conv": torch.randn(2)}, "epoch": 40, "seed": 8972, "stage": "pretrain"}, p_no_comp)
        h1 = compute_sha256(p_no_comp)
        with pytest.raises(ValueError, match="incomplete or lacks completed=True"):
            verify_parent_checkpoint(p_no_comp, expected_hash=h1, target_epoch=40, expected_seed=8972)

        # Checkpoint without seed
        p_no_seed = os.path.join(tmp_dir, "parent_no_seed.pt")
        torch.save({"backbone_state_dict": {"conv": torch.randn(2)}, "epoch": 40, "completed": True, "stage": "pretrain"}, p_no_seed)
        h2 = compute_sha256(p_no_seed)
        with pytest.raises(ValueError, match="lacks seed metadata"):
            verify_parent_checkpoint(p_no_seed, expected_hash=h2, target_epoch=40, expected_seed=8972)

        # Checkpoint without stage
        p_no_stage = os.path.join(tmp_dir, "parent_no_stage.pt")
        torch.save({"backbone_state_dict": {"conv": torch.randn(2)}, "epoch": 40, "completed": True, "seed": 8972}, p_no_stage)
        h3 = compute_sha256(p_no_stage)
        with pytest.raises(ValueError, match="lacks stage metadata"):
            verify_parent_checkpoint(p_no_stage, expected_hash=h3, target_epoch=40, expected_seed=8972)


def test_resume_checkpoint_rejects_stage_mismatch_and_missing_optimizer():
    """P2: Verify resume_checkpoint enforces stage match and requires optimizer_state_dict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_file = ckpt_mgr.save_checkpoint(
            filename="pretrain_ckpt.pt",
            model=model,
            optimizer=opt,
            scheduler=None,
            epoch=10,
            extra_meta={"stage": "pretrain", "head": "arcface", "seed": 8972},
        )

        # Resuming expecting 'finetune' stage MUST FAIL
        with pytest.raises(ValueError, match="Stage mismatch"):
            ckpt_mgr.resume_checkpoint(
                ckpt_file,
                model=model,
                optimizer=opt,
                expected_seed=8972,
                expected_head="arcface",
                expected_stage="finetune",
            )

        # Saving checkpoint without optimizer state
        ckpt_no_opt = os.path.join(tmp_dir, "no_opt.pt")
        torch.save({"backbone_state_dict": model.backbone.state_dict(), "epoch": 10, "extra_meta": {"stage": "pretrain", "seed": 8972, "head": "arcface"}}, ckpt_no_opt)
        with pytest.raises(ValueError, match="does not contain optimizer_state_dict"):
            ckpt_mgr.resume_checkpoint(ckpt_no_opt, model=model, optimizer=opt, expected_seed=8972)


def test_trainer_val_loss_key_error_fixed_on_resume():
    """P2: Verify Trainer.train does NOT raise KeyError: 'val_loss' when resuming history without val_loss."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "ArcFace", "num_class": 4, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)

        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        val_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2)

        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 2, "milestones": [10], "lr_decay_ratio": 0.1, "save_freq": 5},
            "project": {"seed": 8972},
        }
        trainer = Trainer(model, train_loader, val_loader, config, ckpt_mgr, torch.device("cpu"), head_type="ArcFace", stage="finetune")
        # History restored without val_loss key
        trainer.history = {"epochs": [1], "train_loss": [1.5], "lr": [0.001]}

        # Must execute epoch 2 without crashing
        hist = trainer.train(start_epoch=2)
        assert "val_loss" in hist
        assert len(hist["val_loss"]) == 1
        assert hist["epochs"] == [1, 2]


def test_trainer_does_not_create_false_41_epoch_completed_checkpoint():
    """P1: Verify resuming an already completed 40-epoch run does not create a fake 41-epoch checkpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "ArcFace", "num_class": 4, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10], "lr_decay_ratio": 0.1},
            "project": {"seed": 8972},
        }
        trainer = Trainer(model, train_loader, None, config, ckpt_mgr, torch.device("cpu"), head_type="ArcFace", stage="finetune")
        trainer.history = {"epochs": list(range(1, 41)), "train_loss": [0.5]*40, "lr": [0.001]*40}

        # Calling train with start_epoch=41 (resumed after completed 40)
        hist = trainer.train(start_epoch=41)
        assert trainer.cur_epoch == 40  # Must be clamped to total_epochs 40, NOT 41!
        assert trainer.epochs_trained_in_run == 0

        saved_path = trainer.save_state_checkpoint("test_saved.pt", epoch=trainer.cur_epoch)
        loaded = load_checkpoint(saved_path)
        assert loaded["epoch"] == 40  # Stays at 40!


def test_checkpoint_manager_blocks_overwriting_completed_checkpoint():
    """P1: Verify save_checkpoint refuses to overwrite existing completed checkpoints when allow_overwrite=False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "ArcFace", "num_class": 4, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # First save with completed=True
        ckpt_mgr.save_checkpoint("completed_run.pt", model, opt, None, epoch=40, extra_meta={"completed": True, "seed": 8972})

        # Second save with allow_overwrite=False MUST raise FileExistsError
        with pytest.raises(FileExistsError, match="Completed checkpoint already exists"):
            ckpt_mgr.save_checkpoint("completed_run.pt", model, opt, None, epoch=1, extra_meta={"completed": False, "seed": 8972}, allow_overwrite=False)


def test_trainer_periodic_checkpoints_contain_stage_and_head():
    """P1: Verify periodic checkpoints saved during Trainer.train include stage and head in metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        head_conf = {"type": "AdaFace", "num_class": 4, "m": 0.4, "h": 0.333, "scale": 64.0, "t_alpha": 0.01}
        model = build_model(backbone_conf, head_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 1, "milestones": [10], "lr_decay_ratio": 0.1, "save_freq": 1},
            "project": {"seed": 8972},
        }
        trainer = Trainer(model, train_loader, None, config, ckpt_mgr, torch.device("cpu"), head_type="AdaFace", stage="finetune", parent_hash="phash_abc")
        trainer.train(start_epoch=1, model_prefix="test_periodic")

        periodic_path = os.path.join(tmp_dir, "test_periodic_epoch_1.pt")
        assert os.path.exists(periodic_path)
        loaded = load_checkpoint(periodic_path)
        assert loaded["stage"] == "finetune"
        assert loaded["head"] == "AdaFace"
        assert loaded["extra_meta"]["stage"] == "finetune"
        assert loaded["extra_meta"]["head"] == "AdaFace"
        assert loaded["extra_meta"]["parent_hash"] == "phash_abc"


def test_pretrain_trainer_explicit_stage_contract():
    """P1: Verify pretrain Trainer creates checkpoints with stage='pretrain' and prevents stage mismatch on resume."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        head_conf = {"type": "ArcFace", "num_class": 4, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 1, "milestones": [10], "lr_decay_ratio": 0.1, "save_freq": 1},
            "project": {"seed": 8972},
        }

        # Explicitly pass stage="pretrain"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="pretrain",
        )
        assert trainer.stage == "pretrain"

        # Train 1 epoch
        trainer.train(start_epoch=1, model_prefix="pretrain_mobilefacenet_1024d")
        periodic_path = os.path.join(tmp_dir, "pretrain_mobilefacenet_1024d_epoch_1.pt")
        assert os.path.exists(periodic_path)
        ckpt_periodic = load_checkpoint(periodic_path)
        assert ckpt_periodic["stage"] == "pretrain"
        assert ckpt_periodic["extra_meta"]["stage"] == "pretrain"

        # Save formal state checkpoint
        formal_path = trainer.save_state_checkpoint("pretrain_mobilefacenet_1024d.pt", epoch=1)
        ckpt_formal = load_checkpoint(formal_path)
        assert ckpt_formal["stage"] == "pretrain"
        assert ckpt_formal["extra_meta"]["stage"] == "pretrain"

        # Resuming with expected_stage="pretrain" MUST SUCCEED
        resumed_model = build_model(backbone_conf, head_conf)
        resumed_trainer = Trainer(
            model=resumed_model,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="pretrain",
        )
        start_epoch = resumed_trainer.resume_from_checkpoint(formal_path, expected_stage="pretrain")
        assert start_epoch == 2

        # Resuming with expected_stage="finetune" MUST FAIL with Stage mismatch
        mismatch_model = build_model(backbone_conf, head_conf)
        mismatch_trainer = Trainer(
            model=mismatch_model,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
        )
        with pytest.raises(ValueError, match="Stage mismatch! Expected stage 'finetune', but resume checkpoint specifies stage 'pretrain'"):
            mismatch_trainer.resume_from_checkpoint(formal_path, expected_stage="finetune")


def test_default_save_checkpoint_blocks_overwrite_of_completed_checkpoint():
    """P1: Verify completed checkpoint overwrite is strictly blocked by default calls without allow_overwrite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        head_conf = {"type": "ArcFace", "num_class": 4, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 1. Save completed epoch 40 checkpoint
        ckpt_path = ckpt_mgr.save_checkpoint(
            filename="finetune_completed.pt",
            model=model,
            optimizer=opt,
            scheduler=None,
            epoch=40,
            extra_meta={"completed": True, "seed": 8972, "stage": "finetune"},
        )
        assert os.path.exists(ckpt_path)

        # 2. Default call (no allow_overwrite argument passed) with epoch 1 MUST raise FileExistsError
        with pytest.raises(FileExistsError, match="Completed checkpoint already exists"):
            ckpt_mgr.save_checkpoint(
                filename="finetune_completed.pt",
                model=model,
                optimizer=opt,
                scheduler=None,
                epoch=1,
                extra_meta={"completed": False, "seed": 8972, "stage": "finetune"},
            )

        # 3. Trainer.save_state_checkpoint with default arguments on new trainer MUST ALSO raise FileExistsError
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10], "lr_decay_ratio": 0.1},
            "project": {"seed": 8972},
        }
        new_trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
        )
        with pytest.raises(FileExistsError, match="Completed checkpoint already exists"):
            new_trainer.save_state_checkpoint("finetune_completed.pt", epoch=1)


def test_trainer_training_loop_and_save_state_coordination():
    """P1: Verify training loop's final save and notebook's save_state_checkpoint coordinate without FileExistsError collision."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        head_conf = {"type": "ArcFace", "num_class": 4, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(backbone_conf, head_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 1, "milestones": [10], "lr_decay_ratio": 0.1, "save_freq": 1},
            "project": {"seed": 8972},
        }
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
        )

        # 1. Loop saves final epoch 1 checkpoint (named model_test_epoch_1.pt)
        trainer.train(start_epoch=1, model_prefix="model_test")
        final_filename = "model_test_epoch_1.pt"
        final_path = os.path.join(tmp_dir, final_filename)
        assert os.path.exists(final_path)

        # 2. Notebook cell 5 calls save_state_checkpoint on the SAME file with default allow_overwrite=False
        # MUST NOT crash with FileExistsError, but coordinate and reuse verified path!
        coordinated_path = trainer.save_state_checkpoint(final_filename, epoch=1)
        assert coordinated_path == final_path

        # 3. A fresh trainer (simulating re-running the notebook or a different run) MUST be blocked
        fresh_model = build_model(backbone_conf, head_conf)
        fresh_trainer = Trainer(
            model=fresh_model,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
        )
        with pytest.raises(FileExistsError):
            fresh_trainer.save_state_checkpoint(final_filename, epoch=1)


def test_resume_rejects_altered_configurations_arcface_margin_and_backbone_dropout():
    """P2: Verify resume strictly rejects modified configurations (ArcFace margin 0.6 vs 0.3, Backbone dropout 0.6 vs 0.0)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)

        # Model A: Original training settings (UCFace: ArcFace margin=0.6, Dropout=0.6)
        bb_conf_a = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf_a = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model_a = build_model(bb_conf_a, hd_conf_a)
        opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

        ckpt_file = ckpt_mgr.save_checkpoint(
            filename="formal_run_m0.6_drop0.6.pt",
            model=model_a,
            optimizer=opt_a,
            scheduler=None,
            epoch=10,
            extra_meta={"stage": "finetune", "head": "ArcFace", "seed": 8972},
        )
        assert os.path.exists(ckpt_file)
        saved_data = load_checkpoint(ckpt_file)
        assert "config" in saved_data
        assert "config_hash" in saved_data
        assert saved_data["config"]["head"]["margin_arc"] == 0.6
        assert saved_data["config"]["backbone"]["dropout_rate"] == 0.6

        # Scenario 1: Resume model alters ArcFace margin from 0.6 to 0.3
        bb_conf_m1 = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf_m1 = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.3, "scale": 64.0}
        model_m1 = build_model(bb_conf_m1, hd_conf_m1)
        opt_m1 = torch.optim.Adam(model_m1.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="margin_arc mismatch: checkpoint=0.6, current=0.3"):
            ckpt_mgr.resume_checkpoint(ckpt_file, model=model_m1, optimizer=opt_m1, expected_seed=8972)

        # Scenario 2: Resume model alters Backbone dropout from 0.6 to 0.0
        bb_conf_m2 = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        hd_conf_m2 = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model_m2 = build_model(bb_conf_m2, hd_conf_m2)
        opt_m2 = torch.optim.Adam(model_m2.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="dropout_rate mismatch: checkpoint=0.6, current=0.0"):
            ckpt_mgr.resume_checkpoint(ckpt_file, model=model_m2, optimizer=opt_m2, expected_seed=8972)

        # Scenario 3: Resume model with both altered
        bb_conf_m3 = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        hd_conf_m3 = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.3, "scale": 64.0}
        model_m3 = build_model(bb_conf_m3, hd_conf_m3)
        opt_m3 = torch.optim.Adam(model_m3.parameters(), lr=1e-3)

        with pytest.raises(ValueError) as excinfo:
            ckpt_mgr.resume_checkpoint(ckpt_file, model=model_m3, optimizer=opt_m3, expected_seed=8972)
        assert "margin_arc mismatch" in str(excinfo.value)
        assert "dropout_rate mismatch" in str(excinfo.value)

        # Scenario 4: Exact matching configuration -> SUCCESS
        bb_conf_ok = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf_ok = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model_ok = build_model(bb_conf_ok, hd_conf_ok)
        opt_ok = torch.optim.Adam(model_ok.parameters(), lr=1e-3)

        start_ep = ckpt_mgr.resume_checkpoint(ckpt_file, model=model_ok, optimizer=opt_ok, expected_seed=8972)
        assert start_ep == 11


def test_checkpoint_embeds_config_and_config_hash():
    """P2: Verify Trainer.save_state_checkpoint embeds full training configuration and sha256 hash."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        backbone_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        head_conf = {"type": "AdaFace", "num_class": 10, "m": 0.4, "h": 0.333, "scale": 64.0, "t_alpha": 0.01}
        model = build_model(backbone_conf, head_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20], "lr_decay_ratio": 0.1},
            "dataset": {"name": "VGGFace2", "num_classes": 10, "input_size": [112, 112], "pixel_mean": 127.5, "pixel_std": 128.0},
            "project": {"seed": 8972},
        }
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="AdaFace",
            stage="finetune",
            parent_hash="phash_999",
        )

        saved_path = trainer.save_state_checkpoint("test_config_hash.pt", epoch=1)
        loaded = load_checkpoint(saved_path)

        assert "config" in loaded
        assert "config_hash" in loaded
        assert loaded["config"]["backbone"]["dropout_rate"] == 0.6
        assert loaded["config"]["head"]["m"] == 0.4
        assert loaded["config"]["head"]["h"] == 0.333
        assert loaded["config"]["seed"] == 8972
        assert loaded["config"]["parent_hash"] == "phash_999"
        assert loaded["config_hash"] == compute_config_hash(loaded["config"])


def test_resume_rejects_altered_batch_size():
    """P2: Verify resume rejects altered batch_size (e.g. 64 -> 32) and produces different config_hash."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        bb_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb_conf, hd_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(128, 3, 112, 112), torch.randint(0, 10, (128,)))
        train_loader_64 = torch.utils.data.DataLoader(train_ds, batch_size=64)

        config_64 = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20, 30], "lr_decay_ratio": 0.1, "batch_size": 64},
            "dataset": {"name": "VGGFace2", "num_classes": 10},
            "project": {"seed": 8972},
        }
        trainer_64 = Trainer(
            model=model,
            train_loader=train_loader_64,
            val_loader=None,
            config=config_64,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        saved_ckpt = trainer_64.save_state_checkpoint("formal_run_bs64.pt", epoch=1)

        # Attempt to resume with batch_size 32
        model_32 = build_model(bb_conf, hd_conf)
        train_loader_32 = torch.utils.data.DataLoader(train_ds, batch_size=32)
        config_32 = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20, 30], "lr_decay_ratio": 0.1, "batch_size": 32},
            "dataset": {"name": "VGGFace2", "num_classes": 10},
            "project": {"seed": 8972},
        }
        trainer_32 = Trainer(
            model=model_32,
            train_loader=train_loader_32,
            val_loader=None,
            config=config_32,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )

        assert trainer_64.config_hash != trainer_32.config_hash
        with pytest.raises(ValueError, match="batch_size mismatch: checkpoint=64, current=32"):
            trainer_32.resume_from_checkpoint(saved_ckpt)


def test_resume_rejects_altered_scheduler_milestones_and_decay_ratio():
    """P2: Verify resume rejects altered scheduler milestones ([10,20,30] -> [5,15]) and lr_decay_ratio (0.1 -> 0.5)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        bb_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb_conf, hd_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(16, 3, 112, 112), torch.randint(0, 10, (16,)))
        loader = torch.utils.data.DataLoader(train_ds, batch_size=4)

        orig_cfg = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20, 30], "lr_decay_ratio": 0.1, "batch_size": 4},
            "dataset": {"name": "VGGFace2", "num_classes": 10},
            "project": {"seed": 8972},
        }
        orig_trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            config=orig_cfg,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        saved_ckpt = orig_trainer.save_state_checkpoint("orig_sched.pt", epoch=1)

        # 1. Milestone alteration ([10, 20, 30] -> [5, 15])
        altered_ms_cfg = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [5, 15], "lr_decay_ratio": 0.1, "batch_size": 4},
            "dataset": {"name": "VGGFace2", "num_classes": 10},
            "project": {"seed": 8972},
        }
        trainer_ms = Trainer(
            model=build_model(bb_conf, hd_conf),
            train_loader=loader,
            val_loader=None,
            config=altered_ms_cfg,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        assert orig_trainer.config_hash != trainer_ms.config_hash
        with pytest.raises(ValueError, match="milestones mismatch: checkpoint=\\[10, 20, 30\\], current=\\[5, 15\\]"):
            trainer_ms.resume_from_checkpoint(saved_ckpt)

        # 2. Decay ratio alteration (0.1 -> 0.5)
        altered_decay_cfg = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20, 30], "lr_decay_ratio": 0.5, "batch_size": 4},
            "dataset": {"name": "VGGFace2", "num_classes": 10},
            "project": {"seed": 8972},
        }
        trainer_decay = Trainer(
            model=build_model(bb_conf, hd_conf),
            train_loader=loader,
            val_loader=None,
            config=altered_decay_cfg,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        assert orig_trainer.config_hash != trainer_decay.config_hash
        with pytest.raises(ValueError, match="lr_decay_ratio mismatch: checkpoint=0.1, current=0.5"):
            trainer_decay.resume_from_checkpoint(saved_ckpt)


def test_resume_rejects_altered_dataset_index_contents_in_place():
    """P2: Verify resume rejects altered dataset index file content even when file path remains identical."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_list_file = os.path.join(tmp_dir, "train_list.txt")
        with open(train_list_file, "w", encoding="utf-8") as f:
            f.write("identity_001/img_01.jpg 0\nidentity_001/img_02.jpg 0\n")

        initial_hash = compute_sha256(train_list_file)

        ckpt_mgr = CheckpointManager(tmp_dir)
        bb_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb_conf, hd_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(8, 3, 112, 112), torch.randint(0, 10, (8,)))
        loader = torch.utils.data.DataLoader(train_ds, batch_size=4)

        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10, 20, 30], "lr_decay_ratio": 0.1, "batch_size": 4},
            "dataset": {"name": "VGGFace2", "num_classes": 10, "train_list": train_list_file, "data_root": tmp_dir},
            "project": {"seed": 8972},
        }
        trainer_orig = Trainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        assert trainer_orig.train_config["dataset"]["train_list_hash"] == initial_hash
        saved_ckpt = trainer_orig.save_state_checkpoint("ckpt_with_dataset_hash.pt", epoch=1)

        # Alter file contents in-place at the exact same file path
        with open(train_list_file, "w", encoding="utf-8") as f:
            f.write("identity_002/img_99.jpg 1\nidentity_003/img_88.jpg 2\n")

        new_hash = compute_sha256(train_list_file)
        assert new_hash != initial_hash

        trainer_resumed = Trainer(
            model=build_model(bb_conf, hd_conf),
            train_loader=loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        assert trainer_resumed.train_config["dataset"]["train_list_hash"] == new_hash
        assert trainer_orig.config_hash != trainer_resumed.config_hash

        with pytest.raises(ValueError, match="train_list_hash mismatch"):
            trainer_resumed.resume_from_checkpoint(saved_ckpt)


def test_resume_missing_config_fail_closed_and_unverified_bypass():
    """P2: Verify checkpoints lacking config or config_hash fail-closed by default, but allow unverified bypass when specified."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        bb_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb_conf, hd_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
        config = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10], "lr_decay_ratio": 0.1},
            "project": {"seed": 8972},
        }
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        orig_ckpt_path = trainer.save_state_checkpoint("formal_base.pt", epoch=2)
        assert os.path.exists(orig_ckpt_path)

        # 1. Create corrupted checkpoint lacking both 'config' and 'config_hash'
        data_no_config = load_checkpoint(orig_ckpt_path)
        data_no_config.pop("config", None)
        data_no_config.pop("config_hash", None)
        if "extra_meta" in data_no_config:
            data_no_config["extra_meta"].pop("config", None)
            data_no_config["extra_meta"].pop("config_hash", None)
        path_no_config = os.path.join(tmp_dir, "legacy_no_config.pt")
        torch.save(data_no_config, path_no_config)

        # Fail-closed by default (strict_config=True)
        resumed_trainer = Trainer(
            model=build_model(bb_conf, hd_conf),
            train_loader=loader,
            val_loader=None,
            config=config,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        with pytest.raises(ValueError, match="lacks required configuration metadata \\('config'\\)"):
            resumed_trainer.resume_from_checkpoint(path_no_config)

        # Direct CheckpointManager call also fails closed by default
        with pytest.raises(ValueError, match="lacks required configuration metadata \\('config'\\)"):
            ckpt_mgr.resume_checkpoint(path_no_config, model=resumed_trainer.model, optimizer=resumed_trainer.optimizer)

        # Explicit allow_unverified_resume=True succeeds with warning
        epoch_resumed = resumed_trainer.resume_from_checkpoint(path_no_config, allow_unverified_resume=True)
        assert epoch_resumed == 3

        # 2. Create checkpoint with 'config' present but 'config_hash' missing
        data_no_hash = load_checkpoint(orig_ckpt_path)
        data_no_hash.pop("config_hash", None)
        if "extra_meta" in data_no_hash:
            data_no_hash["extra_meta"].pop("config_hash", None)
        if "config" in data_no_hash and isinstance(data_no_hash["config"], dict):
            data_no_hash["config"].pop("config_hash", None)
        path_no_hash = os.path.join(tmp_dir, "corrupt_no_hash.pt")
        torch.save(data_no_hash, path_no_hash)

        with pytest.raises(ValueError, match="lacks required config hash metadata \\('config_hash'\\)"):
            resumed_trainer.resume_from_checkpoint(path_no_hash)

        epoch_resumed_2 = resumed_trainer.resume_from_checkpoint(path_no_hash, strict_config=False)
        assert epoch_resumed_2 == 3


def test_resume_rejects_altered_data_root():
    """P2: Verify resume rejects altered dataset data_root path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_a = os.path.join(tmp_dir, "vgg_a")
        dir_b = os.path.join(tmp_dir, "vgg_b")
        os.makedirs(dir_a, exist_ok=True)
        os.makedirs(dir_b, exist_ok=True)

        ckpt_mgr = CheckpointManager(tmp_dir)
        bb_conf = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd_conf = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb_conf, hd_conf)
        train_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 112, 112), torch.tensor([0, 1, 2, 3]))
        loader = torch.utils.data.DataLoader(train_ds, batch_size=2)

        config_a = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10], "lr_decay_ratio": 0.1},
            "dataset": {"name": "VGGFace2", "num_classes": 10, "data_root": dir_a},
            "project": {"seed": 8972},
        }
        trainer_a = Trainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            config=config_a,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        saved_ckpt = trainer_a.save_state_checkpoint("ckpt_dir_a.pt", epoch=1)

        config_b = {
            "training": {"optimizer": "adam", "lr": 0.001, "epoches": 40, "milestones": [10], "lr_decay_ratio": 0.1},
            "dataset": {"name": "VGGFace2", "num_classes": 10, "data_root": dir_b},
            "project": {"seed": 8972},
        }
        trainer_b = Trainer(
            model=build_model(bb_conf, hd_conf),
            train_loader=loader,
            val_loader=None,
            config=config_b,
            checkpoint_manager=ckpt_mgr,
            device=torch.device("cpu"),
            head_type="ArcFace",
            stage="finetune",
            parent_hash="phash_123",
        )
        assert trainer_a.config_hash != trainer_b.config_hash
        with pytest.raises(ValueError, match="data_root mismatch"):
            trainer_b.resume_from_checkpoint(saved_ckpt)


def test_extract_train_config_and_loader_binding():
    """P2: Verify extract_train_config accurately binds loader batch_size and computes index file hashes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_file = os.path.join(tmp_dir, "train.txt")
        val_file = os.path.join(tmp_dir, "val.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("img1.jpg 0\nimg2.jpg 1\n")
        with open(val_file, "w", encoding="utf-8") as f:
            f.write("val1.jpg 0\n")

        train_ds = torch.utils.data.TensorDataset(torch.randn(8, 3, 112, 112), torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

        config = {
            "training": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0005, "epoches": 50, "milestones": [20, 40], "lr_decay_ratio": 0.2, "batch_size": 64},
            "dataset": {"name": "VGGFace2", "num_classes": 100, "data_root": tmp_dir, "train_list": train_file, "val_list": val_file},
            "project": {"seed": 42},
        }

        # extract_train_config should prefer live loader batch_size (32) over config (64)
        extracted = extract_train_config(
            config=config,
            train_loader=train_loader,
            seed=42,
            stage="pretrain",
        )
        assert extracted["training"]["batch_size"] == 32
        assert extracted["training"]["milestones"] == [20, 40]
        assert extracted["training"]["lr_decay_ratio"] == 0.2
        assert extracted["training"]["optimizer"] == "sgd"
        assert extracted["dataset"]["train_list_hash"] == compute_sha256(train_file)
        assert extracted["dataset"]["val_list_hash"] == compute_sha256(val_file)
        assert extracted["seed"] == 42
        assert extracted["stage"] == "pretrain"


def test_resume_rejects_empty_config_dict_fail_closed():
    """P2: Verify checkpoint with empty config dict {} is rejected (fail-closed) and not bypassed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        bb = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb, hd)
        pt = ckpt_mgr.save_checkpoint("empty_cfg.pt", model, None, None, 1, extra_meta={"config": {}, "config_hash": "dummy_hash"})

        # Fail-closed by default
        with pytest.raises(ValueError, match="lacks required configuration metadata \\('config'\\)"):
            ckpt_mgr.resume_checkpoint(pt, model)

        # Allow unverified bypass when explicitly requested
        resumed_epoch = ckpt_mgr.resume_checkpoint(pt, model, allow_unverified_resume=True)
        assert resumed_epoch == 2


def test_resume_rejects_internal_config_hash_tampering():
    """P2: Verify checkpoint internal config vs config_hash tampering is detected and rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        bb = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb, hd)
        pt = ckpt_mgr.save_checkpoint("valid.pt", model, None, None, 1)

        # Tamper with internal config without updating config_hash
        data = load_checkpoint(pt)
        data["config"]["backbone"]["dropout_rate"] = 0.0
        tampered_pt = os.path.join(tmp_dir, "tampered.pt")
        torch.save(data, tampered_pt)

        with pytest.raises(ValueError, match="Checkpoint configuration internal hash mismatch"):
            ckpt_mgr.resume_checkpoint(tampered_pt, model)


def test_resume_rejects_empty_or_stripped_expected_config():
    """P2: Verify passing empty expected_config={} against a formal checkpoint is rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        bb = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.6}
        hd = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb, hd)
        cfg = {"training": {"batch_size": 32, "milestones": [10], "lr_decay_ratio": 0.1}, "dataset": {"name": "vgg"}}
        c = extract_model_and_train_config(model, config=cfg)
        pt = ckpt_mgr.save_checkpoint("formal_run.pt", model, None, None, 1, extra_meta={"config": c, "config_hash": compute_config_hash(c)})

        # Empty expected_config must NOT silently bypass validation
        with pytest.raises(ValueError):
            ckpt_mgr.resume_checkpoint(pt, model, expected_config={})


def test_direct_resume_checkpoint_rejects_altered_scheduler_and_optimizer():
    """P2: Verify direct CheckpointManager.resume_checkpoint rejects scheduler milestones & lr_decay_ratio mismatch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(tmp_dir)
        bb = {"type": "MobileFaceNet", "feat_dim": 1024, "out_h": 7, "out_w": 7, "dropout_rate": 0.0}
        hd = {"type": "ArcFace", "num_class": 10, "margin_arc": 0.6, "scale": 64.0}
        model = build_model(bb, hd)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 20], gamma=0.1)
        pt = ckpt_mgr.save_checkpoint("base_train.pt", model, opt, sched, 1)

        # Altered scheduler milestones [10, 20] -> [5, 15] and gamma 0.1 -> 0.5
        opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[5, 15], gamma=0.5)
        with pytest.raises(ValueError, match="milestones mismatch|lr_decay_ratio mismatch"):
            ckpt_mgr.resume_checkpoint(pt, model, opt2, sched2)


def test_extract_train_config_with_relative_index_path_and_unwrapped_dataset():
    """P2: Verify extract_train_config resolves relative index files and unwraps Subset datasets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_file = os.path.join(tmp_dir, "train_rel.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("img1.jpg 0\n")

        # Create FaceImageDataset and wrap in Subset
        base_ds = FaceImageDataset(data_root=tmp_dir, index_file=train_file, is_train=True)
        subset_ds = torch.utils.data.Subset(base_ds, indices=[0])
        loader = torch.utils.data.DataLoader(subset_ds, batch_size=16)

        # Config specifies relative path 'train_rel.txt'
        cfg = {"dataset": {"data_root": tmp_dir, "train_list": "train_rel.txt"}}
        extracted = extract_train_config(config=cfg, train_loader=loader)

        # Resolved and computed hash must match file
        assert extracted["dataset"]["train_list_hash"] == compute_sha256(train_file)
        assert extracted["preprocessing"]["pixel_mean"] == 127.5
        assert extracted["preprocessing"]["pixel_std"] == 128.0
        assert extracted["training"]["batch_size"] == 16





