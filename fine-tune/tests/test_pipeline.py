"""
Integration test for the full FaceX-Zoo & UCFace pipeline.
Tests model construction, forward-backward step, checkpointing, parent hash verification, and export.
"""

import sys
import os
import shutil
import tempfile
import pytest
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINETUNE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULES_DIR = os.path.abspath(os.path.join(FINETUNE_DIR, "modules"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if FINETUNE_DIR not in sys.path:
    sys.path.insert(0, FINETUNE_DIR)
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from model_builder import build_model
from checkpoint_manager import CheckpointManager, compute_sha256, verify_parent_checkpoint, load_checkpoint
from backbone.MobileFaceNets import MobileFaceNet


def test_full_pipeline_workflow():
    """Execute complete workflow: build -> step -> save -> resume -> export -> verify."""
    temp_dir = tempfile.mkdtemp(prefix="test_pipeline_")
    try:
        # 1. Build Model: MobileFaceNet 1024D + AdaFace
        backbone_conf = {
            "type": "MobileFaceNet",
            "feat_dim": 1024,
            "out_h": 7,
            "out_w": 7,
            "dropout_rate": 0.6,
        }
        head_conf = {
            "type": "AdaFace",
            "num_class": 50,
            "m": 0.4,
            "h": 0.333,
            "scale": 64.0,
            "t_alpha": 0.01,
        }
        model = build_model(backbone_conf, head_conf)
        model.train()

        # 2. Forward pass with fake images [B=4, C=3, H=112, W=112]
        images = torch.randn(4, 3, 112, 112)
        labels = torch.tensor([0, 10, 25, 49])

        logits, loss_g = model(images, labels)
        assert logits.shape == (4, 50)
        assert loss_g is None

        # Optimizer and Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 3. Save full training checkpoint
        ckpt_mgr = CheckpointManager(temp_dir)
        saved_ckpt = ckpt_mgr.save_checkpoint(
            filename="test_epoch_1.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            extra_meta={"loss": loss.item()},
        )
        assert os.path.exists(saved_ckpt)

        # 4. Resume check
        model_resumed = build_model(backbone_conf, head_conf)
        opt_resumed = torch.optim.Adam(model_resumed.parameters(), lr=1e-3)
        sched_resumed = torch.optim.lr_scheduler.MultiStepLR(opt_resumed, milestones=[10, 20], gamma=0.1)

        next_epoch = ckpt_mgr.resume_checkpoint(
            checkpoint_path=saved_ckpt,
            model=model_resumed,
            optimizer=opt_resumed,
            scheduler=sched_resumed,
        )
        assert next_epoch == 2

        # 5. Export clean inference backbone
        metadata = {
            "model_name": "MobileFaceNet_1024D_AdaFace",
            "feat_dim": 1024,
            "head_type": "AdaFace",
            "epoch": 1,
        }
        export_pt = ckpt_mgr.export_backbone_only(
            model=model,
            output_filename="exported_mobilefacenet_1024d.pt",
            metadata=metadata,
        )
        assert os.path.exists(export_pt)
        assert os.path.exists(os.path.splitext(export_pt)[0] + "_meta.json")

        # 6. Verify exported checkpoint loads cleanly into standalone MobileFaceNet (ronbun format)
        exported_data = load_checkpoint(export_pt, map_location="cpu")
        standalone_backbone = MobileFaceNet(embedding_size=1024, out_h=7, out_w=7, dropout_rate=0.0)
        standalone_backbone.load_state_dict(exported_data["state_dict"], strict=True)
        standalone_backbone.eval()

        # Also verify direct torch.load with weights_only=False works in PyTorch 2.6+
        try:
            direct_data = torch.load(export_pt, map_location="cpu", weights_only=False)
            assert "state_dict" in direct_data
        except TypeError:
            pass  # Older PyTorch without weights_only argument

        # Compare inference embeddings between exported backbone and original model
        model.eval()
        with torch.no_grad():
            emb_orig, norm_orig = model.extract_features(images)
            raw_feats = standalone_backbone(images)
            norm_exp = torch.norm(raw_feats, p=2, dim=1, keepdim=True)
            emb_exp = torch.div(raw_feats, norm_exp)

        assert torch.allclose(emb_orig, emb_exp, atol=1e-5)
        assert torch.allclose(norm_orig, norm_exp, atol=1e-5)

        # 7. Test Parent Hash Verification contract
        # (a) compute hash
        parent_hash = compute_sha256(export_pt)
        # (b) verify successfully with matching hash
        loaded_parent, actual_hash = verify_parent_checkpoint(export_pt, expected_hash=parent_hash)
        assert actual_hash == parent_hash
        # (c) verify failure with mismatched hash
        with pytest.raises(ValueError):
            verify_parent_checkpoint(export_pt, expected_hash="invalid_hash_value_123")
        # (d) verify failure with non-existent path
        with pytest.raises(FileNotFoundError):
            verify_parent_checkpoint("non_existent_file.pt")
        # (e) verify load_checkpoint failure with non-existent path
        with pytest.raises(FileNotFoundError):
            load_checkpoint("non_existent_file.pt")

        # 8. Test export_backbone_only on standalone backbone (non-composite module)
        standalone_export = ckpt_mgr.export_backbone_only(
            model=standalone_backbone,
            output_filename="standalone_export.pt",
            metadata={"source": "standalone"},
        )
        assert os.path.exists(standalone_export)
        standalone_loaded = load_checkpoint(standalone_export)
        assert "state_dict" in standalone_loaded

        # 9. Verify exported checkpoint loads cleanly under strict weights_only=True (ronbun official_loaders requirement)
        try:
            ronbun_style_load = torch.load(export_pt, map_location="cpu", weights_only=True)
            assert "state_dict" in ronbun_style_load
        except TypeError:
            pass  # Older PyTorch without weights_only argument

        # 10. Test corrupted checkpoint file handling
        corrupt_file = os.path.join(temp_dir, "corrupt.pt")
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_DATA")
        with pytest.raises(Exception):
            load_checkpoint(corrupt_file)

        # 11. Test dataset CSV parsing with quoted names and commas
        from dataset import VGGFace2ArchiveInspector
        meta_dir = os.path.join(temp_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        csv_file = os.path.join(meta_dir, "identity_meta.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write('Class_ID, Name, Sample_Num, Flag, Gender\n')
            f.write('n000001, "Dalai, Lama", 424, 0, m\n')
            f.write('n000002, "Fine, Frenzy", 315, 1, f\n')
            f.write('n000003, "Gill, A. A.", 205, 1, m\n')
        inspector = VGGFace2ArchiveInspector(temp_dir)
        parsed_train = inspector.parse_identity_meta(train_only=True)
        assert len(parsed_train) == 2
        assert parsed_train[0]["class_id"] == "n000002"
        assert parsed_train[0]["name"] == "Fine, Frenzy"

        # 12. Test evaluator Feature Norm Diagnostics (and alias)
        from evaluator import FeatureEvaluator
        norms_test = norm_exp.cpu().numpy()
        diag = FeatureEvaluator.compute_feature_norm_diagnostics(norms_test)
        assert "mean_norm" in diag and diag["mean_norm"] > 0
        assert "median_norm" in diag
        fiqa_alias = FeatureEvaluator.compute_continuous_fiqa(norms_test)
        assert fiqa_alias["mean_norm"] == diag["mean_norm"]

        # 13. Test missing image exception (fail-closed, no zero vector fallback)
        from dataset import FaceImageDataset, validate_dataset_files
        fake_index = os.path.join(temp_dir, "fake_index.txt")
        with open(fake_index, "w") as f:
            f.write("non_existent_img.jpg 0\n")
        bad_ds = FaceImageDataset(data_root=temp_dir, index_file=fake_index, is_train=False)
        with pytest.raises(FileNotFoundError):
            _ = bad_ds[0]

        # 14. Test validate_dataset_files
        checked, missing = validate_dataset_files(data_root=temp_dir, index_file=fake_index)
        assert checked == 1
        assert len(missing) == 1

        # 15. Test verify_parent_checkpoint rejects smoke checkpoints
        smoke_parent = os.path.join(temp_dir, "smoke_pretrain_mobilefacenet_1024d.pt")
        torch.save({"backbone_state_dict": {}}, smoke_parent)
        smoke_hash = compute_sha256(smoke_parent)
        with pytest.raises(ValueError, match="Smoke-test checkpoint cannot be used"):
            verify_parent_checkpoint(smoke_parent, expected_hash=smoke_hash, require_hash=True)

        # 16. Test verify_parent_checkpoint requires expected_hash when require_hash=True
        with pytest.raises(ValueError, match="expected_hash is required"):
            verify_parent_checkpoint(export_pt, expected_hash=None, require_hash=True)

        # 17. Test module-level import of Trainer without relative import errors
        import sys
        if MODULES_DIR in sys.path:
            from modules.trainer import Trainer, set_seed
            assert callable(Trainer)
            assert callable(set_seed)

        # 18. Verify parse_identity_meta on actual VGGFace2 meta file if present
        actual_meta_csv = "F:/FaceX-Zoo/VGGFace2/meta/identity_meta.csv"
        if os.path.exists(actual_meta_csv):
            real_inspector = VGGFace2ArchiveInspector("F:/FaceX-Zoo/VGGFace2")
            real_train = real_inspector.parse_identity_meta(train_only=True)
            assert len(real_train) == 8631, f"Expected 8631 train identities, got {len(real_train)}"
            real_all = real_inspector.parse_identity_meta(train_only=False)
            assert len(real_all) == 9131, f"Expected 9131 total identities, got {len(real_all)}"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
