"""Publication failures must recover without rewriting surviving model bytes."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import checkpoint_manager as checkpoints
from modules.backbone_factory import build_backbone
from modules.shared_storage import artifact_lock


@pytest.fixture
def export(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = dict(type="MobileFaceNet", feat_dim=8, out_h=7, out_w=7, dropout_rate=0.)
    model = build_backbone(config)
    metadata = dict(is_smoke=False, completed=True, source_sha256="source-A",
                    source_config=dict(backbone=config), export_data_verification=dict(timestamp="first"))
    manager = checkpoints.CheckpointManager(str(tmp_path))
    yield manager, model, metadata, tmp_path / "model.pt"
    torch.set_num_threads(old_threads)


def interrupt(export, monkeypatch):
    manager, model, metadata, path = export
    replace = checkpoints.os.replace
    def fail_sidecar(src, dst):
        if str(dst).endswith("_meta.json"):
            raise OSError("injected metadata publication failure")
        return replace(src, dst)
    with monkeypatch.context() as patch:
        patch.setattr(checkpoints.os, "replace", fail_sidecar)
        with pytest.raises(OSError, match="injected"):
            manager.export_backbone_only(model, path.name, metadata, target_feat_dim=8)
    assert path.exists() and not path.with_name("model_meta.json").exists()
    assert not list(path.parent.glob("*.tmp"))
    return path.read_bytes(), path.stat().st_mtime_ns


def test_retry_recovers_only_sidecar_and_completed_pair_remains_immutable(export, monkeypatch):
    manager, model, metadata, path = export
    original_metadata = copy.deepcopy(metadata)
    before = interrupt(export, monkeypatch)
    retry = copy.deepcopy(metadata)
    retry["export_data_verification"]["timestamp"] = "second"
    manager.export_backbone_only(model, path.name, retry, target_feat_dim=8)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    sidecar = json.loads(path.with_name("model_meta.json").read_text())
    assert sidecar["sha256"] == checkpoints.compute_sha256(str(path))
    assert sidecar["export_data_verification"]["timestamp"] == "first"
    assert metadata == original_metadata
    with pytest.raises(FileExistsError):
        manager.export_backbone_only(model, path.name, metadata, target_feat_dim=8)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("change", ["weights", "source", "missing_embedded_metadata"])
def test_retry_rejects_different_weights_or_provenance(export, monkeypatch, change):
    manager, model, metadata, path = export
    interrupt(export, monkeypatch)
    if change == "weights":
        with torch.no_grad():
            next(model.parameters()).add_(.001)
    elif change == "source":
        metadata["source_sha256"] = "source-B"
    else:
        payload = checkpoints.load_checkpoint(str(path))
        payload.pop("metadata")
        torch.save(payload, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Interrupted export"):
        manager.export_backbone_only(model, path.name, metadata, target_feat_dim=8)
    assert path.read_bytes() == before
    assert not path.with_name("model_meta.json").exists()


def test_export_lock_blocks_overlapping_writer(export):
    manager, model, metadata, path = export
    with artifact_lock(path), pytest.raises(FileExistsError):
        manager.export_backbone_only(model, path.name, metadata, target_feat_dim=8)
    assert not path.exists()


def test_orphan_sidecar_is_preserved_and_rejected(export):
    manager, model, metadata, path = export
    sidecar = path.with_name("model_meta.json")
    sidecar.write_text("{}")
    with pytest.raises(FileExistsError):
        manager.export_backbone_only(model, path.name, metadata, target_feat_dim=8)
    assert not path.exists() and sidecar.read_text() == "{}"
