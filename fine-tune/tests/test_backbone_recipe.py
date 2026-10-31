"""Backbone aliases and stage-specific LR without formal training."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fine-tune"))
from modules import notebook_workflow as flow
from modules.backbone_factory import canonical_backbone_type, backbone_slug
from modules.checkpoint_manager import compare_configs


@pytest.mark.parametrize("name", ["Resnet", "ResNet", "resnet", "RESNET", " rEsNeT "])
@pytest.mark.parametrize("depth", [50, 100])
def test_resnet_recipe_roundtrip_and_parent_lr(name, depth, tmp_path):
    cfg = flow.resolve_settings(ROOT, overrides={"backbone": {"type": name, "num_layers": depth}})
    assert cfg["backbone"]["type"] == "Resnet"
    assert cfg["training"]["batch_size"] == 64
    assert cfg["training"]["lr"] == 0.0001
    assert flow.stage_settings(cfg, "pretrain")["training"]["lr"] == 0.001
    assert flow.stage_settings(cfg, "finetune")["training"]["lr"] == 0.0001
    flow.validate_paper_parameters(cfg)
    path = tmp_path / "settings.json"
    flow.save_settings(path, cfg)
    assert flow.resolve_settings(ROOT, path) == cfg
    raw = copy.deepcopy(cfg); raw["backbone"]["type"] = name
    flow.validate_settings(raw)
    flow.validate_paper_parameters(raw)
    assert backbone_slug(raw["backbone"]) == f"resnet{depth}_ir"
    assert raw["backbone"]["type"] == name


@pytest.mark.parametrize("name", ["MobileFaceNet", "mobilefacenet", "MOBILEFACENET", " mObIlEfAcEnEt "])
def test_mobile_recipe_stays_unchanged(name):
    cfg = flow.resolve_settings(ROOT, overrides={"backbone": {"type": name}})
    assert cfg["backbone"]["type"] == "MobileFaceNet"
    assert cfg["training"]["lr"] == 0.001
    assert backbone_slug(cfg["backbone"]) == "mobilefacenet"
    flow.validate_paper_parameters(cfg)


def test_stale_saved_resnet_lr_is_preserved_but_rejected(tmp_path):
    cfg = flow.resolve_settings(ROOT, overrides={"backbone": {"type": "RESNET"}})
    cfg["training"]["lr"] = 0.001
    cfg["campaign"].pop("pretrain_lr")  # legacy settings
    path = tmp_path / "old.json"; path.write_text(json.dumps(cfg))
    loaded = flow.resolve_settings(ROOT, path)
    assert loaded["training"]["lr"] == 0.001
    assert loaded["campaign"]["pretrain_lr"] == 0.001
    with pytest.raises(ValueError, match="training.lr=0.0001"):
        flow.validate_paper_parameters(loaded)
    corrected = flow.resolve_settings(ROOT, path, overrides={"training": {"lr": 0.0001}})
    assert corrected["campaign"]["pretrain_lr"] == 0.001
    assert any("Optimizer lr" in message for message in compare_configs(loaded, corrected))


@pytest.mark.parametrize("name,depth", [("MobileFaceNet", None), ("ResNet", 50), ("resnet", 100)])
def test_saved_small_batch_is_preserved_but_rejected_for_formal_baseline(name, depth, tmp_path):
    backbone = {"type": name}
    if depth is not None:
        backbone["num_layers"] = depth
    cfg = flow.resolve_settings(ROOT, overrides={"backbone": backbone, "training": {"batch_size": 16}})
    path = tmp_path / "batch16.json"
    flow.save_settings(path, cfg)
    original = path.read_bytes()
    loaded = flow.resolve_settings(ROOT, path)
    assert loaded["training"]["batch_size"] == 16
    with pytest.raises(ValueError, match="training.batch_size=64"):
        flow.validate_paper_parameters(loaded)
    corrected = flow.resolve_settings(ROOT, path, overrides={"training": {"batch_size": 64}})
    flow.validate_paper_parameters(corrected)
    assert any("batch_size mismatch" in message for message in compare_configs(loaded, corrected))
    assert path.read_bytes() == original


@pytest.mark.parametrize("name", ["resnnet", "", None, 10])
def test_unknown_names_fail(name):
    with pytest.raises(ValueError, match="Unsupported backbone"):
        canonical_backbone_type(name)


def test_matrix_records_effective_parent_lr(monkeypatch):
    cfg = flow.resolve_settings(ROOT, overrides={"backbone": {"type": "resnet"}, "campaign": {"pretrain_lr": 0.0005}})
    monkeypatch.setattr(flow, "_matrix", lambda effective, stage, operation: effective)
    result = flow.run_training_matrix(cfg, "pretrain")
    assert result["training"]["lr"] == 0.0005
    assert cfg["training"]["lr"] == 0.0001


def test_builder_aliases_dispatch_without_large_models(monkeypatch):
    from modules import backbone_factory as factory
    from types import SimpleNamespace
    monkeypatch.setattr(factory, "MobileFaceNet", lambda **kw: ("mobile", kw))
    monkeypatch.setattr(factory, "Resnet", lambda *args, **kw: SimpleNamespace(args=args))
    assert factory.build_backbone({"type": "MOBILEFACENET"})[0] == "mobile"
    assert factory.build_backbone({"type": "RESNET", "num_layers": 100}).num_layers == 100
