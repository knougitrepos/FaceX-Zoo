"""CosFace math, provenance and notebook opt-in, using CPU and temporary outputs."""
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fine-tune"))
from modules import notebook_workflow as flow
from modules.model_builder import build_model
from modules.checkpoint_manager import (
    extract_model_config, compute_config_hash, compare_configs,
)
from head.AM_Softmax import AM_Softmax


def test_cosface_formula_backbone_gradients_and_inference():
    torch.set_num_threads(1)
    torch.manual_seed(8972)
    cfg = flow.resolve_settings(ROOT)
    model = build_model(cfg["backbone"], dict(cfg["heads"]["cosface"], num_class=9))
    assert isinstance(model.head, AM_Softmax)
    assert model.head_type == "CosFace"
    images = torch.randn(4, 3, 112, 112)
    labels = torch.tensor([0, 1, 3, 8])
    features = model.backbone(images)
    cosine = (F.normalize(features, dim=1) @ F.normalize(model.head.weight, dim=0)).clamp(-1, 1)
    expected = 64.0 * (cosine - 0.3 * F.one_hot(labels, num_classes=9))
    actual = model.head(features, labels)
    torch.testing.assert_close(actual, expected)
    # All non-target logits remain s*cos(theta); targets lose exactly s*m.
    torch.testing.assert_close((64.0 * cosine - actual).gather(1, labels[:, None]), torch.full((4, 1), 19.2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.backbone.linear.weight.detach().clone()
    F.cross_entropy(actual, labels).backward()
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.backbone.linear.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.backbone.linear.weight)
    model.eval()
    with torch.no_grad():
        logits, auxiliary = model(images, labels)
        embedding, raw_norm = model.extract_features(images)
    assert logits.shape == (4, 9) and auxiliary is None
    assert embedding.shape == (4, 1024) and raw_norm.shape == (4, 1)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(4))


def test_cosface_margin_is_hashed_and_compared():
    cfg = flow.resolve_settings(ROOT)
    model = build_model(cfg["backbone"], dict(cfg["heads"]["cosface"], num_class=9))
    recorded = extract_model_config(model)
    assert recorded["head"] == {"type": "AM_Softmax", "feat_dim": 1024,
                                "num_class": 9, "margin": 0.3, "scale": 64.0}
    model.head.margin = 0.35
    changed = extract_model_config(model)
    assert compute_config_hash(recorded) != compute_config_hash(changed)
    assert any("Head margin mismatch" in item for item in compare_configs(recorded, changed, full_strict=True))


@pytest.mark.parametrize("name,stage", [
    ("03_finetune.ipynb", "finetune"), ("04_export_and_validate.ipynb", "export"),
])
def test_notebook_adds_cosface_over_legacy_saved_settings(name, stage, tmp_path, monkeypatch):
    # Do not mutate the user's saved settings or launch any training/audit/export.
    from modules import process_runner, finetune_protocol
    def forbidden(*args, **kwargs):
        raise AssertionError("Planned notebook must not launch or prepare datasets")
    monkeypatch.setattr(process_runner, "launch_campaign", forbidden)
    monkeypatch.setattr(finetune_protocol, "prepare_selected_protocols", forbidden)
    saved = flow.resolve_settings(ROOT)
    saved["campaign"]["heads"] = ["arcface", "adaface", "magface"]
    saved["heads"].pop("cosface")
    saved_path = tmp_path / "legacy_settings.json"
    flow.save_settings(saved_path, saved)
    original_bytes = saved_path.read_bytes()
    notebook = json.loads((ROOT / "fine-tune/notebooks" / name).read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).replace(
            'SETTINGS_PATH = BASE_DIR / "fine-tune/data/notebook_settings.json"',
            f"SETTINGS_PATH = Path({str(saved_path)!r})",
        )
        exec(compile(source, name, "exec"), namespace)
    assert namespace["EXECUTE"] is False
    assert len(namespace["plan"]) == 40
    assert {row["head"] for row in namespace["plan"]} == set(flow.HEAD_NAMES)
    assert {row["stage"] for row in namespace["plan"]} == {stage}
    assert namespace["settings"]["heads"]["cosface"]["margin"] == 0.3
    assert namespace["settings"]["heads"]["cosface"]["scale"] == 64.0
    assert saved_path.read_bytes() == original_bytes
    legacy = flow.resolve_settings(ROOT, saved_path)
    assert legacy["campaign"]["heads"] == saved["campaign"]["heads"]
    assert flow.plan_matrix(legacy, "pretrain") == flow.plan_matrix(namespace["settings"], "pretrain")
