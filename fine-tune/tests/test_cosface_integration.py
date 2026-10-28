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
@pytest.mark.parametrize("heads", [["arcface"], ["arcface", "adaface", "magface"], ["cosface"]])
def test_notebook_preserves_explicit_saved_heads(name, stage, heads, tmp_path, monkeypatch):
    # Do not mutate the user's saved settings or launch any training/audit/export.
    from modules import process_runner, finetune_protocol
    def forbidden(*args, **kwargs):
        raise AssertionError("Planned notebook must not launch or prepare datasets")
    monkeypatch.setattr(process_runner, "launch_campaign", forbidden)
    monkeypatch.setattr(finetune_protocol, "prepare_selected_protocols", forbidden)
    saved = flow.resolve_settings(ROOT)
    saved["campaign"]["heads"] = heads
    saved["heads"].pop("cosface")
    saved_path = tmp_path / "legacy_settings.json"
    flow.save_settings(saved_path, saved)
    original_bytes = saved_path.read_bytes()
    notebook = json.loads((ROOT / "fine-tune/notebooks" / name).read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        import re
        source = re.sub(r'^SETTINGS_PATH = .*$',
                        lambda _: f"SETTINGS_PATH = Path({str(saved_path)!r})",
                        "".join(cell["source"]), flags=re.MULTILINE)
        exec(compile(source, name, "exec"), namespace)
    assert namespace["EXECUTE"] is False
    assert namespace["SETTINGS_PATH"] == saved_path
    assert len(namespace["plan"]) == len(heads) * len(saved["project"]["seed_list"]) * len(saved["campaign"]["finetune_datasets"])
    assert {row["head"] for row in namespace["plan"]} == set(heads)
    assert {row["stage"] for row in namespace["plan"]} == {stage}
    assert namespace["settings"]["heads"]["cosface"]["margin"] == 0.3
    assert namespace["settings"]["heads"]["cosface"]["scale"] == 64.0
    assert saved_path.read_bytes() == original_bytes
    legacy = flow.resolve_settings(ROOT, saved_path)
    assert legacy["campaign"]["heads"] == saved["campaign"]["heads"]
    assert flow.plan_matrix(legacy, "pretrain") == flow.plan_matrix(namespace["settings"], "pretrain")


def test_new_campaign_notebooks_run_sequentially_with_fresh_namespaces(tmp_path, monkeypatch, external_tree):
    import re
    from modules import process_runner, finetune_protocol
    data, models = external_tree
    monkeypatch.setenv("FACE_DATASET_ROOT", str(data))
    monkeypatch.setenv("FACE_MODEL_ROOT", str(models))
    monkeypatch.chdir(ROOT)
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan execution must not start a campaign or prepare datasets")
    monkeypatch.setattr(process_runner, "launch_campaign", forbidden)
    monkeypatch.setattr(finetune_protocol, "prepare_selected_protocols", forbidden)
    saved_path = tmp_path / "new_settings.json"
    for path in sorted((ROOT / "fine-tune/notebooks").glob("[0-9][0-9]_*.ipynb")):
        namespace = {}
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = re.sub(r'^SETTINGS_PATH = .*$', lambda _: f"SETTINGS_PATH = Path({str(saved_path)!r})",
                            "".join(cell["source"]), flags=re.MULTILINE)
            exec(compile(source, str(path), "exec"), namespace)
        assert namespace["SETTINGS_PATH"] == saved_path
        if "EXECUTE" in namespace:
            assert namespace["EXECUTE"] is False
    saved = flow.resolve_settings(ROOT, saved_path)
    assert saved["data_integrity"]["mode"] == "content_verified_reuse"
    assert saved["campaign"]["finetune_datasets"] == ["survface", "tinyface", "scface"]
    assert "shared_v1_boundary" in saved["paths"]["checkpoint_dir"]
    assert Path(saved["paths"]["checkpoint_dir"]).is_relative_to(models)
