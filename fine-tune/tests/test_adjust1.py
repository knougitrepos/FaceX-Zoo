"""Scientific contracts for adjust1; bounded CPU tests, never a campaign result."""
import copy
import json
import re
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.target_margin import ArcFace, MagFace
from modules import notebook_workflow as flow
from modules.alignment import prepare_alignment, verify_alignment, bind_aligned_dataset
from modules.checkpoint_manager import compare_configs, compute_config_hash
from modules.dataset import build_dataloader
from test_notebook_workflow import cfg, aligned_cfg


@pytest.mark.parametrize("head_type", [ArcFace, MagFace])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("target", [0, 1])
def test_target_only_boundary_contract(head_type, dtype, sign, target):
    head = head_type(2, 2, scale=64.).to(dtype)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2, dtype=dtype))
    x = torch.tensor([[20 * sign, 0]], dtype=dtype, requires_grad=True)
    labels = torch.tensor([target])
    output = head(x, labels)
    logits, reg = output if isinstance(output, tuple) else (output, 0.)
    loss = F.cross_entropy(logits, labels) + (reg.mean() if torch.is_tensor(reg) else reg)
    loss.backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(head.weight.grad).all()


@pytest.mark.parametrize("head_type", [ArcFace, MagFace])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_target_only_matches_original_interior_and_regularizer(head_type, dtype):
    torch.manual_seed(12)
    head = head_type(4, 3, scale=64.).to(dtype)
    x = (torch.randn(5, 4, dtype=dtype) * 20).requires_grad_()
    labels = torch.tensor([0, 2, 1, 2, 0])
    output = head(x, labels)
    actual, reg = output if isinstance(output, tuple) else (output, torch.zeros(1, dtype=dtype))
    cosine = (F.normalize(x) @ F.normalize(head.weight, dim=0)).clamp(-1, 1)
    sine = torch.sqrt(1 - cosine.pow(2))
    if head_type is MagFace:
        norm = x.norm(dim=1, keepdim=True).clamp(head.l_a, head.u_a)
        margin = head.calc_margin(norm)
        adjusted = cosine * margin.cos() - sine * margin.sin()
        threshold = torch.cos(torch.pi - margin)
        expected_reg = head.lamda * (1 / head.u_a**2 * norm + 1 / norm)
    else:
        adjusted = cosine * head.cos_margin - sine * head.sin_margin
        threshold, expected_reg = head.min_cos_theta, reg
    adjusted = torch.where(cosine > threshold, adjusted, cosine - head.margin_am)
    expected = cosine.clone()
    expected[torch.arange(5), labels] = adjusted[torch.arange(5), labels]
    expected = expected * head.scale
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(reg, expected_reg, rtol=0, atol=0)
    a = F.cross_entropy(actual, labels) + reg.mean()
    b = F.cross_entropy(expected, labels) + expected_reg.mean()
    for ga, gb in zip(torch.autograd.grad(a, (x, head.weight), retain_graph=True), torch.autograd.grad(b, (x, head.weight))):
        torch.testing.assert_close(ga, gb)


def test_alignment_rejects_mutated_image_and_partial(aligned_cfg, tmp_path):
    cfg = aligned_cfg
    ds = cfg["dataset"]
    report = json.loads(Path(ds["alignment_report"]).read_text())
    effective = bind_aligned_dataset(cfg)
    assert effective["dataset"]["train_list"] != ds["train_list"]
    assert effective["dataset"]["val_list"] is None
    changed_root = copy.deepcopy(cfg)
    changed_root["dataset"]["raw_root"] = str(tmp_path / "different_root")
    with pytest.raises(ValueError, match="raw dataset root differs"):
        bind_aligned_dataset(changed_root)
    path = Path(report["train_list"]).read_text().splitlines()[0].rsplit(" ", 1)[0]
    Path(path).write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        bind_aligned_dataset(cfg)
    class Broken:
        policy = report["policy"]
        def __call__(self, image):
            raise ValueError("no face")
    partial = prepare_alignment(ds["train_list"], ds["raw_root"], tmp_path / "failed", Broken(), max_images=2)
    assert not partial["completed"] and partial["failed"] == 2
    with pytest.raises(ValueError, match="partial/failed"):
        verify_alignment(tmp_path / "failed/report.json", ds["train_list"], report["policy"])


def test_policy_changes_invalidate_resume_and_hash():
    base = {"baseline_version": "adjust1_v1", "code_fingerprints": {"a": "hash"},
            "preprocessing": {"alignment": "aligned", "augmentation": {"rotation": 10}},
            "training": {"drop_last": False}, "head": {"numerical_policy": "target_only_original_sqrt_v1"}}
    for section, key, value in (("preprocessing", "alignment", "resize_only"),
                                ("preprocessing", "augmentation", {"rotation": 20}),
                                ("training", "drop_last", True),
                                ("head", "numerical_policy", "epsilon")):
        changed = copy.deepcopy(base)
        changed[section][key] = value
        assert compare_configs(base, changed, full_strict=True)
        assert compute_config_hash(base) != compute_config_hash(changed)


def test_final_partial_batch_and_resume_indices():
    from modules.input_pipeline import EpochBatchSampler
    sampler = EpochBatchSampler(10, 4, 3, shuffle=False)
    assert [len(x) for x in sampler] == [4, 4, 2]
    sampler.set_epoch(1, start_batch=2)
    assert list(sampler) == [[(8, 1), (9, 1)]]


@pytest.mark.parametrize("head", ["arcface", "adaface", "magface", "cosface"])
def test_resume_before_final_partial_batch_is_identical(aligned_cfg, tmp_path, monkeypatch, head):
    from test_optimized_workflow import test_mid_epoch_recovery_matches_uninterrupted_weights
    effective = bind_aligned_dataset(aligned_cfg)
    # 8 images / batch 6: resume after the first step, then consume the final 2.
    test_mid_epoch_recovery_matches_uninterrupted_weights(effective, tmp_path, monkeypatch, head, "training")


def test_settings_parent_is_separate_and_legacy_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = flow.resolve_settings(root)
    assert cfg["parent_head"]["margin_arc"] == .35
    assert cfg["heads"]["arcface"]["margin_arc"] == .6
    assert cfg["dataset"]["val_list"] is None and cfg["training"]["drop_last"] is False
    old = tmp_path / "settings.json"
    old.write_text('{"paths": {}}')
    with pytest.raises(ValueError, match="Legacy settings"):
        flow.resolve_settings(root, old)


def test_notebooks_are_syntactically_valid_and_safe_defaults():
    import ast
    import nbformat
    for path in (Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert not cell.outputs
                assert "EXECUTE = True" not in cell.source
                assert "shared_parent_v3/" not in cell.source


@pytest.mark.parametrize("name", ["00_environment_and_data.ipynb", "01_prepare_vggface2.ipynb", "02_pretrain.ipynb", "03_finetune.ipynb", "04_export_and_validate.ipynb"])
def test_notebook_plan_execution_isolated_settings(name, tmp_path, monkeypatch):
    from modules import process_runner
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan mode must not launch training/alignment")
    monkeypatch.setattr(process_runner, "launch_campaign", forbidden)
    monkeypatch.setattr(process_runner, "launch_alignment", forbidden)
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "settings.json"
    flow.save_settings(path, flow.resolve_settings(root))
    nb = json.loads((root / "fine-tune/notebooks" / name).read_text(encoding="utf-8"))
    namespace = {}
    redirected = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source, count = re.subn(r"(?m)^SETTINGS_PATH = .*?$",
                                   lambda _: f"SETTINGS_PATH = Path({str(path)!r})",
                                   "".join(cell["source"]))
            redirected += count
            # Refuse to execute a changed notebook layout against live settings.
            if "save_settings(" in source:
                assert redirected == 1
            exec(compile(source, name, "exec"), namespace)
    assert redirected == 1
    assert namespace["SETTINGS_PATH"] == path
    assert namespace["settings"]["parent_head"]["margin_arc"] == .35
