"""Numerical/byte-level contracts, including adversarial cache invalidation."""
import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.target_margin import ArcFace, MagFace, target_sine, BOUNDARY_EPS, NUMERICAL_POLICY
from modules.alignment import bind_aligned_dataset
from modules import integrity_cache
from test_notebook_workflow import cfg, aligned_cfg


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_neighbors_and_boundary_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    one = torch.tensor(1., device=device)
    near = torch.nextafter(one, one.new_tensor(0.))
    c = torch.stack([-one, -near, one.new_tensor(0.), near, one]).requires_grad_()
    actual = target_sine(c)
    grad = torch.autograd.grad(actual.sum(), c)[0]
    assert torch.isfinite(grad).all()
    assert grad[0] == 0 and grad[-1] == 0
    assert abs(grad[1]) > 2000 and abs(grad[-2]) > 2000
    interior = c.detach()[1:-1].requires_grad_()
    original = (1 - interior.square()).sqrt()
    safe = target_sine(interior)
    assert torch.equal(original, safe)
    assert torch.equal(torch.autograd.grad(original.sum(), interior)[0],
                       torch.autograd.grad(safe.sum(), interior)[0])
    assert actual[0] == actual[-1] == BOUNDARY_EPS**.5
    assert torch.isnan(target_sine(torch.tensor([float("nan")], device=device))).all()


@pytest.mark.parametrize("head_type", [ArcFace, MagFace])
def test_head_scope_fp32_and_autocast(head_type, monkeypatch):
    import modules.target_margin as module
    head = head_type(4, 30)
    x = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 2, 4])
    sine = module.target_sine
    sizes = []
    def checked(target):
        sizes.append((target.shape, target.dtype))
        return sine(target)
    monkeypatch.setattr(module, "target_sine", checked)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head(x, labels)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.dtype == torch.float32
    assert sizes == [(torch.Size([3]), torch.float32)]
    assert head.numerical_policy == NUMERICAL_POLICY
    with pytest.raises(TypeError, match="FP32"):
        head(x.half(), labels)


def test_magface_boundary_preserves_adaptive_norm_graph():
    head = MagFace(2, 2, scale=64.)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))
    x = torch.tensor([[60., 0.]], requires_grad=True)
    out, reg = head(x, torch.tensor([0]))
    norm = x.norm(dim=1, keepdim=True)
    margin = head.calc_margin(norm)
    expected = 64*(margin.cos() - BOUNDARY_EPS**.5 * margin.sin())
    assert torch.equal(out[:, :1], expected)
    actual = torch.autograd.grad(out[0, 0], x, retain_graph=True)[0]
    oracle = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual, oracle)
    assert abs(actual[0, 0]) > .01
    assert torch.isfinite(reg).all()


def test_decode_receipt_reused_only_with_all_image_hashes(aligned_cfg, monkeypatch):
    first = bind_aligned_dataset(aligned_cfg)["data_integrity_verification"]
    assert first["images_decoded"] > 0 and not first["reused_decode_receipt"]
    def forbidden(*args, **kwargs):
        raise AssertionError("decode repeated despite matching bytes and validation contract")
    monkeypatch.setattr(integrity_cache.cv2, "imdecode", forbidden)
    second = bind_aligned_dataset(aligned_cfg)["data_integrity_verification"]
    assert second["reused_decode_receipt"] and second["images_decoded"] == 0
    assert second["image_hashes_checked"] == first["image_hashes_checked"]
    assert second["receipt_sha256"] == first["receipt_sha256"]
    report = json.loads(Path(aligned_cfg["dataset"]["alignment_report"]).read_text())
    row = json.loads(Path(report["manifest"]).read_text().splitlines()[0])
    path = Path(row["source"])
    stamp = path.stat()
    data = bytearray(path.read_bytes()); data[-1] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="hash mismatch"):
        bind_aligned_dataset(aligned_cfg)


def test_validator_change_decodes_again_and_full_mode_never_reuses(aligned_cfg, monkeypatch):
    first = bind_aligned_dataset(aligned_cfg)["data_integrity_verification"]
    contract = integrity_cache.receipt_contract
    monkeypatch.setattr(integrity_cache, "receipt_contract", lambda *a: {**contract(*a), "new_validation": True})
    changed = bind_aligned_dataset(aligned_cfg)["data_integrity_verification"]
    assert not changed["reused_decode_receipt"] and changed["images_decoded"] > 0
    assert first["receipt_path"] != changed["receipt_path"]
    cfg = copy.deepcopy(aligned_cfg); cfg["data_integrity"] = {"mode": "full"}
    full = bind_aligned_dataset(cfg)["data_integrity_verification"]
    assert full["images_decoded"] > 0 and full["receipt_path"] is None


def test_failed_decode_never_publishes_receipt(aligned_cfg, monkeypatch):
    monkeypatch.setattr(integrity_cache.cv2, "imdecode", lambda *a, **k: None)
    with pytest.raises(ValueError, match="Invalid source"):
        bind_aligned_dataset(aligned_cfg)
    root = Path(aligned_cfg["dataset"]["alignment_report"]).parent / "verification_cache"
    assert not list(root.glob("*.json"))


def test_tampered_receipt_fails_closed(aligned_cfg):
    verification = bind_aligned_dataset(aligned_cfg)["data_integrity_verification"]
    path = Path(verification["receipt_path"])
    data = json.loads(path.read_text()); data["verified_images"] -= 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid image verification receipt"):
        bind_aligned_dataset(aligned_cfg)
