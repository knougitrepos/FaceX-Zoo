"""Boundary gradients and fail-before-update behavior; no production training."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fine-tune"))
from head.ArcFace import ArcFace
from head.MagFace import MagFace
from modules import notebook_workflow as flow
from modules.checkpoint_manager import CheckpointManager, load_checkpoint
from modules.model_builder import build_model
from modules.trainer import Trainer, NonFiniteTrainingError, TRAINING_NUMERICS, set_seed
from test_notebook_workflow import cfg


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("target", [0, 1])
def test_arcface_boundary_has_finite_forward_and_backward(dtype, sign, target):
    head = ArcFace(2, 2, margin_arc=.6, scale=64.).to(dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2, dtype=dtype))
    feats = torch.tensor([[sign, 0.]], dtype=dtype, requires_grad=True)
    logits = head(feats, torch.tensor([target]))
    loss = F.cross_entropy(logits, torch.tensor([target]))
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert torch.isfinite(feats.grad).all()
    assert torch.isfinite(head.weight.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_arcface_interior_matches_original_formula_and_gradients(dtype):
    head = ArcFace(2, 2, margin_arc=.6, scale=64.).to(dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2, dtype=dtype))
    feats = torch.tensor([[.6, .8], [-.8, .6]], dtype=dtype, requires_grad=True)
    labels = torch.tensor([0, 1])
    actual = head(feats, labels)
    actual_grads = torch.autograd.grad(actual.sum(), (feats, head.weight))
    cos = (F.normalize(feats) @ F.normalize(head.weight, dim=0)).clamp(-1, 1)
    adjusted = cos * head.cos_margin - (1 - cos.square()).sqrt() * head.sin_margin
    adjusted = torch.where(cos > head.min_cos_theta, adjusted, cos - head.margin_am)
    expected = cos.clone()
    expected[torch.arange(2), labels] = adjusted[torch.arange(2), labels]
    expected = expected * head.scale
    expected_grads = torch.autograd.grad(expected.sum(), (feats, head.weight))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("target", [0, 1])
def test_magface_boundary_has_finite_forward_and_backward(dtype, sign, target):
    head = MagFace(2, 2, scale=64.).to(dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2, dtype=dtype))
    feats = torch.tensor([[20. * sign, 0.]], dtype=dtype, requires_grad=True)
    labels = torch.tensor([target])
    logits, regularizer = head(feats, labels)
    loss = F.cross_entropy(logits, labels) + regularizer.mean()
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert torch.isfinite(feats.grad).all()
    assert torch.isfinite(head.weight.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_magface_interior_preserves_logits_regularizer_and_gradients(dtype):
    head = MagFace(2, 2, scale=64.).to(dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2, dtype=dtype))
    feats = torch.tensor([[12., 16.], [-24., 18.]], dtype=dtype, requires_grad=True)
    labels = torch.tensor([0, 1])
    actual, actual_g = head(feats, labels)
    actual_loss = F.cross_entropy(actual, labels) + actual_g.mean()
    actual_grads = torch.autograd.grad(actual_loss, (feats, head.weight))

    norm = feats.norm(dim=1, keepdim=True).clamp(head.l_a, head.u_a)
    margin = head.calc_margin(norm)
    cos = (F.normalize(feats) @ F.normalize(head.weight, dim=0)).clamp(-1, 1)
    adjusted = cos * margin.cos() - (1 - cos.pow(2)).sqrt() * margin.sin()
    adjusted = torch.where(cos > torch.cos(torch.pi - margin), adjusted, cos - head.margin_am)
    expected = cos.clone()
    expected[torch.arange(2), labels] = adjusted[torch.arange(2), labels]
    expected = expected * head.scale
    expected_g = head.lamda * (1 / head.u_a**2 * norm + 1 / norm)
    expected_loss = F.cross_entropy(expected, labels) + expected_g.mean()
    expected_grads = torch.autograd.grad(expected_loss, (feats, head.weight))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def make_trainer(cfg, tmp_path):
    set_seed(8972)
    train, _ = flow.make_loaders(cfg)
    model = build_model(cfg["backbone"], cfg["heads"]["arcface"])
    return Trainer(model, train, None, cfg, CheckpointManager(str(tmp_path / "ckpt")),
                   torch.device("cpu"), head_type="arcface", stage="finetune")


def assert_same_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("component", ["loss", "gradient"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_second_batch_stops_without_update_or_checkpoint(cfg, tmp_path, component, bad_value):
    trainer = make_trainer(cfg, tmp_path)
    after_first = {}
    original_step = trainer.optimizer.step

    def record_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        after_first["params"] = [p.detach().clone() for p in trainer.model.parameters()]
        after_first["optimizer"] = copy.deepcopy(trainer.optimizer.state_dict())
        return result

    trainer.optimizer.step = Mock(side_effect=record_step)
    trainer.checkpoint_manager.save_checkpoint = Mock(side_effect=AssertionError("Must not save bad state"))
    trainer.last_saved_ckpt = str(tmp_path / "previous_good.pt")
    calls = 0
    if component == "loss":
        def criterion(logits, labels):
            nonlocal calls
            calls += 1
            loss = F.cross_entropy(logits, labels)
            return loss * bad_value if calls == 2 else loss
        # nn.Module permits a module replacement, not a plain function.
        class FaultyLoss(torch.nn.Module):
            def forward(self, logits, labels):
                return criterion(logits, labels)
        trainer.criterion = FaultyLoss()
    else:
        def bad_gradient(grad):
            nonlocal calls
            calls += 1
            return torch.full_like(grad, bad_value) if calls == 2 else grad
        trainer.model.head.weight.register_hook(bad_gradient)
    with pytest.raises(NonFiniteTrainingError, match=f"Non-finite {component}.*batch 2"):
        trainer.train()
    assert trainer.optimizer.step.call_count == 1
    assert_same_tree(after_first["optimizer"], trainer.optimizer.state_dict())
    for saved, current in zip(after_first["params"], trainer.model.parameters()):
        assert torch.equal(saved, current)
        assert current.grad is None
    trainer.checkpoint_manager.save_checkpoint.assert_not_called()
    progress = json.loads(trainer.progress_path.read_text())
    assert progress["phase"] == "failed" and progress["completed"] is False
    assert progress["invalid_component"] == component
    assert progress["batches_completed"] == 1 and progress["failure_batch"] == 2
    assert progress["optimizer_step_skipped"] is True
    assert progress["last_checkpoint"].endswith("previous_good.pt")


def test_checkpoint_records_numerics_and_resume_source(cfg, tmp_path):
    trainer = make_trainer(cfg, tmp_path)
    trainer.train(model_prefix="test")
    path = Path(trainer.last_saved_ckpt)
    payload = load_checkpoint(str(path))
    assert payload["extra_meta"]["training_numerics"] == TRAINING_NUMERICS
    resumed = make_trainer(cfg, tmp_path / "second")
    assert resumed.resume_from_checkpoint(str(path)) == 2
    assert resumed.last_saved_ckpt == str(path.resolve())
    assert len(resumed.resume_source["sha256"]) == 64
    assert resumed.resume_source["source_training_numerics"] == TRAINING_NUMERICS
