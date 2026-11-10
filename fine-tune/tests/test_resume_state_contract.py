"""Formal resume must restore every required component, or reject before loading."""
from pathlib import Path
import copy
import json
import random
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.checkpoint_manager import CheckpointManager, load_checkpoint
from modules.checkpoint_compatibility import MANIFEST, source_transition, verification_only_transition
from modules.baseline_policy import code_fingerprints


class Toy(torch.nn.Module):
    def __init__(self, value=99.):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1, bias=False)
        self.head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.backbone.weight.fill_(value)
            self.head.weight.fill_(value)


def states(model):
    opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 20], gamma=.1)
    return opt, sched


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    # No GPU work, even on GPU-equipped test machines.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = Toy(1)
    opt, sch = states(m)
    for _ in range(11):
        opt.zero_grad()
        (m.backbone.weight.sum()*0 + m.head.weight.sum()*0).backward()
        opt.step()
        sch.step()
    manager = CheckpointManager(str(tmp_path))
    path = manager.save_checkpoint("complete.pt", m, opt, sch, 11, batch_idx=2,
        extra_meta=dict(seed=8972, head="arcface", stage="pretrain",
                        epoch_state=dict(epoch=12,next_batch=2,loss_sum=3.,loss_count=4)))
    return manager, load_checkpoint(path), tmp_path


@pytest.mark.parametrize("key", ["backbone_state_dict", "head_state_dict",
    "optimizer_state_dict", "scheduler_state_dict", "rng_state", "epoch", "batch_idx"])
@pytest.mark.parametrize("mode", ["missing", "none", "empty"])
def test_rejects_incomplete_component_before_loading(checkpoint, key, mode):
    manager, payload, root = checkpoint
    if mode == "missing":
        payload.pop(key)
    else:
        payload[key] = None if mode == "none" else {}
    path=root/"bad.pt"; torch.save(payload,path)
    model=Toy();opt,sch=states(model)
    initial=torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=key):
        manager.resume_checkpoint(str(path), model,opt,sch,expected_config=payload["config"])
    assert model.backbone.weight.item()==model.head.weight.item()==99
    assert sch.last_epoch==0 and torch.equal(torch.get_rng_state(),initial)


@pytest.mark.parametrize("key", ["torch_rng", "numpy_rng", "python_rng", "cuda_rng"])
def test_rejects_missing_rng_substate(checkpoint,key):
    manager,payload,root=checkpoint
    payload["rng_state"].pop(key)
    path=root/"bad.pt";torch.save(payload,path)
    model=Toy();opt,sch=states(model)
    with pytest.raises(ValueError,match="rng_state"):
        manager.resume_checkpoint(str(path),model,opt,sch,expected_config=payload["config"])


@pytest.mark.parametrize("change", ["weights", "optimizer", "scheduler", "rng", "epoch_state", "missing_recovery"])
def test_rejects_invalid_structure_before_loading(checkpoint,change):
    manager,payload,root=checkpoint
    if change=="weights":payload["head_state_dict"]["weight"]=torch.zeros(2,1)
    if change=="optimizer":payload["optimizer_state_dict"]["param_groups"][0]["params"]=[]
    if change=="scheduler":payload["scheduler_state_dict"].pop("last_epoch")
    if change=="rng":payload["rng_state"]["torch_rng"]=torch.zeros(2,dtype=torch.float32)
    if change=="epoch_state":payload["extra_meta"]["epoch_state"]["next_batch"]=3
    if change=="missing_recovery":payload["extra_meta"].pop("epoch_state")
    path=root/"bad.pt";torch.save(payload,path)
    model=Toy();opt,sch=states(model)
    with pytest.raises(ValueError,match="Resume State Error"):
        manager.resume_checkpoint(str(path),model,opt,sch,expected_config=payload["config"])
    assert model.backbone.weight.item()==99


def test_complete_checkpoint_restores_state_and_rng(checkpoint):
    manager,payload,root=checkpoint
    path=root/"complete.pt"
    model=Toy();opt,sch=states(model)
    nxt,meta=manager.resume_checkpoint(str(path),model,opt,sch,expected_config=payload["config"],return_meta=True)
    assert nxt==12 and meta["epoch_state"]["next_batch"]==2
    assert model.backbone.weight.item()==model.head.weight.item()==1
    assert sch.state_dict()==payload["scheduler_state_dict"]
    assert opt.param_groups[0]["lr"]==payload["optimizer_state_dict"]["param_groups"][0]["lr"]
    assert len(opt.state)==2
    assert torch.equal(torch.get_rng_state(),payload["rng_state"]["torch_rng"])
    assert random.getstate()==payload["rng_state"]["python_rng"]
    got=np.random.get_state();expected=payload["rng_state"]["numpy_rng"]
    assert got[0]==expected[0] and np.array_equal(got[1],expected[1]) and got[2:]==expected[2:]


def test_exact_validation_source_edges_only():
    record=json.loads(MANIFEST.read_text())
    current=code_fingerprints()
    assert record["current_sources"]==current
    edges=[record,*record["transitions"]]
    for edge in edges:
        old,new=edge["previous_sources"],edge["current_sources"]
        assert source_transition(old,new)==edge["id"]
        a=dict(code_fingerprints=old,training={"batch_size":64},seed=8972)
        b={**a,"code_fingerprints":new}
        assert verification_only_transition(a,b)==edge["id"]
        assert not verification_only_transition(a,{**b,"seed":8973})
        for k in new:
            assert source_transition(old,{**new,k:"f"*64}) is None
    assert source_transition(current,record["previous_sources"]) is None

