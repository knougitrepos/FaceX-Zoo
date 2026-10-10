"""Scientific guardrails, spawn loading, and exact batch-boundary recovery."""
import copy
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.dataset import build_dataloader
from modules import notebook_workflow as flow
from modules.trainer import Trainer, TrainingStopped, set_seed
from modules.checkpoint_manager import CheckpointManager, load_checkpoint
from modules.model_builder import build_model
from modules.finetune_protocol import prepare_protocol, verify_protocol
from test_notebook_workflow import cfg


def test_spawn_workers_preserve_sample_order_and_augmentation(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    index = tmp_path / "samples.txt"
    base = np.arange(120*120*3, dtype=np.uint32).reshape(120,120,3).astype(np.uint8)
    for i in range(8):
        cv2.imwrite(str(root / f"{i}.png"), np.roll(base, i * 3, axis=0))
    index.write_text("".join(f"{i}.png {i%2}\n" for i in range(8)))
    args = dict(data_root=str(root), index_file=str(index), batch_size=2, seed=8972,
                cache_dir=str(tmp_path / "cache"), is_train=True)
    zero = build_dataloader(**args, num_workers=0)
    two = build_dataloader(**args, num_workers=2)
    try:
        for epoch in (1, 2):
            zero.batch_sampler.set_epoch(epoch)
            two.batch_sampler.set_epoch(epoch)
            a, b = list(zero), list(two)
            assert len(a) == len(b) == 4
            for (x, y), (xx, yy) in zip(a,b):
                assert torch.equal(x, xx) and torch.equal(y, yy)
        assert len(pickle.dumps(two.dataset.samples)) < 1024
        # Offset skips completed batches without replaying decoding/augmentation.
        zero.batch_sampler.set_epoch(2, 2)
        for (x, y), (xx, yy) in zip(list(zero), a[2:]):
            assert torch.equal(x,xx) and torch.equal(y,yy)
    finally:
        if two._iterator:
            two._iterator._shutdown_workers()


def test_low_resolution_parameters_cannot_change_for_speed():
    root = Path(__file__).resolve().parents[2]
    original = flow.resolve_settings(root)
    flow.validate_paper_parameters(original)
    for override in ({"training":{"batch_size":128}}, {"training":{"epoches":20}},
                     {"heads":{"arcface":{"margin_arc":0.5}}}):
        with pytest.raises(ValueError,match="Paper"):
            flow.validate_paper_parameters(flow.merge_settings(original,override))
    assert len(flow.plan_matrix(original,"pretrain")) == 1
    assert len(flow.plan_matrix(original,"finetune")) == 30
    expanded = flow.merge_settings(original,{"campaign":{"finetune_datasets":["survface","tinyface","scface"]}})
    with pytest.raises(ValueError,match="disabled"):
        flow.plan_matrix(expanded,"finetune")


def test_protocol_rejects_overlap_and_source_mutation(cfg):
    profile = cfg["finetune_profiles"]["tinyface"]
    verify_protocol(profile)
    evaluation = Path(profile["evaluation_manifest"])
    evaluation.write_text(evaluation.read_text().replace("source_identity_id", "source_identity_id").replace(
        "eval/c/1.png,c,", "eval/c/1.png,a,"))
    with pytest.raises(ValueError,match="overlap"):
        prepare_protocol(profile)
    with pytest.raises(ValueError,match="manifest changed"):
        verify_protocol(profile)


@pytest.mark.parametrize("head", ["arcface","adaface","magface"])
@pytest.mark.parametrize("stop_phase", ["training", "validation"])
def test_mid_epoch_recovery_matches_uninterrupted_weights(cfg, tmp_path, monkeypatch, head, stop_phase):
    cfg = flow.merge_settings(cfg,{"training":{"epoches":2,"save_freq":1,"print_freq":1},
                                   "runtime":{"stop_file":str(tmp_path/"stop.request")}})
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    def create(directory):
        set_seed(8972)
        model = build_model(cfg["backbone"],cfg["heads"][head])
        train,val = flow.make_loaders(cfg)
        return Trainer(model,train,val,cfg,CheckpointManager(str(directory)),torch.device("cpu"),head_type=head,stage="pretrain")
    try:
        reference = create(tmp_path/"reference")
        reference.train(model_prefix="ref")
        expected = {k:v.detach().clone() for k,v in reference.model.state_dict().items()}
        broken = create(tmp_path/"broken")
        original_progress = broken._progress
        def request_stop(phase,**values):
            original_progress(phase,**values)
            if phase==stop_phase and values.get("batches_completed")==0:
                Path(cfg["runtime"]["stop_file"]).touch()
        monkeypatch.setattr(broken,"_progress",request_stop)
        with pytest.raises(TrainingStopped):
            broken.train(model_prefix="partial")
        recovery = broken.last_saved_ckpt
        payload = load_checkpoint(recovery)
        assert payload["completed"] is False and payload["epoch"] == 0
        assert payload["extra_meta"]["epoch_state"]["next_batch"] == (1 if stop_phase == "training" else broken.train_loader.batch_sampler.total_batches)
        assert json.loads(broken.progress_path.read_text())["phase"] == "stopped"
        Path(cfg["runtime"]["stop_file"]).unlink()
        resumed = create(tmp_path/"resumed")
        start = resumed.resume_from_checkpoint(recovery)
        resumed.train(start_epoch=start,model_prefix="resumed")
        for key,value in resumed.model.state_dict().items():
            assert torch.equal(value,expected[key]),key
        assert resumed.history == reference.history
    finally:
        torch.set_num_threads(threads)


def test_notebook_subprocess_smoke(cfg):
    from modules.process_runner import launch_campaign
    root=Path(__file__).resolve().parents[2]
    rows=launch_campaign(root,cfg,"pretrain",smoke=True)
    assert len(rows)==1 and rows[0]["is_smoke"] is True and rows[0]["completed"] is False
    assert not list(Path(cfg["paths"]["checkpoint_dir"]).rglob("*.pt"))


def test_legacy_input_rng_is_not_silently_resumed():
    from modules.checkpoint_manager import compare_configs
    legacy = {"preprocessing": {"input_size": [112,112], "pixel_mean":127.5,"pixel_std":128.0}}
    current = copy.deepcopy(legacy)
    current["preprocessing"]["sampling"] = "epoch_index_rng_v1"
    assert any("sampling" in diff for diff in compare_configs(legacy,current,full_strict=True))


def test_export_rejects_changed_preprocessing(cfg, monkeypatch):
    monkeypatch.setattr(flow, "_formal_device", lambda config: torch.device("cpu"))
    monkeypatch.setattr(flow, "validate_paper_parameters", lambda config: None)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        flow.run_training_matrix(cfg,"pretrain")
        flow.run_training_matrix(cfg,"finetune")
        changed = flow.merge_settings(cfg,{"dataset":{"pixel_std":127.5}})
        with pytest.raises(ValueError,match="Export settings differ.*pixel_std"):
            flow.export_matrix(changed)
    finally:
        torch.set_num_threads(threads)
