import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import clean_benchmarks as b
from modules import clean_training as t


def test_parent_checkpoint_records_arcface_margin_and_scale():
    config = t.checkpoint_config("parent", t.recipe("parent"), "train.txt", matrix=False)
    head = config["head"]
    assert head["margin_arc"] == 0.5
    assert head["margin_am"] == pytest.approx(0.5 * math.sin(0.5))
    assert head["scale"] == 64.0


@pytest.mark.parametrize("clip_limit", [None, 5.0])
@pytest.mark.parametrize("invalid", ["loss", "gradient"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="production CUDA loss and gradient check")
def test_nonfinite_batch_never_steps_optimizer(monkeypatch, clip_limit, invalid):
    class Classifier(torch.nn.Linear):
        def forward(self, xs, ys):
            logits = super().forward(xs)
            penalty = logits.new_tensor(float("inf")) if invalid == "loss" else None
            return logits, penalty

    class Batches(list):
        batch_sampler = SimpleNamespace(set_epoch=lambda *_: None)

    model = Classifier(2, 2).cuda()
    if invalid == "gradient":
        model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    opt = torch.optim.SGD([{"params": [model.weight]}, {"params": [model.bias]}], lr=0.025)
    monkeypatch.setattr(opt, "step", lambda: pytest.fail("optimizer updated a nonfinite batch"))
    batches = Batches([(torch.eye(2), torch.tensor([0, 1]))])
    r = dict(t.recipe("parent"), max_grad_norm=clip_limit)
    with pytest.raises(FloatingPointError, match=f"Nonfinite {invalid} at epoch 3, batch 1"):
        t.train_epoch(model, opt, "parent", r, batches, 3)


@pytest.mark.parametrize("version,one_epoch_warmup", [
    (t._PRE_PARENT_V3, True),
    ({"clean_training": "402cc58c5d7b9f943749f866da7bdebcf3f975ee11f6c6167dd8a70833e00bce"}, False),
])
def test_completed_batch256_parent_requires_matching_recipe_and_implementation(tmp_path, version, one_epoch_warmup):
    from modules.checkpoint_manager import compute_sha256

    folder = tmp_path / "parent"
    folder.mkdir()
    checkpoint = folder / "epoch_24_0123456789ab.pt"
    checkpoint.write_bytes(b"test checkpoint")
    r = dict(t.recipe("parent"), batch=256, lr=0.05, warmup_start=0.005)
    if one_epoch_warmup:
        for key in ("warmup_epochs", "backbone_decay", "max_grad_norm"):
            del r[key]
        r.update(lr=0.1, warmup_start=0.01, conv_linear_decay=0.0001)
    implementation = {**t.files_for_protocol(), **version}
    plan = dict(stage="parent", recipe=r, implementation=implementation, parent_sha256=None)
    rows = [dict(epoch=i) for i in range(1, 25)]
    for epoch in b.MILESTONES:
        rows[epoch - 1]["zero_shot"] = {}
    report = dict(status="completed", epoch=24, epochs=rows, parent_sha256=None,
                  checkpoint=checkpoint.name, sha256=compute_sha256(checkpoint))

    def publish():
        t.write_json(folder / "plan.json", plan)
        report["plan_sha256"] = compute_sha256(folder / "plan.json")
        t.write_json(folder / "summary.json", report)
        (folder / "COMPLETED").write_text(compute_sha256(folder / "summary.json"))

    publish()
    assert t.final_artifact(folder)[0] == checkpoint
    r["batch"] = 128
    publish()
    with pytest.raises(ValueError, match="Incomplete clean training stage"):
        t.final_artifact(folder)
    r["batch"] = 256
    implementation["clean_protocols"] = "0" * 64
    publish()
    with pytest.raises(ValueError, match="Incomplete clean training stage"):
        t.final_artifact(folder)


def test_finetune_health_gate_transition_and_strict_thresholds():
    history = [{"loss": value} for value in (4., 3., 2., 6.)]
    row = dict(epoch=4, loss=8., head_direction_norm=0.85, output_bn_weight_l2=1.0)
    assert not t.finetune_health_issue(row, history[:3], 3)
    assert not t.finetune_health_issue(dict(row, epoch=5, loss=7.), history, 3)
    assert "loss" in t.finetune_health_issue(dict(row, epoch=5, loss=7.001), history, 3)
    for loss in (float("nan"), float("inf"), -float("inf")):
        assert "loss" in t.finetune_health_issue(dict(row, epoch=1, loss=loss), [], 3)


@pytest.mark.parametrize("metric", ["head_direction_norm", "output_bn_weight_l2"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU stage, real checkpoint and resume path")
def test_finetune_health_failure_preserves_checkpoints_and_blocks_resume(tmp_path, monkeypatch, metric):
    from modules.backbone_factory import build_backbone

    r = dict(t.recipe("finetune"), epochs=2, images=2, classes=2)
    parent = build_backbone(dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=0.0))
    monkeypatch.setattr(t, "recipe", lambda stage: r)
    monkeypatch.setattr(t, "input_list", lambda *args: tmp_path / "train_list.txt")
    monkeypatch.setattr(t, "load_parent", lambda *args: ({"backbone_state_dict": parent.state_dict()}, "a" * 64))
    monkeypatch.setattr(
        t, "build_dataloader", lambda *a, **kw: SimpleNamespace(dataset=range(2), num_workers=0, prefetch_factor=None)
    )
    run = tmp_path / "run"
    folder = run / "finetune"
    progress_path = folder / "progress.json"
    first_epoch = {}

    # Control only the two epochs' parameter values. Diagnostics and publication use run_stage.
    @torch.no_grad()
    def controlled_epoch(model, opt, stage, recipe, loader, epoch):
        model.head.weight.zero_()
        model.head.weight[0, 0] = 1
        model.head.weight[0, 1] = -1
        model.backbone.bn.weight.fill_(1)
        if epoch == 2:
            first_epoch.update(json.loads(progress_path.read_bytes()))
            if metric == "head_direction_norm":
                model.head.weight[0, 1] = 1
            else:
                model.backbone.bn.weight.zero_()
        return 2.0

    monkeypatch.setattr(t, "train_epoch", controlled_epoch)
    with pytest.raises(RuntimeError, match=metric):
        t.run_stage(tmp_path, run, "finetune", keep_raw_results=False)
    failed_bytes = progress_path.read_bytes()
    failed = json.loads(failed_bytes)
    assert first_epoch["epoch"] == 1 and failed["epoch"] == 2
    assert "health_gate_failure" not in first_epoch["epochs"][0]
    assert metric in failed["epochs"][-1]["health_gate_failure"]
    assert failed["epochs"][-1][metric] == pytest.approx(1.0 if metric == "head_direction_norm" else 0.0)
    monkeypatch.setattr(t, "train_epoch", lambda *a, **kw: pytest.fail("resumed a rejected checkpoint"))
    with pytest.raises(RuntimeError, match=metric):
        t.run_stage(tmp_path, run, "finetune", keep_raw_results=False, resume=True)
    assert progress_path.read_bytes() == failed_bytes
    for record in (first_epoch, failed):
        checkpoint = t.receipt_checkpoint(folder, record)
        state = t.load_checkpoint(str(checkpoint))
        assert state["completed"] is False and state["epoch"] == record["epoch"]
    assert metric in (folder / "last_failure.txt").read_text()
    assert not (folder / "COMPLETED").exists()
    assert not (folder / "summary.json").exists()


def test_recipe_phase_boundaries_and_origin_budget():
    parent, ft = t.recipe("parent"), t.recipe("finetune")
    assert (parent["batch"], parent["epochs"], parent["milestones"]) == (128, 24, [10, 18, 22])
    assert parent["images"] % parent["batch"] == 2
    assert (parent["warmup_start"], parent["lr"], parent["warmup_epochs"]) == (0.0025, 0.025, 2)
    assert parent["max_grad_norm"] == 5.0
    assert parent["augmentation"]["affine_probability"] == 0
    assert (ft["classes"], ft["batch"], ft["warmup_epochs"], ft["epochs"]) == (5319, 64, 3, 40)
    assert b.MILESTONES == (2, 5, 10, 18, 24)
    unknown = [0.8, 0.8] + [0.3] * 8
    mask = np.array([True, True, True] + [False] * 10)
    row = b.survface_roc(
        [0.99] * 3 + unknown, [0.90, 0.79, 0.99] + [np.nan] * 10, np.array([20, 1, 21] + [0] * 10), mask
    )["0.10"]
    assert (row["tp"], row["fp"], row["registered"], row["unknown"]) == (1, 0, 3, 10)
    assert row["rank_failures"] == row["threshold_failures"] == 1
    with pytest.raises(ValueError):
        t.run_directory(Path("."), "../old")


def test_fold_threshold_never_fits_held_out_scores_and_rank1_rejects_ties():
    pairs, vectors = [], []
    for fold in range(10):
        for i in range(600):
            label = i < 300
            score = (0.9 if label else 0.1) if fold else (0.6 if label else 0.5)
            offset = len(vectors)
            vectors.extend([[1.0, 0.0], [score, (1 - score**2) ** 0.5]])
            pairs.append([offset, offset + 1, label])
    result = b.lfw_accuracy(np.asarray(vectors), np.asarray(pairs))
    assert result["folds"][0]["threshold"] == pytest.approx(0.9)
    assert result["folds"][0]["correct"] == 300
    rows = [
        dict(identity_id=x, role=y, distance=d)
        for x, y, d in [
            ("a", "test_gallery", ""),
            ("b", "test_gallery", ""),
            ("a", "test_registered_probe", "1"),
        ]
    ]
    result = b.rank_one(np.ones((3, 2)), rows)
    assert result["distances"]["1"]["correct"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real GPU stage handoff")
def test_clean_stage_recovery_and_handoff(tmp_path, monkeypatch):
    # Keep the real GPU model, loader, optimization, checkpoints and stage entry points.
    # Only the cohort/epoch budget and costly benchmark measurements are reduced.
    original_recipe, build_loader = t.recipe, t.build_dataloader

    def short_recipe(stage):
        return dict(
            original_recipe(stage), epochs=2 if stage == "parent" else 4, classes=2, images=8, batch=4
        )

    monkeypatch.setattr(t, "recipe", short_recipe)
    monkeypatch.setattr(t, "MILESTONES", (2,))
    panel = {"lfw": {"accuracy": 0.5}, "tinyface": {"rank1": 0.1}, "scface": {"rank1": 0.2}}
    monkeypatch.setattr(t, "prepare_panels", lambda root: ({"fixture": True}, {"prepare_seconds": 0}))
    monkeypatch.setattr(t, "monitor", lambda *args, **kwargs: dict(metrics=panel, seconds=0.0))
    monkeypatch.setattr(t, "build_dataloader", lambda *a, **k: build_loader(*a, **{**k, "num_workers": 0}))
    data = tmp_path / "inputs"
    data.mkdir()
    rng = np.random.default_rng(567)
    file = data / "train.txt"
    with file.open("w", encoding="utf-8") as out:
        for i in range(8):
            p = data / f"{i}.png"
            assert cv2.imwrite(str(p), rng.integers(0, 256, (112, 112, 3), dtype=np.uint8))
            out.write(f"{p} {i % 2}\n")
    monkeypatch.setattr(t, "input_list", lambda *args: file)
    run = tmp_path / "new"
    with pytest.raises(FileNotFoundError):
        t.run_stage(data, run, "finetune")
    train = t.train_epoch

    def interrupt(model, opt, stage, r, loader, epoch, **kwargs):
        if epoch == 2:
            raise InterruptedError("fixture")
        return train(model, opt, stage, r, loader, epoch, **kwargs)

    monkeypatch.setattr(t, "train_epoch", interrupt)
    with pytest.raises(InterruptedError):
        t.run_stage(data, run, "parent")
    assert not (run / "parent/COMPLETED").exists()
    with pytest.raises(FileNotFoundError):
        t.run_stage(data, run, "finetune")
    monkeypatch.setattr(t, "train_epoch", train)
    parent = t.run_stage(data, run, "parent", resume=True)
    assert [r["epoch"] for r in parent["epochs"] if "zero_shot" in r] == [2]
    assert parent["epochs"][0]["ending_lrs"] == pytest.approx([0.01, 0.01])
    assert parent["epochs"][1]["ending_lrs"] == pytest.approx([0.025, 0.025])
    plan = json.loads((run / "parent/plan.json").read_bytes())
    assert plan["num_workers"] == 0 and plan["prefetch_factor"] is None
    for row in parent["epochs"]:
        assert row["grad_norm_clipped_max"] > 0 and row["max_linear_row_norm"] > 0
        assert 0 <= row["head_direction_norm"] <= 1
    # Recovery after the summary was committed but before its marker was published.
    (run / "parent/COMPLETED").unlink()
    monkeypatch.setattr(t, "prepare_panels", lambda root: ({"fixture": True}, {"prepare_seconds": 7}))
    assert t.run_stage(data, run, "parent", resume=True) == parent
    child = t.run_stage(data, run, "finetune")
    assert child["parent_sha256"] == parent["sha256"]
    assert t.final_artifact(run / "finetune")[0].exists()
    assert len(list((run / "parent").glob("*.pt"))) == 1
    with pytest.raises(FileExistsError):
        t.run_stage(data, run, "finetune", resume=True)
    monkeypatch.setattr(t, "evaluate_survface", lambda *args: {"0.10": {"tpir_at_20": 0.15}})
    report = t.final_evaluation(data, run)
    assert report["gap_to_table_i_pp"] == pytest.approx(-9.80)
    assert t.final_evaluation(data, run) == report
    (run / "finetune/summary.json").write_text(json.dumps({}))
    with pytest.raises(ValueError):
        t.final_evaluation(data, run)

    # An unhealthy epoch must survive on disk even at the final milestone,
    # and resuming must not turn a failed final epoch into a completed model.
    rejected = tmp_path / "rejected"
    monkeypatch.setattr(t, "recipe", lambda stage: dict(short_recipe(stage), epochs=1))
    monkeypatch.setattr(t, "MILESTONES", (1,))
    original = t.parent_diagnostics
    monkeypatch.setattr(t, "parent_diagnostics", lambda m: {**original(m), "max_abs_prelu": 11.0})
    monkeypatch.setattr(t, "monitor", lambda *a, **kw: pytest.fail("unhealthy model reached evaluation"))
    for resume in (False, True):
        with pytest.raises(RuntimeError, match="Parent health gate stopped epoch 1: max_abs_prelu"):
            t.run_stage(data, rejected, "parent", resume=resume)
        folder = rejected / "parent"
        progress = json.loads((folder / "progress.json").read_bytes())
        assert progress["epoch"] == 1 and "max_abs_prelu" in progress["epochs"][0]["health_gate_failure"]
        checkpoint = t.receipt_checkpoint(folder, progress)
        assert t.load_checkpoint(str(checkpoint))["completed"] is False
        assert "Parent health gate" in (folder / "last_failure.txt").read_text()
        assert not (folder / "COMPLETED").exists()
