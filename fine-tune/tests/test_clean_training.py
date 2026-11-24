import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from modules import clean_benchmarks as b
from modules import clean_training as t


def test_recipe_phase_boundaries_and_origin_budget():
    parent, ft = t.recipe("parent"), t.recipe("finetune")
    assert (parent["batch"], parent["epochs"], parent["milestones"]) == (256, 24, [10, 18, 22])
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

    def interrupt(model, opt, stage, r, loader, epoch):
        if epoch == 2:
            raise InterruptedError("fixture")
        return train(model, opt, stage, r, loader, epoch)

    monkeypatch.setattr(t, "train_epoch", interrupt)
    with pytest.raises(InterruptedError):
        t.run_stage(data, run, "parent")
    assert not (run / "parent/COMPLETED").exists()
    with pytest.raises(FileNotFoundError):
        t.run_stage(data, run, "finetune")
    monkeypatch.setattr(t, "train_epoch", train)
    parent = t.run_stage(data, run, "parent", resume=True)
    assert [r["epoch"] for r in parent["epochs"] if "zero_shot" in r] == [2]
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
