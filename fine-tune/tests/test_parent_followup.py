import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "C:/ronbun")
from modules import parent_followup as followup


def recipe(parent="old"):
    return dict(parent_hash=parent, code_fingerprints={"train.py": "same"},
                config_hash="derived", training=dict(weight_decay=0.0001, lr=0.001),
                dataset=dict(num_classes=5319, train_list_hash="frozen"), seed=8972)


def test_same_finetune_allows_only_explicit_parent_replacement():
    old, new = recipe(), recipe("new")
    followup.require_same_finetune(old, new, "new")
    assert old["parent_hash"] == "old"
    new["training"]["weight_decay"] += 0.00001
    with pytest.raises(ValueError, match="beyond"):
        followup.require_same_finetune(old, new, "new")


def test_same_finetune_rejects_data_and_unreviewed_source_changes(monkeypatch):
    from modules import notebook_workflow as w
    monkeypatch.setattr(w, "source_transition", lambda *args: False)
    old, new = recipe(), recipe("new")
    new["code_fingerprints"]["train.py"] = "changed"
    with pytest.raises(ValueError, match="source transition"):
        followup.require_same_finetune(old, new, "new")
    new = recipe("new")
    new["dataset"]["num_classes"] = 4255
    with pytest.raises(ValueError, match="beyond"):
        followup.require_same_finetune(old, new, "new")


def test_parent_pilot_cannot_start_downstream_training(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps([dict(completed=False, status="paused_epoch_limit", epoch=2)]))
    with pytest.raises(ValueError, match="full schedule"):
        followup.require_parent_result(dict(result=str(result)), {}, tmp_path)


def test_result_cannot_substitute_a_different_parent_path(tmp_path, monkeypatch):
    from modules import notebook_workflow as w
    outside = tmp_path / "other.pt"
    outside.write_bytes(b"not-the-selected-parent")
    root = tmp_path / "selected"
    monkeypatch.setattr(w, "checkpoint_path", lambda *args: root / "parent.pt")
    result = tmp_path / "result.json"
    result.write_text(json.dumps([dict(completed=True, status="completed", checkpoint=str(outside),
                                      sha256=followup.compute_sha256(outside))]))
    with pytest.raises(ValueError, match="outside"):
        followup.require_parent_result(dict(result=str(result)), dict(settings={}), root)


def test_pinned_sources_reject_mid_wait_edits(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    record = followup.pin(source)
    plan = dict(sources={str(source): record["sha256"]})
    for name in ("launch", "request", "followup_reference", "baseline_yaml", "validation_protocol", "interpretation_guide"):
        plan[name] = record
    followup.verify_sources(plan)
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="changed"):
        followup.verify_sources(plan)


def result(uid, tpir=0.2, count=60423):
    return dict(metrics=pd.DataFrame([dict(dataset="survface", compression_profile="origin_1024",
        method="global_safe", target_fpir=t, model_uid=uid, mated_count=count, non_mated_count=121736,
        tpir=0.99, realized_fpir=0.7, origin_test_roc_tpir_at_20=tpir,
        origin_test_roc_fpir=t, unknown_cosine_ge_095_count=100)
        for t in (.01, .05, .10, .20)]))


def test_origin_comparison_uses_test_roc_not_calibrated_tpir():
    before, after = result("old", .2), result("new", .3)
    before["metrics"]["mated_count"] = before["metrics"]["mated_count"].astype(float)
    before["metrics"]["non_mated_count"] = before["metrics"]["non_mated_count"].astype(float)
    table = followup.comparison_table(before, after, "old")
    assert all(table.reference_tpir_pct == 20)
    assert all(table.control_tpir_pct == 30)
    assert all(table.control_minus_reference_pp == 10)
    assert table.iloc[0].ucface_table_i_tpir_pct == 29.85


@pytest.mark.parametrize("change", ["duplicate", "missing", "exceeded"])
def test_origin_evidence_rejects_ambiguous_or_invalid_operating_points(change):
    sample = result("new")
    rows = sample["metrics"]
    if change == "duplicate":
        sample["metrics"] = pd.concat([rows, rows.iloc[:1]])
    elif change == "missing":
        sample["metrics"] = rows.iloc[1:]
    else:
        rows.loc[0, "origin_test_roc_fpir"] = .5
    with pytest.raises(ValueError):
        followup.origin_rows(sample)


def test_origin_comparison_rejects_different_cohort_counts():
    with pytest.raises(ValueError, match="cohort"):
        followup.comparison_table(result("old"), result("new", count=60000), "old")
