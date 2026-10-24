"""FaceX-owned protocol publication tests; no ronbun imports or checkout."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.research_protocol import publish_handoff, sha
from modules import finetune_protocol


def profile(root, name):
    output = root / name
    record = output / "fixture/protocol.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(dict(name=name, fixture=True)))
    return dict(name=name, research_output=str(output), protocol_record=str(record),
                split_policy="facex_ronbun_v1", enabled=True, local_audit_required=False)


def test_dataset_addition_keeps_previous_indexes_immutable(tmp_path):
    scface = profile(tmp_path, "scface")
    first = publish_handoff([scface])
    original = {p: sha(p) for p in (first, tmp_path / "handoff.json", Path(scface["protocol_record"]))}
    tinyface = profile(tmp_path, "tinyface")
    second = publish_handoff([scface, tinyface], first)
    assert second != first
    assert set(json.loads(first.read_text())["protocols"]) == {"scface"}
    assert set(json.loads(second.read_text())["protocols"]) == {"scface", "tinyface"}
    assert all(sha(p) == digest for p, digest in original.items())
    assert publish_handoff([scface, tinyface], second) == second
    # First-snapshot compatibility also supports a caller without saved settings.
    assert publish_handoff([tinyface]) == second


def test_existing_dataset_cannot_be_reassigned_in_same_namespace(tmp_path):
    scface = profile(tmp_path, "scface")
    first = publish_handoff([scface])
    changed = Path(scface["research_output"]) / "new/protocol.json"
    changed.parent.mkdir()
    changed.write_text('{"new_split":true}')
    with pytest.raises(FileExistsError, match="new namespace"):
        publish_handoff([{**scface, "protocol_record": str(changed)}], first)


def test_tampered_previous_protocol_cannot_be_republished(tmp_path):
    scface = profile(tmp_path, "scface")
    first = publish_handoff([scface])
    Path(scface["protocol_record"]).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Prior handoff protocol changed"):
        publish_handoff([scface], first)


def test_preparation_exposes_exact_handoff_in_results_and_settings(tmp_path, monkeypatch):
    scface = profile(tmp_path, "scface")
    tinyface = profile(tmp_path, "tinyface")
    # Publication is the unit under test; full protocol construction has a
    # separate fixture/real-data audit. Unselected protocols start unprepared.
    pending = dict(tinyface, protocol_record=str(tmp_path / "tinyface/UNPREPARED/protocol.json"))
    def prepare(selected):
        if selected["name"] == "tinyface":
            selected.update(tinyface)
        return {"name": selected["name"]}
    monkeypatch.setattr(finetune_protocol, "prepare_protocol", prepare)
    cfg = dict(campaign=dict(finetune_datasets=["scface"]), finetune_profiles=dict(scface=scface, tinyface=pending))
    first = finetune_protocol.prepare_selected_protocols(cfg)
    initial = cfg["campaign"]["handoff_path"]
    assert first[0]["handoff_path"] == initial
    cfg["campaign"]["finetune_datasets"].append("tinyface")
    second = finetune_protocol.prepare_selected_protocols(cfg)
    assert cfg["campaign"]["handoff_path"] != initial
    assert all(row["handoff_path"] == cfg["campaign"]["handoff_path"] for row in second)
