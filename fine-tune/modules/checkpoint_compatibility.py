"""Exact audited source transitions, never a general resume bypass.

The checked-in manifest pins both complete source maps. Any other source change
still fails; model/data/optimizer configuration must remain exactly equal.
"""
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "configs/verification_resume_compatibility.json"


def source_transition(previous, current):
    if not previous or not current or previous == current:
        return None
    if not MANIFEST.is_file():
        return None
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Each edge is explicitly reviewed. No inferred transitive/reverse edges.
    for edge in [record, *record.get("transitions", [])]:
        if (edge.get("schema") == "verification-only-source-transition-v1"
                and previous == edge.get("previous_sources") and current == edge.get("current_sources")):
            return edge["id"]
    return None


def verification_only_transition(previous, current):
    transition = source_transition(previous.get("code_fingerprints"), current.get("code_fingerprints"))
    ignored = {"code_fingerprints", "config_hash"}
    a = {k: v for k, v in previous.items() if k not in ignored}
    b = {k: v for k, v in current.items() if k not in ignored}
    return transition if transition and a == b else None
