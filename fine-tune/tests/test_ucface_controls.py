import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.control_geometry import TARGET, similarity
from modules import ucface_controls as controls


@pytest.mark.parametrize("count", [2, 5])
def test_similarity_maps_source_landmarks_to_arc112(count):
    points = TARGET[:count] * 2 + [17, -11]
    matrix = similarity(points)
    np.testing.assert_allclose(points @ matrix[:, :2].T + matrix[:, 2], TARGET[:count], atol=1e-12)
    with pytest.raises(ValueError, match="Degenerate"):
        similarity(np.ones_like(points))


def test_pinned_survface_control_single_variable():
    from modules import notebook_workflow as workflow
    path = Path("F:/face_datasets/.state/facex/shared_v1_boundary/notebook_settings_mobilefacenet.json")
    if not path.exists():
        pytest.skip("Pinned campaign unavailable")
    base = workflow.resolve_settings(Path(__file__).resolve().parents[2], path)
    with pytest.raises(ValueError, match='not a UCFace baseline'):
        controls.survface_settings(base)
    selected = controls.survface_settings(base, purpose='diagnostic_control')
    assert selected["training"] == base["training"]
    assert selected["augmentation"] == base["augmentation"]
    assert selected["finetune_profiles"]["survface"] == base["finetune_profiles"]["survface"]
    assert selected["campaign"]["parent_sha256"] == controls.PARENT
    assert base["heads"]["arcface"]["margin_am"] == 0
    assert selected["heads"]["arcface"] == dict(base["heads"]["arcface"], margin_am=1-math.cos(.6))
    assert all(selected["paths"][k] != v for k, v in base["paths"].items())
    plan = workflow.plan_matrix(selected, "finetune")
    assert [(r["dataset"], r["head"], r["seed"]) for r in plan] == [("survface", "arcface", 8972)]
