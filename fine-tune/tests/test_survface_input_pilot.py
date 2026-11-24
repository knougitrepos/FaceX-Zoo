import numpy as np

from modules.survface_input_pilot import evaluate, transform


def test_padding_preserves_rectangular_crop():
    x = np.full((14, 28, 3), 40, np.uint8)
    assert np.all(transform(x, "stretch") == 40)
    padded = transform(x, "aspect_pad")
    assert (padded[:28] == 128).all() and (padded[84:] == 128).all()
    assert (padded[28:84] == 40).all()


def test_fit_threshold_ties_and_genuine_score_failure():
    # Ten-way tie exhausts a one-error fit budget. A high wrong match
    # cannot count as TP even though this registered identity ranks second.
    rows = [("gallery", "A", 0, ""), ("gallery", "B", 1, ""),
            ("probe", "B", 1, ""), ("probe", "A", 0, "")]
    rows += [("fit", "C", 0, "")] * 10 + [("unknown", "D", 0, "")]
    fit = [.8, .6]
    vec = np.array([[1, 0], [0, 1], [1, 0], [1, 0]] + [fit] * 10 + [[1, 0]], dtype=np.float32)
    m, hits, _ = evaluate(vec, rows)
    assert m["fit_fp"] == 0 and m["fp"] == 1
    assert hits.tolist() == [False, True] and m["threshold_failure"] == 1


def test_top_twenty_required_even_above_threshold():
    galleries = np.eye(21)
    scores = np.ones(21)
    scores[0] = .5
    scores /= np.linalg.norm(scores)
    rows = [("gallery", str(i), i, "") for i in range(21)]
    rows += [("probe", "0", 0, ""), ("fit", "F", 0, ""), ("unknown", "U", 0, "")]
    vectors = np.vstack([galleries, scores, -scores, scores])
    m, _, _ = evaluate(vectors, rows)
    assert m["rank_failure"] == 1 and m["tp"] == 0
