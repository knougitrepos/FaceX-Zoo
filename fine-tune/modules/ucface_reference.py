"""Paper evidence and claim boundaries; this module never changes a training recipe.

Sources: UCFace, TIFS 2024, DOI 10.1109/TIFS.2024.3426973, section IV-A;
local paper/부록.pdf, Table IX and implementation/augmentation paragraphs.
Unspecified author details must not be inferred from a local performance gain.
"""

PAPER_TRAINING = {"tinyface": (2570, 7804), "scface": (50, 800), "survface": (5319, 220888)}
PURPOSES = {"local_protocol_baseline", "diagnostic_control"}


def selection_scope(settings, rows, purpose):
    if purpose not in PURPOSES:
        raise ValueError("Select local_protocol_baseline or diagnostic_control; exact UCFace reproduction is not verified")
    controlled = any(r["recipe"] == "arcface_continuous_margin" for r in rows)
    if controlled and purpose != "diagnostic_control":
        raise ValueError("continuous_margin is a diagnostic_control, never a baseline recipe")
    if purpose == "local_protocol_baseline" and settings["heads"]["arcface"]["margin_am"] != 0:
        raise ValueError("Baseline requires the unchanged FaceX-Zoo ArcFace margin_am=0")
    populations = {}
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        profile = settings["finetune_profiles"][ds]
        actual = [profile["num_classes"], profile["expected_images"]]
        populations[ds] = dict(local_training=actual, paper_training=list(PAPER_TRAINING[ds]),
                               counts_match=actual == list(PAPER_TRAINING[ds]))
    return dict(purpose=purpose, exact_ucface_reproduction=False,
                author_checkpoint_verified=False, author_preprocessing_verified=False,
                embedding_dimension=dict(local=settings["backbone"]["feat_dim"], main_paper=1024,
                    author_recollection=512, decision="Use main-paper 1024 for Table I; supplement Table XVII is a separate 512D experiment"),
                populations=populations, origin_comparison_required=True,
                reference="doi:10.1109/TIFS.2024.3426973; IV-A; supplement Table IX",
                limitations=[
                    "Local parent and preprocessing; author weights and complete recipe not verified",
                    "SCFace training identity list is researcher-defined, not author-confirmed",
                    "Paper reports five-run means; a single seed is not that aggregate",
                    "SurvFace full-FR reproduction cannot reuse FR-unseen calibration roles without redesign",
                ])


def reproduction_review(selection):
    scope = selection_scope(selection["settings"], selection["rows"], selection["purpose"])
    return dict(status="not_verified", training_ready=False, **scope,
                required_before_reproduction=[
                    "Resolve author preprocessing and SCFace identity split, or explicitly report a local reimplementation",
                    "Use a separate full-training SurvFace protocol for paper comparison (5319 IDs / 220888 images)",
                    "Keep the existing FR-excluded calibration experiment and its 4255-ID models separate",
                    "Compare uncompressed Origin metrics before interpreting PCA/PQ/calibration effects",
                ])


def require_reproduction_ready(selection):
    report = reproduction_review(selection)
    if not report["training_ready"]:
        raise ValueError("UCFace reproduction is not verified. Review the displayed data/recipe differences; "
                         "EXECUTE=True must not silently launch a local or diagnostic campaign.")


def validate_campaign_purpose(cfg, stage, selection=None):
    """Stop old in-memory notebook requests before spawning or preparing data."""
    if selection is not None:
        from .selected_campaign import validate_selection_execution
        validate_selection_execution(selection)
    elif stage in ("finetune", "export"):
        margin = cfg.get("heads", {}).get("arcface", {}).get("margin_am", 0)
        if margin != 0 and cfg.get("campaign", {}).get("purpose") != "diagnostic_control":
            raise ValueError("Modified ArcFace margin requires an explicit diagnostic_control campaign")
