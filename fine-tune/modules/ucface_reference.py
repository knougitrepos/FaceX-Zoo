"""Paper evidence and claim boundaries; this module never changes a training recipe.

Sources: UCFace, TIFS 2024, DOI 10.1109/TIFS.2024.3426973, section IV-A;
local paper/부록.pdf, Table IX and implementation/augmentation paragraphs.
Unspecified author details must not be inferred from a local performance gain.
"""

PAPER_TRAINING = {"tinyface": (2570, 7804), "scface": (50, 800), "survface": (5319, 220888)}
PURPOSES = {"local_protocol_baseline", "paper_baseline", "diagnostic_control"}
FULL_FR_RECIPES = {"full_fr_resize_only", "full_fr_pfld_five_point_alignment"}


def selection_scope(settings, rows, purpose):
    if purpose not in PURPOSES:
        raise ValueError("Select local_protocol_baseline, paper_baseline, or diagnostic_control; exact UCFace reproduction is not verified")
    controlled = any(r["recipe"] == "arcface_continuous_margin" for r in rows)
    if controlled and purpose != "diagnostic_control":
        raise ValueError("continuous_margin is a diagnostic_control, never a baseline recipe")
    full_fr = any(r["recipe"] in FULL_FR_RECIPES for r in rows)
    parent_binding = settings.get("campaign", {}).get("parent_binding")
    ft_control = settings.get("campaign", {}).get("finetune_control")
    if parent_binding and purpose != "diagnostic_control":
        raise ValueError("Replacement Parent must be a diagnostic_control")
    if ft_control and not parent_binding:
        raise ValueError("finetune_control requires an explicit parent_binding under diagnostic_control")
    if full_fr and purpose != "paper_baseline" and not (purpose == "diagnostic_control" and parent_binding):
        raise ValueError("Full 5,319-ID SurvFace recipes require purpose='paper_baseline'")
    if purpose == "paper_baseline" and any(r["dataset"] == "survface" and r["recipe"] not in FULL_FR_RECIPES for r in rows):
        raise ValueError("paper_baseline requires full_fr_resize_only or full_fr_pfld_five_point_alignment for SurvFace")
    if purpose in ("local_protocol_baseline", "paper_baseline") and settings["heads"]["arcface"]["margin_am"] != 0:
        raise ValueError("Baseline requires the unchanged FaceX-Zoo ArcFace margin_am=0")
    populations = {}
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        profile = settings["finetune_profiles"][ds]
        ds_recipes = {r["recipe"] for r in rows if r["dataset"] == ds}
        if ds == "survface" and ds_recipes <= FULL_FR_RECIPES:
            actual = [5319, 220888]
        else:
            actual = [profile["num_classes"], profile["expected_images"]]
        populations[ds] = dict(local_training=actual, paper_training=list(PAPER_TRAINING[ds]),
                               counts_match=actual == list(PAPER_TRAINING[ds]))
    result = dict(
        purpose=purpose,
        exact_ucface_reproduction=False,
        author_checkpoint_verified=False,
        author_preprocessing_verified=False,
        embedding_dimension=dict(
            local=settings["backbone"]["feat_dim"],
            main_paper=1024,
            author_recollection=512,
            decision="Use main-paper 1024 for Table I; supplement Table XVII is a separate 512D experiment",
        ),
        populations=populations,
        origin_comparison_required=True,
        reference="doi:10.1109/TIFS.2024.3426973; IV-A; supplement Table IX",
        limitations=[
            "Local parent and preprocessing; author weights and complete recipe not verified",
            "SCFace training identity list is researcher-defined, not author-confirmed",
            "Paper reports five-run means; a single seed is not that aggregate",
            "SurvFace full-FR reproduction cannot reuse FR-unseen calibration roles without redesign",
        ],
    )
    if purpose == "paper_baseline" or full_fr:
        result["calibration_unseen_in_fr"] = False
    if parent_binding:
        import copy
        result["parent_binding"] = copy.deepcopy(parent_binding)
        if ft_control:
            from .parent_controls import (
                FINETUNE_DETACH_MARGIN_LR_0002_SCHEMA,
                FINETUNE_DETACH_MARGIN_SCHEMA,
                FINETUNE_OUTPUT_BN_SCHEMA,
                FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA,
                validate_finetune_control_spec,
            )
            validate_finetune_control_spec(
                ft_control,
                parent_binding,
                settings.get("training", {}).get("weight_decay"),
                rows=rows,
                training_cfg=settings.get("training"),
                head_cfg=settings.get("heads", {}).get("magface"),
            )
            result["finetune_control"] = copy.deepcopy(ft_control)
            if ft_control.get("schema") == FINETUNE_OUTPUT_BN_SCHEMA:
                result["controlled_factor"] = (
                    "Parent training.weight_decay=0.0 reused; fine-tune training.weight_decay=0.0001 "
                    "retained with output bn.bias frozen at 0 and output bn.weight weight_decay=0.0"
                )
            elif ft_control.get("schema") == FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA:
                result["controlled_factor"] = (
                    "Parent training.weight_decay=0.0 reused; fine-tune training.weight_decay 0.0001 -> 0.0 "
                    "with output bn.bias frozen at 0"
                )
            elif ft_control.get("schema") == FINETUNE_DETACH_MARGIN_SCHEMA:
                result["controlled_factor"] = (
                    "Parent training.weight_decay=0.0 reused; fine-tune training.weight_decay 0.0001 -> 0.0, "
                    "output bn.bias frozen at 0, and heads.magface.detach_margin_norm=True"
                )
            elif ft_control.get("schema") == FINETUNE_DETACH_MARGIN_LR_0002_SCHEMA:
                result["controlled_factor"] = (
                    "Parent training.weight_decay=0.0 reused; fine-tune training.lr 0.001 -> 0.0002, "
                    "training.weight_decay 0.0001 -> 0.0, output bn.bias frozen at 0, and heads.magface.detach_margin_norm=True"
                )
            else:
                result["controlled_factor"] = "Parent training.weight_decay=0.0 reused; fine-tune training.weight_decay 0.0001 -> 0.0"
        else:
            result["controlled_factor"] = "Parent training.weight_decay; downstream recipe unchanged"
    return result


def reproduction_review(selection):
    scope = selection_scope(selection["settings"], selection["rows"], selection["purpose"])
    counts_ok = all(item["counts_match"] for item in scope["populations"].values())
    ready = False
    status = "not_verified"
    if selection["purpose"] == "paper_baseline" and counts_ok:
        from pathlib import Path
        from .survface_paper import paper_processed_dir, paper_protocol_dir
        ds_root = selection["settings"].get("storage", {}).get("dataset_root")
        surv_rows = [r for r in selection["rows"] if r["dataset"] == "survface"]
        if not surv_rows:
            ready = True
            status = "paper_counts_ready"
        else:
            proto_ok = (paper_protocol_dir(ds_root) / "protocol.json").is_file()
            prep_ok = all((paper_processed_dir(ds_root, r["recipe"]) / "COMPLETED").is_file() for r in surv_rows)
            ready = bool(proto_ok and prep_ok)
            status = "paper_counts_ready" if ready else "preprocessing_required"
    return dict(
        status=status,
        training_ready=ready,
        **scope,
        required_before_reproduction=[
            "Resolve author preprocessing and SCFace identity split, or explicitly report a local reimplementation",
            "Use a separate full-training SurvFace protocol for paper comparison (5319 IDs / 220888 images)",
            "Keep the existing FR-excluded calibration experiment and its 4255-ID models separate",
            "Compare uncompressed Origin metrics before interpreting PCA/PQ/calibration effects",
        ],
    )


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
