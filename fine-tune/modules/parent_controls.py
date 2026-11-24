"""One-factor parent regularization controls, with explicit immutable provenance."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone, timedelta

import torch

from . import notebook_workflow as w
from .checkpoint_manager import compute_sha256, load_checkpoint, verify_parent_checkpoint

SCHEMA = "facex-parent-weight-decay-control-v1"
FINETUNE_WD_ZERO_SCHEMA = "facex-finetune-weight-decay-control-v1"
FINETUNE_WD_ZERO_SUBDIR = "finetune_wd_zero_v1"
FINETUNE_OUTPUT_BN_SCHEMA = "facex-finetune-output-bn-control-v1"
FINETUNE_OUTPUT_BN_SUBDIR = "finetune_output_bn_v1"
FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA = "facex-finetune-wd-zero-freeze-bias-control-v1"
FINETUNE_WD_ZERO_FREEZE_BIAS_SUBDIR = "finetune_wd_zero_freeze_bias_v1"
FINETUNE_DETACH_MARGIN_SCHEMA = "facex-finetune-detach-margin-control-v1"
FINETUNE_DETACH_MARGIN_SUBDIR = "finetune_detach_margin_v1"
FINETUNE_DETACH_MARGIN_LR_0002_SCHEMA = "facex-finetune-detach-margin-lr-0002-control-v1"
FINETUNE_DETACH_MARGIN_LR_0002_SUBDIR = "finetune_detach_margin_lr_0002_v1"
FINETUNE_WD_ZERO_RUN_ID = "survface_magface_full_fr_resize_only_s8972"
FINETUNE_WD_ZERO_EXPECTED_ROW = dict(
    run_id=FINETUNE_WD_ZERO_RUN_ID,
    dataset="survface",
    head="magface",
    recipe="full_fr_resize_only",
    seed=8972,
)
FIXED_512_INDICES_SHA256 = "521e2f993ab31f027f1af3b3ff70d80e42a27529e9a537a440bddcf760cefb6c"
WD_ZERO_PARENT_SHA256 = "640ff9bf64d22462464d65cb42acb5cc86f3549c88ec568158169015d54a7213"
SURVFACE_FULL_FR_TRAIN_LIST_SHA256 = "5bb88f70d3981c2d4c8e4d6ba90c75ddc8d5fc775e0c0b189b397f12a863ece5"
REFERENCE_WD_0001_512_TRAIN_MODE = {
    "pre_finetune": {
        "sha256": WD_ZERO_PARENT_SHA256,
        "epoch": 40,
        "stage": "pretrain",
        "weight_decay": 0.0,
        "norm_median": 45.04,
        "norm_lt_10": 0,
        "diff_id_cos_mean": -0.0016,
        "diff_id_cos_p95": 0.2446,
    },
    1: {
        "sha256": "f60a48662a9abc6afb3bb58d7231dc90529cf7ebb199d30eb8bb888857595699",
        "epoch": 1,
        "weight_decay": 0.0001,
        "parent_hash": WD_ZERO_PARENT_SHA256,
        "train_list_hash": SURVFACE_FULL_FR_TRAIN_LIST_SHA256,
        "norm_median": 10.13,
        "norm_lt_10": 241,
        "diff_id_cos_mean": 0.0925,
        "diff_id_cos_p95": 0.4143,
    },
    2: {
        "sha256": "c7946a5358c3f79a6bdcec22ab60db478d85f4e7db4133fa6d34b87c69fcb5e5",
        "epoch": 2,
        "weight_decay": 0.0001,
        "parent_hash": WD_ZERO_PARENT_SHA256,
        "train_list_hash": SURVFACE_FULL_FR_TRAIN_LIST_SHA256,
        "norm_median": 9.00,
        "norm_lt_10": 351,
        "diff_id_cos_mean": 0.0865,
        "diff_id_cos_p95": 0.5369,
    },
    5: {
        "sha256": "afa654fef9fe406728f0812aad91065d54fed926d43dea83457cac08cca3d4d5",
        "epoch": 5,
        "weight_decay": 0.0001,
        "parent_hash": WD_ZERO_PARENT_SHA256,
        "train_list_hash": SURVFACE_FULL_FR_TRAIN_LIST_SHA256,
        "norm_median": 9.20,
        "norm_lt_10": 329,
        "diff_id_cos_mean": 0.0515,
        "diff_id_cos_p95": 0.4581,
    },
    10: {
        "sha256": "ae9d7be919818d698626fe124c5adcab64fb0d56f4efae7b6847105ba8cc4b6f",
        "epoch": 10,
        "weight_decay": 0.0001,
        "parent_hash": WD_ZERO_PARENT_SHA256,
        "train_list_hash": SURVFACE_FULL_FR_TRAIN_LIST_SHA256,
        "norm_median": 5.60,
        "norm_lt_10": 473,
        "diff_id_cos_mean": 0.0310,
        "diff_id_cos_p95": 0.3832,
    },
    40: {
        "sha256": "a2686428692c6741edce217109484807ac471bbe4f33a743fae6ee305f703ddf",
        "epoch": 40,
        "weight_decay": 0.0001,
        "parent_hash": WD_ZERO_PARENT_SHA256,
        "train_list_hash": SURVFACE_FULL_FR_TRAIN_LIST_SHA256,
        "norm_median": 4.71,
        "norm_lt_10": 484,
        "diff_id_cos_mean": 0.0134,
        "diff_id_cos_p95": 0.2792,
    },
}
REQUIRED_FINETUNE_CONTROL_KEYS = {
    "schema",
    "purpose",
    "changed_parameter",
    "reference_value",
    "control_value",
    "parent_sha256",
    "reference_selection",
    "reference_selection_sha256",
    "reference_checkpoint",
    "reference_sha256",
    "reference_model_uid",
    "reference_report",
}
RUN_ID = "pretrain_arcface_seed8972"
GUIDE = Path(__file__).resolve().parents[1] / "notebooks/PARENT_CONTROL_GUIDE.md"


def make_parent_control(settings, reference, reference_sha256, output_root, *,
                        stop_after_epoch=2, keep_raw_results=True):
    """Build a plan without starting processes or altering shared notebook settings."""
    if type(keep_raw_results) is not bool:
        raise ValueError("KEEP_RAW_RESULTS must be boolean")
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= 40):
        raise ValueError("STOP_AFTER_EPOCH must be None or an integer in 1..40")
    if len(reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in reference_sha256):
        raise ValueError("Reference requires an explicit SHA-256")
    cfg = copy.deepcopy(settings)
    root = Path(output_root).resolve()
    source = Path(reference).resolve()
    for protected in (source.parent, Path(settings["paths"]["checkpoint_dir"]).resolve(),
                      Path(settings["dataset"]["raw_root"]).resolve(),
                      Path(settings["dataset"]["alignment_report"]).resolve().parent):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError("Control output must be separate from reference models and datasets")
    cfg["project"].update(seed=8972, seed_list=[8972], keep_raw_results=keep_raw_results)
    cfg["campaign"].update(parent_seed=8972, pretrain_epochs=40, parent_checkpoint=None,
                           parent_sha256=None, purpose="diagnostic_control")
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["save_freq"] = 1
    cfg["data_integrity"] = dict(mode="immutable_verified_reuse", force_reverify=False)
    for key in ("checkpoint_dir", "log_dir", "export_dir", "report_dir", "launch_dir"):
        cfg["paths"][key] = str(root / key.removesuffix("_dir"))
    cfg["runtime"] = {"stop_after_epoch": stop_after_epoch}
    cfg["parent_control"] = dict(schema=SCHEMA, purpose="diagnostic_control",
        reference=dict(path=str(source), sha256=reference_sha256), output_root=str(root),
        changed_parameter="training.weight_decay", reference_value=0.0001, control_value=0.0,
        final_epoch=40, pilot_epoch=2, exact_ucface_reproduction=False,
        module_sha256=compute_sha256(__file__))
    original_plan = root / "control.json"
    if original_plan.is_file():
        original = json.loads(original_plan.read_text(encoding="utf-8"))
        # Keep the original declaration immutable. The current orchestration
        # source is recorded per attempt; scientific differences remain blocked.
        declared = lambda value: {k:v for k,v in value.items() if k != "module_sha256"}
        if declared(original) != declared(cfg["parent_control"]):
            raise ValueError("An incompatible parent control already owns this output root")
        cfg["parent_control"] = original
    cfg["runtime"]["control_driver_sha256"] = compute_sha256(__file__)
    w.validate_settings(cfg)
    return cfg


def inspect_parent_control(cfg):
    """Verify the reference, shared data receipt and every scientific recipe field."""
    control = cfg.get("parent_control", {})
    if control.get("schema") != SCHEMA or control.get("control_value") != 0.0:
        raise ValueError("An explicit parent weight-decay control is required")
    reference = control["reference"]
    parent, _ = verify_parent_checkpoint(reference["path"], expected_hash=reference["sha256"],
        require_hash=True, require_completed=True, expected_seed=8972, target_epoch=40)
    previous = parent["config"]
    current_sources = w.code_fingerprints()
    if previous["code_fingerprints"] != current_sources and not w.source_transition(previous["code_fingerprints"], current_sources):
        raise ValueError("Reference source transition has not been audited")
    bound = w.stage_settings(cfg, "pretrain")
    bound["training"]["epoches"] = cfg["campaign"]["pretrain_epochs"]
    bound = w.bind_aligned_dataset(bound)
    with torch.random.fork_rng(devices=[]):
        model = w.build_model(bound["backbone"], dict(bound["parent_head"], num_class=bound["dataset"]["num_classes"]))
        actual = w.extract_model_and_train_config(model, bound, seed=8972, stage="pretrain", head_type="arcface")
    expected = copy.deepcopy(previous)
    if expected["training"]["weight_decay"] != 0.0001:
        raise ValueError("Reference must use weight_decay=0.0001")
    expected["training"]["weight_decay"] = 0.0
    for item in (actual, expected):
        item.pop("config_hash", None)
        item.pop("code_fingerprints", None)
    differences = w.compare_configs(expected, actual, full_strict=True)
    if differences:
        raise ValueError("Control differs beyond Parent weight_decay: " + "; ".join(differences))
    receipt = bound.get("data_integrity_verification", {})
    root = Path(control["output_root"])
    disk_root = next(p for p in (root, *root.parents) if p.exists())
    checkpoint_bytes = Path(reference["path"]).stat().st_size
    retained = 40 if cfg["project"]["keep_raw_results"] else 2
    estimate = checkpoint_bytes * (retained + 4)
    free = shutil.disk_usage(disk_root).free
    if free < estimate:
        raise OSError(f"Insufficient output space: estimated {estimate}, free {free}")
    return dict(status="ready", run_id=RUN_ID, purpose="diagnostic_control",
        scientific_changes=[dict(parameter="training.weight_decay", previous=0.0001, current=0.0)],
        reference=reference, planned_epochs=40, stop_after_epoch=cfg.get("runtime", {}).get("stop_after_epoch"),
        data_verification=receipt, output_root=str(root), estimated_checkpoint_bytes=estimate,
        free_bytes=free, keep_raw_results=cfg["project"]["keep_raw_results"],
        interpretation_guide=str(GUIDE), exact_ucface_reproduction=False)


def validate_control_resume(cfg, checkpoint):
    if checkpoint is None:
        return
    path = Path(checkpoint).resolve()
    run_dir = w.checkpoint_path(cfg, "pretrain", "arcface", 8972).parent.resolve()
    if not path.is_relative_to(run_dir):
        raise ValueError("Resume must be a checkpoint of this NEW parent control, never the reference parent")
    payload = load_checkpoint(str(path))
    if (payload["config"]["training"]["weight_decay"] != 0.0
            or payload.get("extra_meta", {}).get("execution_settings", {}).get("parent_control") != cfg["parent_control"]):
        raise ValueError("Resume provenance differs from this parent control")


def verify_control_parent(binding):
    """A new Parent must be complete and differ from its reference only in WD."""
    if set(binding) != {"path", "sha256", "reference"}:
        raise ValueError("Parent binding requires path, sha256 and immutable reference")
    parent, digest = verify_parent_checkpoint(binding["path"], expected_hash=binding["sha256"],
        require_hash=True, require_completed=True, expected_seed=8972, target_epoch=40)
    reference = binding["reference"]
    old, _ = verify_parent_checkpoint(reference["path"], expected_hash=reference["sha256"],
        require_hash=True, require_completed=True, expected_seed=8972, target_epoch=40)
    control = parent.get("extra_meta", {}).get("execution_settings", {}).get("parent_control", {})
    if control.get("schema") != SCHEMA or control.get("reference") != reference:
        raise ValueError("Parent control provenance is missing or different")
    before, after = copy.deepcopy(old["config"]), copy.deepcopy(parent["config"])
    if before["training"]["weight_decay"] != 0.0001 or after["training"]["weight_decay"] != 0.0:
        raise ValueError("Parent control must change only weight_decay 0.0001 -> 0")
    if (before["code_fingerprints"] != after["code_fingerprints"]
            and not w.source_transition(before["code_fingerprints"], after["code_fingerprints"])):
        raise ValueError("Parent source change has not been audited")
    before["training"]["weight_decay"] = 0.0
    for item in (before, after):
        item.pop("config_hash", None)
        item.pop("code_fingerprints", None)
    if w.compare_configs(before, after, full_strict=True):
        raise ValueError("Completed Parent differs beyond regularization")
    return parent, digest


def make_followup_selection(baseline, parent_path, parent_sha256, control_root):
    """Exactly one full-FR MagFace with the original downstream regularization."""
    from .selected_campaign import make_selection, save_selection
    from .ucface_controls import ALIGNMENT_CONTROL
    root = Path(control_root)
    control = json.loads((root / "control.json").read_text(encoding="utf-8"))
    binding = dict(path=str(Path(parent_path).resolve()), sha256=parent_sha256, reference=control["reference"])
    verify_control_parent(binding)
    cfg = copy.deepcopy(baseline)
    cfg["campaign"].update(parent_checkpoint=binding["path"], parent_sha256=parent_sha256,
                           parent_binding=binding, purpose="diagnostic_control")
    if cfg["training"]["weight_decay"] != 0.0001:
        raise ValueError("Downstream weight_decay must remain 0.0001")
    model_root = Path(cfg["storage"]["model_root"])
    selection = make_selection(cfg, [dict(dataset="survface", recipe="full_fr_resize_only",
        heads=["magface"], seeds=[8972])], root / "followup", model_root / "facex/diagnostics" / ALIGNMENT_CONTROL,
        purpose="diagnostic_control")
    path = root / "followup/selection.json"
    save_selection(path, selection)
    return path


def validate_finetune_control_spec(ft_control, parent_binding, training_wd, *, rows=None, training_cfg=None, head_cfg=None):
    """Enforce exact schema, values, and single target row for fine-tune diagnostic controls."""
    if not isinstance(ft_control, dict) or set(ft_control) != REQUIRED_FINETUNE_CONTROL_KEYS:
        raise ValueError("Invalid finetune_control declaration keys")
    schema = ft_control.get("schema")
    if ft_control.get("purpose") != "diagnostic_control" or type(training_wd) not in (int, float):
        raise ValueError("Fine-tune control requires purpose=diagnostic_control and numeric training.weight_decay")
    if schema == FINETUNE_WD_ZERO_SCHEMA:
        if (ft_control.get("changed_parameter") != "training.weight_decay"
                or type(ft_control.get("reference_value")) is not float
                or ft_control.get("reference_value") != 0.0001
                or type(ft_control.get("control_value")) is not float
                or ft_control.get("control_value") != 0.0
                or float(training_wd) != 0.0):
            raise ValueError("Fine-tune weight_decay control requires schema=facex-finetune-weight-decay-control-v1 and training.weight_decay=0.0")
        if training_cfg is not None and (
                training_cfg.get("output_bn_freeze_bias") or training_cfg.get("output_bn_weight_decay") is not None):
            raise ValueError("finetune_wd_zero_control must not enable output BN freeze/selective decay options")
        if head_cfg is not None and head_cfg.get("detach_margin_norm"):
            raise ValueError("finetune_wd_zero_control must not enable MagFace detach_margin_norm")
    elif schema == FINETUNE_OUTPUT_BN_SCHEMA:
        expected_ref = {"weight_decay": 0.0001, "output_bn_freeze_bias": False, "output_bn_weight_decay": None}
        expected_ctrl = {"weight_decay": 0.0001, "output_bn_freeze_bias": True, "output_bn_weight_decay": 0.0}
        if (ft_control.get("changed_parameter") != "training.output_bn_freeze_bias+output_bn_weight_decay"
                or ft_control.get("reference_value") != expected_ref
                or ft_control.get("control_value") != expected_ctrl
                or float(training_wd) != 0.0001):
            raise ValueError("Output BN control requires schema=facex-finetune-output-bn-control-v1 and training.weight_decay=0.0001")
        if training_cfg is not None:
            out_bn_wd = training_cfg.get("output_bn_weight_decay")
            if (training_cfg.get("output_bn_freeze_bias") is not True
                    or type(out_bn_wd) not in (int, float)
                    or float(out_bn_wd) != 0.0):
                raise ValueError("finetune_output_bn_control requires output_bn_freeze_bias=True and output_bn_weight_decay=0.0")
        if head_cfg is not None and head_cfg.get("detach_margin_norm"):
            raise ValueError("finetune_output_bn_control must not enable MagFace detach_margin_norm")
    elif schema == FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA:
        expected_ref = {"weight_decay": 0.0001, "output_bn_freeze_bias": False}
        expected_ctrl = {"weight_decay": 0.0, "output_bn_freeze_bias": True}
        if (ft_control.get("changed_parameter") != "training.weight_decay+output_bn_freeze_bias"
                or ft_control.get("reference_value") != expected_ref
                or ft_control.get("control_value") != expected_ctrl
                or float(training_wd) != 0.0):
            raise ValueError("WD=0 + frozen output BN bias control requires schema=facex-finetune-wd-zero-freeze-bias-control-v1 and training.weight_decay=0.0")
        if training_cfg is not None and (
                training_cfg.get("output_bn_freeze_bias") is not True
                or training_cfg.get("output_bn_weight_decay") is not None):
            raise ValueError("finetune_wd_zero_freeze_bias_control requires output_bn_freeze_bias=True and output_bn_weight_decay=None")
        if head_cfg is not None and head_cfg.get("detach_margin_norm"):
            raise ValueError("finetune_wd_zero_freeze_bias_control must not enable MagFace detach_margin_norm")
    elif schema == FINETUNE_DETACH_MARGIN_SCHEMA:
        expected_ref = {"weight_decay": 0.0001, "output_bn_freeze_bias": False, "detach_margin_norm": False}
        expected_ctrl = {"weight_decay": 0.0, "output_bn_freeze_bias": True, "detach_margin_norm": True}
        if (ft_control.get("changed_parameter") != "training.weight_decay+output_bn_freeze_bias+heads.magface.detach_margin_norm"
                or ft_control.get("reference_value") != expected_ref
                or ft_control.get("control_value") != expected_ctrl
                or float(training_wd) != 0.0):
            raise ValueError("MagFace detach_margin_norm control requires schema=facex-finetune-detach-margin-control-v1 and training.weight_decay=0.0")
        if training_cfg is not None and (
                training_cfg.get("output_bn_freeze_bias") is not True
                or training_cfg.get("output_bn_weight_decay") is not None):
            raise ValueError("finetune_detach_margin_control requires output_bn_freeze_bias=True and output_bn_weight_decay=None")
        if head_cfg is not None and head_cfg.get("detach_margin_norm") is not True:
            raise ValueError("finetune_detach_margin_control requires heads.magface.detach_margin_norm=True")
    elif schema == FINETUNE_DETACH_MARGIN_LR_0002_SCHEMA:
        expected_ref = {"lr": 0.001, "weight_decay": 0.0001, "output_bn_freeze_bias": False, "detach_margin_norm": False}
        expected_ctrl = {"lr": 0.0002, "weight_decay": 0.0, "output_bn_freeze_bias": True, "detach_margin_norm": True}
        if (ft_control.get("changed_parameter") != "training.lr+weight_decay+output_bn_freeze_bias+heads.magface.detach_margin_norm"
                or ft_control.get("reference_value") != expected_ref
                or ft_control.get("control_value") != expected_ctrl
                or float(training_wd) != 0.0):
            raise ValueError("MagFace detach_margin_norm lr=0.0002 control requires schema=facex-finetune-detach-margin-lr-0002-control-v1 and training.weight_decay=0.0")
        if training_cfg is not None:
            lr_val = training_cfg.get("lr")
            if (isinstance(lr_val, bool) or type(lr_val) not in (int, float)
                    or float(lr_val) != 0.0002
                    or training_cfg.get("output_bn_freeze_bias") is not True
                    or training_cfg.get("output_bn_weight_decay") is not None):
                raise ValueError("finetune_detach_margin_lr_0002_control requires training.lr=0.0002, output_bn_freeze_bias=True, and output_bn_weight_decay=None")
        if head_cfg is not None and head_cfg.get("detach_margin_norm") is not True:
            raise ValueError("finetune_detach_margin_lr_0002_control requires heads.magface.detach_margin_norm=True")
    else:
        raise ValueError(f"Unsupported finetune_control schema: {schema!r}")
    if schema in (
        FINETUNE_WD_ZERO_SCHEMA,
        FINETUNE_OUTPUT_BN_SCHEMA,
        FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA,
        FINETUNE_DETACH_MARGIN_SCHEMA,
    ) and training_cfg is not None and "lr" in training_cfg:
        lr_val = training_cfg["lr"]
        if isinstance(lr_val, bool) or type(lr_val) not in (int, float) or float(lr_val) != 0.001:
            raise ValueError(f"{schema} must retain baseline training.lr=0.001")
    if not isinstance(parent_binding, dict) or ft_control.get("parent_sha256") != parent_binding.get("sha256"):
        raise ValueError("Fine-tune control parent_sha256 must match parent_binding.sha256")
    for key in ("parent_sha256", "reference_selection_sha256", "reference_sha256"):
        val = ft_control.get(key, "")
        if not isinstance(val, str) or len(val) != 64 or any(c not in "0123456789abcdef" for c in val):
            raise ValueError(f"Fine-tune control requires 64-char hex SHA-256 for {key}")
    for key in ("reference_selection", "reference_checkpoint", "reference_model_uid", "reference_report"):
        if not isinstance(ft_control.get(key), str) or not ft_control[key].strip():
            raise ValueError(f"Fine-tune control requires non-empty string for {key}")
    if rows is not None:
        if not isinstance(rows, list) or len(rows) != 1 or rows[0] != FINETUNE_WD_ZERO_EXPECTED_ROW:
            raise ValueError(
                f"{schema} requires exactly one target row: "
                "survface / full_fr_resize_only / magface / seed 8972"
            )


def _check_expected_settings_diff(ref_settings, new_settings, ref_ckpt_config, new_expected_config, label):
    """Compare full selection settings and extracted configs after normalizing caller-permitted training deltas."""
    if new_settings.get("training", {}).get("save_freq") != 1 or new_settings.get("project", {}).get("keep_raw_results") is not True:
        raise ValueError(f"{label} requires save_freq=1 and KEEP_RAW_RESULTS=True")
    ref_settings["training"]["save_freq"] = 1
    ref_settings["project"]["keep_raw_results"] = True
    for item in (ref_settings, new_settings):
        item.pop("paths", None)
        item.pop("selection_provenance", None)
        item.pop("runtime", None)
        item.pop("data_integrity", None)
        if isinstance(item.get("campaign"), dict):
            item["campaign"].pop("finetune_control", None)
    if ref_settings != new_settings:
        diff_keys = sorted(k for k in set(ref_settings) | set(new_settings) if ref_settings.get(k) != new_settings.get(k))
        raise ValueError(f"{label} settings differ beyond controlled factor: " + ", ".join(diff_keys))

    if (ref_ckpt_config.get("code_fingerprints") != new_expected_config.get("code_fingerprints")
            and not w.source_transition(ref_ckpt_config.get("code_fingerprints"), new_expected_config.get("code_fingerprints"))):
        raise ValueError(f"Unreviewed downstream source transition in {label}")
    for item in (ref_ckpt_config, new_expected_config):
        item.pop("config_hash", None)
        item.pop("code_fingerprints", None)
    if ref_ckpt_config != new_expected_config:
        diff_keys = sorted(k for k in set(ref_ckpt_config) | set(new_expected_config) if ref_ckpt_config.get(k) != new_expected_config.get(k))
        raise ValueError(f"{label} extracted config differs beyond controlled factor: " + ", ".join(diff_keys))


def require_only_finetune_wd_zero(ref_settings, new_settings, ref_ckpt_config, new_expected_config):
    """Zero-tolerance check: only fine-tune training.weight_decay (0.0001 -> 0.0) may differ."""
    old_s, new_s = copy.deepcopy(ref_settings), copy.deepcopy(new_settings)
    if old_s.get("training", {}).get("weight_decay") != 0.0001 or new_s.get("training", {}).get("weight_decay") != 0.0:
        raise ValueError("Fine-tune control must change training.weight_decay 0.0001 -> 0.0")
    old_s["training"]["weight_decay"] = 0.0
    old_c, new_c = copy.deepcopy(ref_ckpt_config), copy.deepcopy(new_expected_config)
    if old_c.get("training", {}).get("weight_decay") != 0.0001 or new_c.get("training", {}).get("weight_decay") != 0.0:
        raise ValueError("Fine-tune control extracted config must change training.weight_decay 0.0001 -> 0.0")
    old_c["training"]["weight_decay"] = 0.0
    _check_expected_settings_diff(old_s, new_s, old_c, new_c, "Fine-tune WD=0 control")


def require_only_finetune_output_bn_control(ref_settings, new_settings, ref_ckpt_config, new_expected_config):
    """Zero-tolerance check: only output_bn_freeze_bias=True and output_bn_weight_decay=0.0 may differ."""
    old_s, new_s = copy.deepcopy(ref_settings), copy.deepcopy(new_settings)
    old_tr, new_tr = old_s.get("training", {}), new_s.get("training", {})
    if (old_tr.get("weight_decay") != 0.0001
            or old_tr.get("output_bn_freeze_bias")
            or old_tr.get("output_bn_weight_decay") is not None):
        raise ValueError("Reference settings must retain baseline weight_decay=0.0001 without output BN control")
    if (new_tr.get("weight_decay") != 0.0001
            or new_tr.get("output_bn_freeze_bias") is not True
            or new_tr.get("output_bn_weight_decay") != 0.0):
        raise ValueError("Output BN control must retain training.weight_decay=0.0001 and set output_bn_freeze_bias=True, output_bn_weight_decay=0.0")
    old_tr["output_bn_freeze_bias"] = True
    old_tr["output_bn_weight_decay"] = 0.0

    old_c, new_c = copy.deepcopy(ref_ckpt_config), copy.deepcopy(new_expected_config)
    oc_tr, nc_tr = old_c.get("training", {}), new_c.get("training", {})
    if (oc_tr.get("weight_decay") != 0.0001
            or "output_bn_freeze_bias" in oc_tr
            or "output_bn_weight_decay" in oc_tr):
        raise ValueError("Reference extracted config must retain weight_decay=0.0001 without output BN control")
    if (nc_tr.get("weight_decay") != 0.0001
            or nc_tr.get("output_bn_freeze_bias") is not True
            or nc_tr.get("output_bn_weight_decay") != 0.0):
        raise ValueError("Output BN control extracted config must set output_bn_freeze_bias=True and output_bn_weight_decay=0.0")
    oc_tr["output_bn_freeze_bias"] = True
    oc_tr["output_bn_weight_decay"] = 0.0
    _check_expected_settings_diff(old_s, new_s, old_c, new_c, "Fine-tune output BN control")


def require_only_finetune_wd_zero_freeze_bias_control(ref_settings, new_settings, ref_ckpt_config, new_expected_config):
    """Zero-tolerance check: only weight_decay (0.0001 -> 0.0) and output_bn_freeze_bias=True may differ."""
    old_s, new_s = copy.deepcopy(ref_settings), copy.deepcopy(new_settings)
    old_tr, new_tr = old_s.get("training", {}), new_s.get("training", {})
    if (old_tr.get("weight_decay") != 0.0001
            or old_tr.get("output_bn_freeze_bias")
            or old_tr.get("output_bn_weight_decay") is not None):
        raise ValueError("Reference settings must retain baseline weight_decay=0.0001 without output BN control")
    if (new_tr.get("weight_decay") != 0.0
            or new_tr.get("output_bn_freeze_bias") is not True
            or new_tr.get("output_bn_weight_decay") is not None):
        raise ValueError("WD=0 + frozen output BN bias control requires training.weight_decay=0.0, output_bn_freeze_bias=True, and output_bn_weight_decay=None")
    old_tr["weight_decay"] = 0.0
    old_tr["output_bn_freeze_bias"] = True

    old_c, new_c = copy.deepcopy(ref_ckpt_config), copy.deepcopy(new_expected_config)
    oc_tr, nc_tr = old_c.get("training", {}), new_c.get("training", {})
    if (oc_tr.get("weight_decay") != 0.0001
            or "output_bn_freeze_bias" in oc_tr
            or "output_bn_weight_decay" in oc_tr):
        raise ValueError("Reference extracted config must retain weight_decay=0.0001 without output BN control")
    if (nc_tr.get("weight_decay") != 0.0
            or nc_tr.get("output_bn_freeze_bias") is not True
            or "output_bn_weight_decay" in nc_tr):
        raise ValueError("WD=0 + frozen output BN bias extracted config must set weight_decay=0.0 and output_bn_freeze_bias=True only")
    oc_tr["weight_decay"] = 0.0
    oc_tr["output_bn_freeze_bias"] = True
    _check_expected_settings_diff(old_s, new_s, old_c, new_c, "Fine-tune WD=0 + frozen output BN bias control")


def require_only_finetune_detach_margin_control(ref_settings, new_settings, ref_ckpt_config, new_expected_config):
    """Zero-tolerance check: only weight_decay=0.0, output_bn_freeze_bias=True, and heads.magface.detach_margin_norm=True may differ."""
    old_s, new_s = copy.deepcopy(ref_settings), copy.deepcopy(new_settings)
    old_tr, new_tr = old_s.get("training", {}), new_s.get("training", {})
    old_hd_s = old_s.get("heads", {}).get("magface", {})
    new_hd_s = new_s.get("heads", {}).get("magface", {})
    if (old_tr.get("weight_decay") != 0.0001
            or old_tr.get("output_bn_freeze_bias")
            or old_tr.get("output_bn_weight_decay") is not None
            or old_hd_s.get("detach_margin_norm")):
        raise ValueError("Reference settings must retain baseline weight_decay=0.0001 without output BN or detach_margin_norm controls")
    if (new_tr.get("weight_decay") != 0.0
            or new_tr.get("output_bn_freeze_bias") is not True
            or new_tr.get("output_bn_weight_decay") is not None
            or new_hd_s.get("detach_margin_norm") is not True):
        raise ValueError("MagFace detach_margin_norm control requires weight_decay=0.0, output_bn_freeze_bias=True, output_bn_weight_decay=None, and detach_margin_norm=True")
    old_tr["weight_decay"] = 0.0
    old_tr["output_bn_freeze_bias"] = True
    old_hd_s["detach_margin_norm"] = True

    old_c, new_c = copy.deepcopy(ref_ckpt_config), copy.deepcopy(new_expected_config)
    oc_tr, nc_tr = old_c.get("training", {}), new_c.get("training", {})
    oc_hd, nc_hd = old_c.get("head", {}), new_c.get("head", {})
    if (oc_tr.get("weight_decay") != 0.0001
            or "output_bn_freeze_bias" in oc_tr
            or "output_bn_weight_decay" in oc_tr
            or "detach_margin_norm" in oc_hd):
        raise ValueError("Reference extracted config must retain weight_decay=0.0001 without output BN or detach_margin_norm controls")
    if (nc_tr.get("weight_decay") != 0.0
            or nc_tr.get("output_bn_freeze_bias") is not True
            or "output_bn_weight_decay" in nc_tr
            or nc_hd.get("detach_margin_norm") is not True):
        raise ValueError("MagFace detach_margin_norm extracted config must set weight_decay=0.0, output_bn_freeze_bias=True, and detach_margin_norm=True")
    oc_tr["weight_decay"] = 0.0
    oc_tr["output_bn_freeze_bias"] = True
    oc_hd["detach_margin_norm"] = True
    _check_expected_settings_diff(old_s, new_s, old_c, new_c, "Fine-tune detach_margin_norm control")


def require_only_finetune_detach_margin_lr_0002_control(ref_settings, new_settings, ref_ckpt_config, new_expected_config):
    """Zero-tolerance check: only lr=0.0002, weight_decay=0.0, output_bn_freeze_bias=True, and detach_margin_norm=True may differ."""
    old_s, new_s = copy.deepcopy(ref_settings), copy.deepcopy(new_settings)
    old_tr, new_tr = old_s.get("training", {}), new_s.get("training", {})
    old_hd_s = old_s.get("heads", {}).get("magface", {})
    new_hd_s = new_s.get("heads", {}).get("magface", {})
    if (old_tr.get("lr") != 0.001
            or old_tr.get("weight_decay") != 0.0001
            or old_tr.get("output_bn_freeze_bias")
            or old_tr.get("output_bn_weight_decay") is not None
            or old_hd_s.get("detach_margin_norm")):
        raise ValueError("Reference settings must retain baseline lr=0.001 and weight_decay=0.0001 without output BN or detach_margin_norm controls")
    if (new_tr.get("lr") != 0.0002
            or new_tr.get("weight_decay") != 0.0
            or new_tr.get("output_bn_freeze_bias") is not True
            or new_tr.get("output_bn_weight_decay") is not None
            or new_hd_s.get("detach_margin_norm") is not True):
        raise ValueError("MagFace detach_margin_norm lr=0.0002 control requires lr=0.0002, weight_decay=0.0, output_bn_freeze_bias=True, output_bn_weight_decay=None, and detach_margin_norm=True")
    old_tr["lr"] = 0.0002
    old_tr["weight_decay"] = 0.0
    old_tr["output_bn_freeze_bias"] = True
    old_hd_s["detach_margin_norm"] = True

    old_c, new_c = copy.deepcopy(ref_ckpt_config), copy.deepcopy(new_expected_config)
    oc_tr, nc_tr = old_c.get("training", {}), new_c.get("training", {})
    oc_hd, nc_hd = old_c.get("head", {}), new_c.get("head", {})
    if (oc_tr.get("lr") != 0.001
            or oc_tr.get("weight_decay") != 0.0001
            or "output_bn_freeze_bias" in oc_tr
            or "output_bn_weight_decay" in oc_tr
            or "detach_margin_norm" in oc_hd):
        raise ValueError("Reference extracted config must retain lr=0.001 and weight_decay=0.0001 without output BN or detach_margin_norm controls")
    if (nc_tr.get("lr") != 0.0002
            or nc_tr.get("weight_decay") != 0.0
            or nc_tr.get("output_bn_freeze_bias") is not True
            or "output_bn_weight_decay" in nc_tr
            or nc_hd.get("detach_margin_norm") is not True):
        raise ValueError("MagFace detach_margin_norm lr=0.0002 extracted config must set lr=0.0002, weight_decay=0.0, output_bn_freeze_bias=True, and detach_margin_norm=True")
    oc_tr["lr"] = 0.0002
    oc_tr["weight_decay"] = 0.0
    oc_tr["output_bn_freeze_bias"] = True
    oc_hd["detach_margin_norm"] = True
    _check_expected_settings_diff(old_s, new_s, old_c, new_c, "Fine-tune detach_margin_norm lr=0.0002 control")


def verify_finetune_control_settings(selection, row, cfg):
    """Verify current fine-tune diagnostic control settings against the pinned reference selection and checkpoint."""
    from .selected_campaign import _expected, load_selection, settings_for
    from .ucface_controls import check_file
    ft_control = cfg.get("campaign", {}).get("finetune_control")
    validate_finetune_control_spec(
        ft_control,
        cfg.get("campaign", {}).get("parent_binding"),
        cfg.get("training", {}).get("weight_decay"),
        training_cfg=cfg.get("training"),
        head_cfg=cfg.get("heads", {}).get("magface"),
    )
    check_file(ft_control["reference_selection"], ft_control["reference_selection_sha256"])
    check_file(ft_control["reference_checkpoint"], ft_control["reference_sha256"])
    ref_selection = load_selection(ft_control["reference_selection"])
    if len(ref_selection["rows"]) != 1 or ref_selection["rows"][0] != row:
        raise ValueError("Fine-tune control row differs from the pinned reference selection")
    ref_cfg = settings_for(ref_selection, ref_selection["rows"][0])
    ref_payload = load_checkpoint(ft_control["reference_checkpoint"])
    if ft_control["schema"] == FINETUNE_OUTPUT_BN_SCHEMA:
        require_only_finetune_output_bn_control(ref_cfg, cfg, ref_payload["config"], _expected(cfg, row))
    elif ft_control["schema"] == FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA:
        require_only_finetune_wd_zero_freeze_bias_control(ref_cfg, cfg, ref_payload["config"], _expected(cfg, row))
    elif ft_control["schema"] == FINETUNE_DETACH_MARGIN_SCHEMA:
        require_only_finetune_detach_margin_control(ref_cfg, cfg, ref_payload["config"], _expected(cfg, row))
    elif ft_control["schema"] == FINETUNE_DETACH_MARGIN_LR_0002_SCHEMA:
        require_only_finetune_detach_margin_lr_0002_control(ref_cfg, cfg, ref_payload["config"], _expected(cfg, row))
    else:
        require_only_finetune_wd_zero(ref_cfg, cfg, ref_payload["config"], _expected(cfg, row))


verify_finetune_wd_zero_settings = verify_finetune_control_settings


def _check_common_finetune_control_payload(
    payload, path, cfg, expected, label, *, check_wd_zero_dir=False, check_output_bn_dir=False,
    check_freeze_bias_dir=False, check_detach_margin_dir=False
):
    from .checkpoint_manager import compute_config_hash
    ft_control = cfg.get("campaign", {}).get("finetune_control")
    validate_finetune_control_spec(
        ft_control,
        cfg.get("campaign", {}).get("parent_binding"),
        cfg.get("training", {}).get("weight_decay"),
        training_cfg=cfg.get("training"),
        head_cfg=cfg.get("heads", {}).get("magface"),
    )
    resolved_path = Path(path).resolve()
    ref_dir = Path(ft_control["reference_checkpoint"]).resolve().parent
    if resolved_path == ref_dir or resolved_path.is_relative_to(ref_dir):
        raise ValueError(f"Reference ft_wd=0.0001 checkpoint cannot be used as a {label} checkpoint")
    diag_parent = Path(ft_control["reference_selection"]).resolve().parent.parent
    if check_wd_zero_dir:
        wd0_dir = (diag_parent / FINETUNE_WD_ZERO_SUBDIR).resolve()
        if resolved_path == wd0_dir or resolved_path.is_relative_to(wd0_dir):
            raise ValueError(f"Previous finetune_wd_zero_v1 checkpoint cannot be used as a {label} checkpoint")
    if check_output_bn_dir:
        obn_dir = (diag_parent / FINETUNE_OUTPUT_BN_SUBDIR).resolve()
        if resolved_path == obn_dir or resolved_path.is_relative_to(obn_dir):
            raise ValueError(f"Previous finetune_output_bn_v1 checkpoint cannot be used as a {label} checkpoint")
    if check_freeze_bias_dir:
        fb_dir = (diag_parent / FINETUNE_WD_ZERO_FREEZE_BIAS_SUBDIR).resolve()
        if resolved_path == fb_dir or resolved_path.is_relative_to(fb_dir):
            raise ValueError(f"Previous finetune_wd_zero_freeze_bias_v1 checkpoint cannot be used as a {label} checkpoint")
    if check_detach_margin_dir:
        dm_dir = (diag_parent / FINETUNE_DETACH_MARGIN_SUBDIR).resolve()
        if resolved_path == dm_dir or resolved_path.is_relative_to(dm_dir):
            raise ValueError(f"Previous finetune_detach_margin_v1 checkpoint cannot be used as a {label} checkpoint")
    ckpt_cfg = payload.get("config", {})
    if (ckpt_cfg.get("code_fingerprints") != expected.get("code_fingerprints")
            and not w.source_transition(ckpt_cfg.get("code_fingerprints"), expected.get("code_fingerprints"))):
        raise ValueError(f"Unreviewed source transition in {label} checkpoint")
    ckpt_clean, exp_clean = copy.deepcopy(ckpt_cfg), copy.deepcopy(expected)
    for item in (ckpt_clean, exp_clean):
        item.pop("config_hash", None)
        item.pop("code_fingerprints", None)
    if ckpt_clean != exp_clean:
        raise ValueError(f"{label} checkpoint config differs from exact expected configuration")
    meta = payload.get("extra_meta", payload.get("metadata", {}))
    exec_settings = meta.get("execution_settings") if isinstance(meta, dict) else None
    if not isinstance(exec_settings, dict) or exec_settings.get("campaign", {}).get("finetune_control") != ft_control:
        raise ValueError(f"{label} checkpoint lacks matching finetune_control provenance")
    if compute_config_hash(ckpt_cfg) != payload.get("config_hash"):
        raise ValueError("Checkpoint config_hash mismatch")
    return ckpt_cfg


def verify_finetune_wd_zero_checkpoint_payload(payload, path, cfg, expected, *, completed=True):
    """Reject any checkpoint with non-zero weight_decay, wrong optimizer param_groups, or reference provenance."""
    ckpt_cfg = payload.get("config", {})
    if ckpt_cfg.get("training", {}).get("weight_decay") != 0.0:
        raise ValueError(f"finetune_wd_zero_control requires checkpoint training.weight_decay=0.0, found {ckpt_cfg.get('training', {}).get('weight_decay')}")
    opt_groups = payload.get("optimizer_state_dict", {}).get("param_groups", [])
    if not isinstance(opt_groups, list) or not opt_groups or any(
            g.get("weight_decay") is None or float(g.get("weight_decay")) != 0.0 for g in opt_groups):
        raise ValueError("finetune_wd_zero_control requires optimizer_state_dict.param_groups weight_decay=0.0")
    _check_common_finetune_control_payload(payload, path, cfg, expected, "finetune_wd_zero_control")


def verify_finetune_output_bn_checkpoint_payload(payload, path, cfg, expected, *, completed=True):
    """Enforce frozen zero bn.bias, split [0.0001, 0.0] optimizer param_groups, and exact provenance."""
    ckpt_tr = payload.get("config", {}).get("training", {})
    if (ckpt_tr.get("weight_decay") != 0.0001
            or ckpt_tr.get("output_bn_freeze_bias") is not True
            or ckpt_tr.get("output_bn_weight_decay") != 0.0):
        raise ValueError("finetune_output_bn_control requires training.weight_decay=0.0001, output_bn_freeze_bias=True, and output_bn_weight_decay=0.0")
    opt_groups = payload.get("optimizer_state_dict", {}).get("param_groups", [])
    if (not isinstance(opt_groups, list) or len(opt_groups) != 2
            or opt_groups[0].get("name") != "default"
            or float(opt_groups[0].get("weight_decay", -1)) != 0.0001
            or opt_groups[1].get("name") != "output_bn_weight"
            or float(opt_groups[1].get("weight_decay", -1)) != 0.0
            or len(opt_groups[1].get("params", [])) != 1):
        raise ValueError("finetune_output_bn_control requires two optimizer param_groups: default(wd=0.0001) and single-param output_bn_weight(wd=0.0)")
    bn_bias = payload.get("backbone_state_dict", {}).get("bn.bias")
    if bn_bias is None or torch.count_nonzero(bn_bias).item() != 0:
        raise ValueError("finetune_output_bn_control checkpoint must retain backbone_state_dict['bn.bias'] frozen at exact zero")
    _check_common_finetune_control_payload(payload, path, cfg, expected, "finetune_output_bn_control", check_wd_zero_dir=True)


def verify_finetune_wd_zero_freeze_bias_checkpoint_payload(payload, path, cfg, expected, *, completed=True):
    """Enforce training.weight_decay=0.0, frozen zero bn.bias, single param_group wd=0.0, and clean provenance."""
    ckpt_tr = payload.get("config", {}).get("training", {})
    if (ckpt_tr.get("weight_decay") != 0.0
            or ckpt_tr.get("output_bn_freeze_bias") is not True
            or "output_bn_weight_decay" in ckpt_tr):
        raise ValueError("finetune_wd_zero_freeze_bias_control requires training.weight_decay=0.0, output_bn_freeze_bias=True, and no output_bn_weight_decay")
    opt_groups = payload.get("optimizer_state_dict", {}).get("param_groups", [])
    if (not isinstance(opt_groups, list) or len(opt_groups) != 1
            or opt_groups[0].get("weight_decay") is None
            or float(opt_groups[0]["weight_decay"]) != 0.0):
        raise ValueError("finetune_wd_zero_freeze_bias_control requires single optimizer param_group with weight_decay=0.0")
    bn_bias = payload.get("backbone_state_dict", {}).get("bn.bias")
    if bn_bias is None or torch.count_nonzero(bn_bias).item() != 0:
        raise ValueError("finetune_wd_zero_freeze_bias_control checkpoint must retain backbone_state_dict['bn.bias'] frozen at exact zero")
    _check_common_finetune_control_payload(
        payload, path, cfg, expected, "finetune_wd_zero_freeze_bias_control", check_wd_zero_dir=True, check_output_bn_dir=True
    )


def verify_finetune_detach_margin_checkpoint_payload(payload, path, cfg, expected, *, completed=True):
    """Enforce weight_decay=0.0, frozen zero bn.bias, detach_margin_norm=True, single wd=0.0 group, and clean provenance."""
    ckpt_cfg = payload.get("config", {})
    ckpt_tr = ckpt_cfg.get("training", {})
    ckpt_hd = ckpt_cfg.get("head", {})
    if (ckpt_tr.get("weight_decay") != 0.0
            or ckpt_tr.get("output_bn_freeze_bias") is not True
            or "output_bn_weight_decay" in ckpt_tr
            or ckpt_hd.get("detach_margin_norm") is not True):
        raise ValueError(
            "finetune_detach_margin_control requires training.weight_decay=0.0, output_bn_freeze_bias=True, no output_bn_weight_decay, and head.detach_margin_norm=True"
        )
    opt_groups = payload.get("optimizer_state_dict", {}).get("param_groups", [])
    if (not isinstance(opt_groups, list) or len(opt_groups) != 1
            or opt_groups[0].get("weight_decay") is None
            or float(opt_groups[0]["weight_decay"]) != 0.0):
        raise ValueError("finetune_detach_margin_control requires single optimizer param_group with weight_decay=0.0")
    bn_bias = payload.get("backbone_state_dict", {}).get("bn.bias")
    if bn_bias is None or torch.count_nonzero(bn_bias).item() != 0:
        raise ValueError("finetune_detach_margin_control checkpoint must retain backbone_state_dict['bn.bias'] frozen at exact zero")
    _check_common_finetune_control_payload(
        payload,
        path,
        cfg,
        expected,
        "finetune_detach_margin_control",
        check_wd_zero_dir=True,
        check_output_bn_dir=True,
        check_freeze_bias_dir=True,
    )


def verify_finetune_detach_margin_lr_0002_checkpoint_payload(payload, path, cfg, expected, *, completed=True):
    """Enforce lr=0.0002, wd=0.0, frozen zero bn.bias, detach_margin_norm=True, single wd=0.0 group, and clean provenance."""
    ckpt_cfg = payload.get("config", {})
    ckpt_tr = ckpt_cfg.get("training", {})
    ckpt_hd = ckpt_cfg.get("head", {})
    ckpt_lr = ckpt_tr.get("lr")
    if (isinstance(ckpt_lr, bool) or type(ckpt_lr) not in (int, float)
            or float(ckpt_lr) != 0.0002
            or ckpt_tr.get("weight_decay") != 0.0
            or ckpt_tr.get("output_bn_freeze_bias") is not True
            or "output_bn_weight_decay" in ckpt_tr
            or ckpt_hd.get("detach_margin_norm") is not True):
        raise ValueError(
            "finetune_detach_margin_lr_0002_control requires training.lr=0.0002, training.weight_decay=0.0, output_bn_freeze_bias=True, no output_bn_weight_decay, and head.detach_margin_norm=True"
        )
    opt_groups = payload.get("optimizer_state_dict", {}).get("param_groups", [])
    if (not isinstance(opt_groups, list) or len(opt_groups) != 1
            or opt_groups[0].get("weight_decay") is None
            or float(opt_groups[0]["weight_decay"]) != 0.0):
        raise ValueError("finetune_detach_margin_lr_0002_control requires single optimizer param_group with weight_decay=0.0")
    init_lr = opt_groups[0].get("initial_lr", opt_groups[0].get("lr"))
    if init_lr is None or abs(float(init_lr) - 0.0002) > 1e-12:
        raise ValueError("finetune_detach_margin_lr_0002_control optimizer param_group initial_lr/lr must equal 0.0002")
    bn_bias = payload.get("backbone_state_dict", {}).get("bn.bias")
    if bn_bias is None or torch.count_nonzero(bn_bias).item() != 0:
        raise ValueError("finetune_detach_margin_lr_0002_control checkpoint must retain backbone_state_dict['bn.bias'] frozen at exact zero")
    _check_common_finetune_control_payload(
        payload,
        path,
        cfg,
        expected,
        "finetune_detach_margin_lr_0002_control",
        check_wd_zero_dir=True,
        check_output_bn_dir=True,
        check_freeze_bias_dir=True,
        check_detach_margin_dir=True,
    )


def _prepare_single_row_control_base(baseline, control_root, output_root, default_subdir, label):
    from .selected_campaign import _validate, final_path, load_selection, settings_for

    model_root = Path(baseline.get("storage", {}).get("model_root", "F:/models"))
    root = (Path(control_root) if control_root is not None else model_root / "facex/diagnostics/parent_weight_decay_zero_v1").resolve()
    out_root = (Path(output_root) if output_root is not None else root / default_subdir).resolve()
    protected_dirs = [root / "checkpoint", root / "followup", root / "followup_execution"]
    for prior_subdir in (
        FINETUNE_WD_ZERO_SUBDIR,
        FINETUNE_OUTPUT_BN_SUBDIR,
        FINETUNE_WD_ZERO_FREEZE_BIAS_SUBDIR,
        FINETUNE_DETACH_MARGIN_SUBDIR,
        FINETUNE_DETACH_MARGIN_LR_0002_SUBDIR,
    ):
        if default_subdir != prior_subdir:
            protected_dirs.append(root / prior_subdir)
    for protected in protected_dirs:
        prot = protected.resolve()
        if out_root == prot or out_root.is_relative_to(prot) or prot.is_relative_to(out_root):
            raise ValueError(f"{label} output must be separate from completed parent, followup, and prior diagnostic artifacts")
    ref_selection_path = root / "followup/selection.json"
    if not ref_selection_path.is_file():
        raise FileNotFoundError(f"Completed WD=0 Parent followup selection is required: {ref_selection_path}")
    if baseline.get("project", {}).get("keep_raw_results") is not True:
        raise ValueError(f"{label} requires KEEP_RAW_RESULTS=True to preserve epoch 1, 2, 5, 10, 40 checkpoints")
    ref_selection_sha = compute_sha256(str(ref_selection_path))
    ref_selection = load_selection(ref_selection_path)
    if len(ref_selection["rows"]) != 1 or ref_selection["rows"][0]["run_id"] != FINETUNE_WD_ZERO_RUN_ID:
        raise ValueError("Reference followup selection must be the single SurvFace full-FR MagFace run")
    binding = copy.deepcopy(ref_selection["settings"]["campaign"]["parent_binding"])
    verify_control_parent(binding)
    ref_row = ref_selection["rows"][0]
    ref_cfg = settings_for(ref_selection, ref_row)
    ref_ckpt = final_path(ref_cfg, ref_row)
    if not ref_ckpt.is_file():
        raise FileNotFoundError(f"Completed WD=0 Parent + ft_wd=0.0001 reference checkpoint is required: {ref_ckpt}")
    ref_hash_file = ref_ckpt.with_name(ref_ckpt.stem + "_hash.json")
    ref_sha = (json.loads(ref_hash_file.read_text(encoding="utf-8"))["sha256"]
               if ref_hash_file.is_file() else compute_sha256(str(ref_ckpt)))
    ref_payload = _validate(ref_ckpt, ref_cfg, ref_row, digest=ref_sha)
    return model_root, out_root, ref_selection_path, ref_selection_sha, binding, ref_cfg, ref_ckpt, ref_sha, ref_payload


def build_finetune_wd_zero_selection(baseline, control_root=None, output_root=None):
    """Reuse completed WD=0 Parent and change only SurvFace MagFace fine-tune weight_decay 0.0001 -> 0.0."""
    from .selected_campaign import _expected, make_selection, settings_for
    from .ucface_controls import ALIGNMENT_CONTROL

    model_root, out_root, ref_selection_path, ref_selection_sha, binding, ref_cfg, ref_ckpt, ref_sha, ref_payload = (
        _prepare_single_row_control_base(baseline, control_root, output_root, FINETUNE_WD_ZERO_SUBDIR, "finetune_wd_zero_control")
    )
    cfg = copy.deepcopy(baseline)
    cfg["project"]["keep_raw_results"] = True
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["save_freq"] = 1
    ft_control = dict(
        schema=FINETUNE_WD_ZERO_SCHEMA,
        purpose="diagnostic_control",
        changed_parameter="training.weight_decay",
        reference_value=0.0001,
        control_value=0.0,
        parent_sha256=binding["sha256"],
        reference_selection=str(ref_selection_path.resolve()),
        reference_selection_sha256=ref_selection_sha,
        reference_checkpoint=str(ref_ckpt.resolve()),
        reference_sha256=ref_sha,
        reference_model_uid="magface-ad3b19a576ec14a0883c",
        reference_report="F:/ronbun_results/facex_workflow/reports/facex-1587c845c7c2c5f137132e9c",
    )
    cfg["campaign"].update(
        parent_checkpoint=binding["path"],
        parent_sha256=binding["sha256"],
        parent_binding=binding,
        finetune_control=ft_control,
        purpose="diagnostic_control",
    )
    selection = make_selection(
        cfg,
        [dict(dataset="survface", recipe="full_fr_resize_only", heads=["magface"], seeds=[8972])],
        out_root,
        model_root / "facex/diagnostics" / ALIGNMENT_CONTROL,
        purpose="diagnostic_control",
    )
    new_cfg = settings_for(selection, selection["rows"][0])
    require_only_finetune_wd_zero(ref_cfg, new_cfg, ref_payload["config"], _expected(new_cfg, selection["rows"][0]))
    return selection


def make_finetune_wd_zero_selection(baseline, control_root=None, output_root=None):
    """Build and save the immutable selection.json for the fine-tune weight_decay=0.0 diagnostic control."""
    from .selected_campaign import save_selection
    selection = build_finetune_wd_zero_selection(baseline, control_root=control_root, output_root=output_root)
    path = Path(selection["root"]) / "selection.json"
    save_selection(path, selection)
    return path


def build_finetune_output_bn_selection(baseline, control_root=None, output_root=None):
    """Reuse completed WD=0 Parent; freeze output bn.bias at 0 and split bn.weight decay=0.0 while keeping wd=0.0001."""
    from .selected_campaign import _expected, make_selection, settings_for
    from .ucface_controls import ALIGNMENT_CONTROL

    model_root, out_root, ref_selection_path, ref_selection_sha, binding, ref_cfg, ref_ckpt, ref_sha, ref_payload = (
        _prepare_single_row_control_base(baseline, control_root, output_root, FINETUNE_OUTPUT_BN_SUBDIR, "finetune_output_bn_control")
    )
    cfg = copy.deepcopy(baseline)
    cfg["project"]["keep_raw_results"] = True
    cfg["training"]["weight_decay"] = 0.0001
    cfg["training"]["output_bn_freeze_bias"] = True
    cfg["training"]["output_bn_weight_decay"] = 0.0
    cfg["training"]["save_freq"] = 1
    ft_control = dict(
        schema=FINETUNE_OUTPUT_BN_SCHEMA,
        purpose="diagnostic_control",
        changed_parameter="training.output_bn_freeze_bias+output_bn_weight_decay",
        reference_value={"weight_decay": 0.0001, "output_bn_freeze_bias": False, "output_bn_weight_decay": None},
        control_value={"weight_decay": 0.0001, "output_bn_freeze_bias": True, "output_bn_weight_decay": 0.0},
        parent_sha256=binding["sha256"],
        reference_selection=str(ref_selection_path.resolve()),
        reference_selection_sha256=ref_selection_sha,
        reference_checkpoint=str(ref_ckpt.resolve()),
        reference_sha256=ref_sha,
        reference_model_uid="magface-ad3b19a576ec14a0883c",
        reference_report="F:/ronbun_results/facex_workflow/reports/facex-1587c845c7c2c5f137132e9c",
    )
    cfg["campaign"].update(
        parent_checkpoint=binding["path"],
        parent_sha256=binding["sha256"],
        parent_binding=binding,
        finetune_control=ft_control,
        purpose="diagnostic_control",
    )
    selection = make_selection(
        cfg,
        [dict(dataset="survface", recipe="full_fr_resize_only", heads=["magface"], seeds=[8972])],
        out_root,
        model_root / "facex/diagnostics" / ALIGNMENT_CONTROL,
        purpose="diagnostic_control",
    )
    new_cfg = settings_for(selection, selection["rows"][0])
    require_only_finetune_output_bn_control(ref_cfg, new_cfg, ref_payload["config"], _expected(new_cfg, selection["rows"][0]))
    return selection


def make_finetune_output_bn_selection(baseline, control_root=None, output_root=None):
    """Build and save the immutable selection.json for the finetune_output_bn_control campaign."""
    from .selected_campaign import save_selection
    selection = build_finetune_output_bn_selection(baseline, control_root=control_root, output_root=output_root)
    path = Path(selection["root"]) / "selection.json"
    save_selection(path, selection)
    return path


def build_finetune_wd_zero_freeze_bias_selection(baseline, control_root=None, output_root=None):
    """Reuse completed WD=0 Parent; set fine-tune weight_decay=0.0 and freeze output bn.bias at exact zero."""
    from .selected_campaign import _expected, make_selection, settings_for
    from .ucface_controls import ALIGNMENT_CONTROL

    model_root, out_root, ref_selection_path, ref_selection_sha, binding, ref_cfg, ref_ckpt, ref_sha, ref_payload = (
        _prepare_single_row_control_base(
            baseline, control_root, output_root, FINETUNE_WD_ZERO_FREEZE_BIAS_SUBDIR, "finetune_wd_zero_freeze_bias_control"
        )
    )
    cfg = copy.deepcopy(baseline)
    cfg["project"]["keep_raw_results"] = True
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["output_bn_freeze_bias"] = True
    cfg["training"].pop("output_bn_weight_decay", None)
    cfg["training"]["save_freq"] = 1
    ft_control = dict(
        schema=FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA,
        purpose="diagnostic_control",
        changed_parameter="training.weight_decay+output_bn_freeze_bias",
        reference_value={"weight_decay": 0.0001, "output_bn_freeze_bias": False},
        control_value={"weight_decay": 0.0, "output_bn_freeze_bias": True},
        parent_sha256=binding["sha256"],
        reference_selection=str(ref_selection_path.resolve()),
        reference_selection_sha256=ref_selection_sha,
        reference_checkpoint=str(ref_ckpt.resolve()),
        reference_sha256=ref_sha,
        reference_model_uid="magface-ad3b19a576ec14a0883c",
        reference_report="F:/ronbun_results/facex_workflow/reports/facex-1587c845c7c2c5f137132e9c",
    )
    cfg["campaign"].update(
        parent_checkpoint=binding["path"],
        parent_sha256=binding["sha256"],
        parent_binding=binding,
        finetune_control=ft_control,
        purpose="diagnostic_control",
    )
    selection = make_selection(
        cfg,
        [dict(dataset="survface", recipe="full_fr_resize_only", heads=["magface"], seeds=[8972])],
        out_root,
        model_root / "facex/diagnostics" / ALIGNMENT_CONTROL,
        purpose="diagnostic_control",
    )
    new_cfg = settings_for(selection, selection["rows"][0])
    require_only_finetune_wd_zero_freeze_bias_control(ref_cfg, new_cfg, ref_payload["config"], _expected(new_cfg, selection["rows"][0]))
    return selection


def make_finetune_wd_zero_freeze_bias_selection(baseline, control_root=None, output_root=None):
    """Build and save the immutable selection.json for the finetune_wd_zero_freeze_bias_control campaign."""
    from .selected_campaign import save_selection
    selection = build_finetune_wd_zero_freeze_bias_selection(baseline, control_root=control_root, output_root=output_root)
    path = Path(selection["root"]) / "selection.json"
    save_selection(path, selection)
    return path


def build_finetune_detach_margin_selection(baseline, control_root=None, output_root=None):
    """Reuse completed WD=0 Parent; set wd=0.0, output_bn_freeze_bias=True, and heads.magface.detach_margin_norm=True."""
    from .selected_campaign import _expected, make_selection, settings_for
    from .ucface_controls import ALIGNMENT_CONTROL

    model_root, out_root, ref_selection_path, ref_selection_sha, binding, ref_cfg, ref_ckpt, ref_sha, ref_payload = (
        _prepare_single_row_control_base(
            baseline, control_root, output_root, FINETUNE_DETACH_MARGIN_SUBDIR, "finetune_detach_margin_control"
        )
    )
    cfg = copy.deepcopy(baseline)
    cfg["project"]["keep_raw_results"] = True
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["output_bn_freeze_bias"] = True
    cfg["training"].pop("output_bn_weight_decay", None)
    cfg["training"]["save_freq"] = 1
    cfg["heads"]["magface"]["detach_margin_norm"] = True
    ft_control = dict(
        schema=FINETUNE_DETACH_MARGIN_SCHEMA,
        purpose="diagnostic_control",
        changed_parameter="training.weight_decay+output_bn_freeze_bias+heads.magface.detach_margin_norm",
        reference_value={"weight_decay": 0.0001, "output_bn_freeze_bias": False, "detach_margin_norm": False},
        control_value={"weight_decay": 0.0, "output_bn_freeze_bias": True, "detach_margin_norm": True},
        parent_sha256=binding["sha256"],
        reference_selection=str(ref_selection_path.resolve()),
        reference_selection_sha256=ref_selection_sha,
        reference_checkpoint=str(ref_ckpt.resolve()),
        reference_sha256=ref_sha,
        reference_model_uid="magface-ad3b19a576ec14a0883c",
        reference_report="F:/ronbun_results/facex_workflow/reports/facex-1587c845c7c2c5f137132e9c",
    )
    cfg["campaign"].update(
        parent_checkpoint=binding["path"],
        parent_sha256=binding["sha256"],
        parent_binding=binding,
        finetune_control=ft_control,
        purpose="diagnostic_control",
    )
    selection = make_selection(
        cfg,
        [dict(dataset="survface", recipe="full_fr_resize_only", heads=["magface"], seeds=[8972])],
        out_root,
        model_root / "facex/diagnostics" / ALIGNMENT_CONTROL,
        purpose="diagnostic_control",
    )
    new_cfg = settings_for(selection, selection["rows"][0])
    require_only_finetune_detach_margin_control(ref_cfg, new_cfg, ref_payload["config"], _expected(new_cfg, selection["rows"][0]))
    return selection


def make_finetune_detach_margin_selection(baseline, control_root=None, output_root=None):
    """Build and save the immutable selection.json for the finetune_detach_margin_control campaign."""
    from .selected_campaign import save_selection
    selection = build_finetune_detach_margin_selection(baseline, control_root=control_root, output_root=output_root)
    path = Path(selection["root"]) / "selection.json"
    save_selection(path, selection)
    return path


def build_finetune_detach_margin_lr_0002_selection(baseline, control_root=None, output_root=None):
    """Reuse completed WD=0 Parent; keep B-1 settings and change only fine-tune training.lr 0.001 -> 0.0002."""
    from .selected_campaign import _expected, make_selection, settings_for
    from .ucface_controls import ALIGNMENT_CONTROL

    model_root, out_root, ref_selection_path, ref_selection_sha, binding, ref_cfg, ref_ckpt, ref_sha, ref_payload = (
        _prepare_single_row_control_base(
            baseline, control_root, output_root, FINETUNE_DETACH_MARGIN_LR_0002_SUBDIR, "finetune_detach_margin_lr_0002_control"
        )
    )
    cfg = copy.deepcopy(baseline)
    cfg["project"]["keep_raw_results"] = True
    cfg["training"]["lr"] = 0.0002
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["output_bn_freeze_bias"] = True
    cfg["training"].pop("output_bn_weight_decay", None)
    cfg["training"]["save_freq"] = 1
    cfg["heads"]["magface"]["detach_margin_norm"] = True
    ft_control = dict(
        schema=FINETUNE_DETACH_MARGIN_LR_0002_SCHEMA,
        purpose="diagnostic_control",
        changed_parameter="training.lr+weight_decay+output_bn_freeze_bias+heads.magface.detach_margin_norm",
        reference_value={"lr": 0.001, "weight_decay": 0.0001, "output_bn_freeze_bias": False, "detach_margin_norm": False},
        control_value={"lr": 0.0002, "weight_decay": 0.0, "output_bn_freeze_bias": True, "detach_margin_norm": True},
        parent_sha256=binding["sha256"],
        reference_selection=str(ref_selection_path.resolve()),
        reference_selection_sha256=ref_selection_sha,
        reference_checkpoint=str(ref_ckpt.resolve()),
        reference_sha256=ref_sha,
        reference_model_uid="magface-ad3b19a576ec14a0883c",
        reference_report="F:/ronbun_results/facex_workflow/reports/facex-1587c845c7c2c5f137132e9c",
    )
    cfg["campaign"].update(
        parent_checkpoint=binding["path"],
        parent_sha256=binding["sha256"],
        parent_binding=binding,
        finetune_control=ft_control,
        purpose="diagnostic_control",
    )
    selection = make_selection(
        cfg,
        [dict(dataset="survface", recipe="full_fr_resize_only", heads=["magface"], seeds=[8972])],
        out_root,
        model_root / "facex/diagnostics" / ALIGNMENT_CONTROL,
        purpose="diagnostic_control",
    )
    new_cfg = settings_for(selection, selection["rows"][0])
    require_only_finetune_detach_margin_lr_0002_control(
        ref_cfg, new_cfg, ref_payload["config"], _expected(new_cfg, selection["rows"][0])
    )
    return selection


def make_finetune_detach_margin_lr_0002_selection(baseline, control_root=None, output_root=None):
    """Build and save the immutable selection.json for the finetune_detach_margin_lr_0002_control campaign."""
    from .selected_campaign import save_selection
    selection = build_finetune_detach_margin_lr_0002_selection(baseline, control_root=control_root, output_root=output_root)
    path = Path(selection["root"]) / "selection.json"
    save_selection(path, selection)
    return path


def _eval_512_train_mode_checkpoint(ckpt_path, model, xs, diff_mask, device, *,
                                    expected_epoch=None, expected_wd=None,
                                    expected_output_bn_wd=None, expected_freeze_bias=False,
                                    expected_parent_hash=None, expected_train_list_hash=None,
                                    expected_cfg=None, expected_ft_control=None):
    import numpy as np
    from .checkpoint_manager import compute_config_hash
    payload = load_checkpoint(str(ckpt_path))
    ckpt_cfg = payload.get("config", {})
    if compute_config_hash(ckpt_cfg) != payload.get("config_hash"):
        raise ValueError(f"Checkpoint config_hash mismatch: {ckpt_path}")
    if expected_epoch is not None and int(payload.get("epoch", -1)) != int(expected_epoch):
        raise ValueError(f"Checkpoint epoch mismatch for {ckpt_path}: expected {expected_epoch}, found {payload.get('epoch')}")
    if expected_parent_hash is not None and (
            payload.get("parent_hash") != expected_parent_hash or ckpt_cfg.get("parent_hash") != expected_parent_hash):
        raise ValueError(f"Checkpoint parent_hash mismatch for {ckpt_path}")
    if expected_train_list_hash is not None and ckpt_cfg.get("dataset", {}).get("train_list_hash") != expected_train_list_hash:
        raise ValueError(f"Checkpoint train_list_hash mismatch for {ckpt_path}")
    if expected_wd is not None:
        actual_wd = ckpt_cfg.get("training", {}).get("weight_decay")
        if actual_wd is None or float(actual_wd) != float(expected_wd):
            raise ValueError(f"Checkpoint weight_decay mismatch for {ckpt_path}: expected {expected_wd}, found {actual_wd}")
        opt_groups = payload.get("optimizer_state_dict", {}).get("param_groups", [])
        if expected_output_bn_wd is not None:
            if (not isinstance(opt_groups, list) or len(opt_groups) != 2
                    or float(opt_groups[0].get("weight_decay", -1)) != float(expected_wd)
                    or float(opt_groups[1].get("weight_decay", -1)) != float(expected_output_bn_wd)):
                raise ValueError(f"Checkpoint optimizer param_groups mismatch for {ckpt_path}")
        elif not isinstance(opt_groups, list) or not opt_groups or any(
                g.get("weight_decay") is None or float(g.get("weight_decay")) != float(expected_wd) for g in opt_groups):
            raise ValueError(f"Checkpoint optimizer param_groups weight_decay mismatch for {ckpt_path}")
    if expected_freeze_bias:
        bn_bias = payload["backbone_state_dict"].get("bn.bias")
        if bn_bias is None or torch.count_nonzero(bn_bias).item() != 0:
            raise ValueError(f"Checkpoint bn.bias must be frozen at 0 for {ckpt_path}")
    if expected_cfg is not None:
        if (ckpt_cfg.get("code_fingerprints") != expected_cfg.get("code_fingerprints")
                and not w.source_transition(ckpt_cfg.get("code_fingerprints"), expected_cfg.get("code_fingerprints"))):
            raise ValueError(f"Unreviewed source transition in checkpoint {ckpt_path}")
        c_clean, e_clean = copy.deepcopy(ckpt_cfg), copy.deepcopy(expected_cfg)
        for item in (c_clean, e_clean):
            item.pop("config_hash", None)
            item.pop("code_fingerprints", None)
        if c_clean != e_clean:
            raise ValueError(f"Checkpoint config differs from expected for {ckpt_path}")
    if expected_ft_control is not None:
        meta = payload.get("extra_meta", payload.get("metadata", {}))
        exec_s = meta.get("execution_settings") if isinstance(meta, dict) else None
        if not isinstance(exec_s, dict) or exec_s.get("campaign", {}).get("finetune_control") != expected_ft_control:
            raise ValueError(f"Checkpoint finetune_control provenance mismatch for {ckpt_path}")

    sd = copy.deepcopy(payload["backbone_state_dict"])
    bn_w_norm = round(float(sd["bn.weight"].float().norm().item()), 4)
    bn_b_norm = round(float(sd["bn.bias"].float().norm().item()), 4)
    model.backbone.load_state_dict(sd, strict=True)
    model.backbone.train()
    chunks = []
    with torch.no_grad():
        for b in range(8):
            torch.manual_seed(8972 + b)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(8972 + b)
            chunks.append(model.backbone(xs[b * 64:(b + 1) * 64]).float())
        feats = torch.cat(chunks, dim=0)
        norms = feats.norm(dim=1).cpu().numpy()
        unit = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()
        cos = (unit @ unit.T)[diff_mask]
    return dict(
        checkpoint=str(ckpt_path),
        sha256=compute_sha256(str(ckpt_path)),
        epoch=int(payload.get("epoch", 0)),
        weight_decay=float(ckpt_cfg.get("training", {}).get("weight_decay", 0.0)),
        parent_hash=payload.get("parent_hash"),
        train_list_hash=ckpt_cfg.get("dataset", {}).get("train_list_hash"),
        norm_median=round(float(np.median(norms)), 2),
        norm_lt_10=int((norms < 10.0).sum()),
        diff_id_cos_mean=round(float(cos.mean()), 4),
        diff_id_cos_p95=round(float(np.quantile(cos, 0.95)), 4),
        bn_w_norm=bn_w_norm,
        bn_b_norm=bn_b_norm,
    )


def _diagnose_512_control_epochs(selection, *, expected_schema, control_key, expected_ctrl_wd,
                                  expected_output_bn_wd=None, expected_freeze_bias=False,
                                  epochs=(1, 2, 5, 10, 40), evaluate_live=False):
    import hashlib
    import numpy as np
    from .dataset import FaceImageDataset
    from .selected_campaign import _expected, final_path, settings_for

    ft_control = selection.get("settings", {}).get("campaign", {}).get("finetune_control")
    validate_finetune_control_spec(
        ft_control,
        selection.get("settings", {}).get("campaign", {}).get("parent_binding"),
        selection.get("settings", {}).get("training", {}).get("weight_decay"),
        training_cfg=selection.get("settings", {}).get("training"),
        head_cfg=selection.get("settings", {}).get("heads", {}).get("magface"),
    )
    if ft_control.get("schema") != expected_schema:
        raise ValueError(f"Expected {expected_schema}, found {ft_control.get('schema')!r}")
    row = selection["rows"][0]
    cfg = settings_for(selection, row)
    expected_ctrl_cfg = _expected(cfg, row)
    expected_ref_cfg = copy.deepcopy(expected_ctrl_cfg)
    expected_ref_cfg["training"]["lr"] = 0.001
    expected_ref_cfg["training"]["weight_decay"] = 0.0001
    expected_ref_cfg["training"].pop("output_bn_freeze_bias", None)
    expected_ref_cfg["training"].pop("output_bn_weight_decay", None)
    expected_ref_cfg["head"].pop("detach_margin_norm", None)
    parent_sha = cfg["campaign"]["parent_sha256"]
    train_list_sha = cfg["dataset"]["train_list_hash"]
    ctrl_dir = final_path(cfg, row).parent
    ref_dir = Path(ft_control["reference_checkpoint"]).parent
    ctrl_files = {int(e): ctrl_dir / f"finetune_mobilefacenet_1024d_magface_epoch_{int(e)}.pt" for e in epochs}
    ref_files = {int(e): ref_dir / f"finetune_mobilefacenet_1024d_magface_epoch_{int(e)}.pt" for e in epochs}
    any_ctrl_exists = any(p.is_file() for p in ctrl_files.values())

    live_ctrl = {}
    live_ref = {}
    if evaluate_live or any_ctrl_exists:
        ds = FaceImageDataset(
            cfg["dataset"]["data_root"],
            cfg["dataset"]["train_list"],
            target_size=tuple(cfg["dataset"]["input_size"]),
            pixel_mean=cfg["dataset"]["pixel_mean"],
            pixel_std=cfg["dataset"]["pixel_std"],
            is_train=True,
            seed=8972,
            alignment="resize_only",
            augmentation=cfg["augmentation"],
        )
        idx = np.random.default_rng(8972).choice(len(ds), 512, replace=False)
        idx_sha = hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()
        if idx_sha != FIXED_512_INDICES_SHA256:
            raise ValueError(f"512-sample index SHA-256 mismatch: {idx_sha}")
        device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
        xs = torch.stack([ds[int(i)][0] for i in idx]).to(device)
        ys = np.array([ds[int(i)][1] for i in idx])
        diff_mask = ys[:, None] != ys[None, :]
        model = w.build_model(cfg["backbone"], dict(cfg["heads"]["magface"], num_class=cfg["dataset"]["num_classes"])).to(device)
        for e, p in ctrl_files.items():
            if p.is_file():
                live_ctrl[e] = _eval_512_train_mode_checkpoint(
                    p, model, xs, diff_mask, device,
                    expected_epoch=e, expected_wd=expected_ctrl_wd,
                    expected_output_bn_wd=expected_output_bn_wd,
                    expected_freeze_bias=expected_freeze_bias,
                    expected_parent_hash=parent_sha, expected_train_list_hash=train_list_sha,
                    expected_cfg=expected_ctrl_cfg, expected_ft_control=ft_control,
                )
        for e, p in ref_files.items():
            if p.is_file():
                live_ref[e] = _eval_512_train_mode_checkpoint(
                    p, model, xs, diff_mask, device,
                    expected_epoch=e, expected_wd=0.0001,
                    expected_parent_hash=parent_sha, expected_train_list_hash=train_list_sha,
                    expected_cfg=expected_ref_cfg,
                )

    rows = []
    for e in epochs:
        e = int(e)
        if e in live_ref:
            ref_metrics = {"source": "live_checkpoint_eval", "checkpoint_exists": True, **live_ref[e]}
        else:
            cached = copy.deepcopy(REFERENCE_WD_0001_512_TRAIN_MODE.get(e))
            ref_metrics = {
                "source": "cached_verified_reference",
                "checkpoint_exists": ref_files[e].is_file(),
                "checkpoint": str(ref_files[e]),
                **cached,
            }
        if e in live_ctrl:
            ctrl_entry = {
                "status": "evaluated",
                "source": "live_checkpoint_eval",
                **live_ctrl[e],
                "delta_norm_lt_10_vs_ref": live_ctrl[e]["norm_lt_10"] - ref_metrics["norm_lt_10"],
                "delta_norm_median_vs_ref": round(live_ctrl[e]["norm_median"] - ref_metrics["norm_median"], 2),
            }
        else:
            ctrl_entry = {
                "status": "not_trained_yet",
                "expected_checkpoint": str(ctrl_files[e]),
            }
        rows.append({"epoch": e, "reference_wd_0001": ref_metrics, control_key: ctrl_entry})
    return dict(
        schema=expected_schema,
        sample_protocol=dict(
            seed=8972,
            sample_count=512,
            batch_size=64,
            num_batches=8,
            indices_sha256=FIXED_512_INDICES_SHA256,
            train_list_sha256=train_list_sha,
            mode="train_bn_with_epoch1_augmentation_and_fixed_batch_dropout_seed_8972_plus_b",
        ),
        parent_before_finetune={
            "checkpoint": cfg["campaign"]["parent_checkpoint"],
            **copy.deepcopy(REFERENCE_WD_0001_512_TRAIN_MODE["pre_finetune"]),
        },
        epochs=rows,
    )


def diagnose_finetune_wd_zero_epochs(selection, epochs=(1, 2, 5, 10, 40), *, evaluate_live=False):
    """Stage-1 diagnostic: fixed 512 SurvFace train images norm (<10 count & median) and inter-identity cosine."""
    return _diagnose_512_control_epochs(
        selection,
        expected_schema=FINETUNE_WD_ZERO_SCHEMA,
        control_key="control_wd_0",
        expected_ctrl_wd=0.0,
        epochs=epochs,
        evaluate_live=evaluate_live,
    )


def diagnose_finetune_output_bn_epochs(selection, epochs=(1, 2, 5, 10, 40), *, evaluate_live=False):
    """Stage-1 diagnostic for finetune_output_bn_control on the fixed 512 SurvFace train images."""
    return _diagnose_512_control_epochs(
        selection,
        expected_schema=FINETUNE_OUTPUT_BN_SCHEMA,
        control_key="control_output_bn",
        expected_ctrl_wd=0.0001,
        expected_output_bn_wd=0.0,
        expected_freeze_bias=True,
        epochs=epochs,
        evaluate_live=evaluate_live,
    )


def diagnose_finetune_wd_zero_freeze_bias_epochs(selection, epochs=(1, 2, 5, 10, 40), *, evaluate_live=False):
    """Stage-1 diagnostic for finetune_wd_zero_freeze_bias_control on the fixed 512 SurvFace train images."""
    return _diagnose_512_control_epochs(
        selection,
        expected_schema=FINETUNE_WD_ZERO_FREEZE_BIAS_SCHEMA,
        control_key="control_wd_0_freeze_bias",
        expected_ctrl_wd=0.0,
        expected_output_bn_wd=None,
        expected_freeze_bias=True,
        epochs=epochs,
        evaluate_live=evaluate_live,
    )


def diagnose_finetune_detach_margin_epochs(selection, epochs=(1, 2, 5, 10, 40), *, evaluate_live=False):
    """Stage-1 diagnostic for finetune_detach_margin_control on the fixed 512 SurvFace train images."""
    return _diagnose_512_control_epochs(
        selection,
        expected_schema=FINETUNE_DETACH_MARGIN_SCHEMA,
        control_key="control_detach_margin",
        expected_ctrl_wd=0.0,
        expected_output_bn_wd=None,
        expected_freeze_bias=True,
        epochs=epochs,
        evaluate_live=evaluate_live,
    )


def diagnose_finetune_detach_margin_lr_0002_epochs(selection, epochs=(1, 2, 5, 10, 40), *, evaluate_live=False):
    """Stage-1 diagnostic for finetune_detach_margin_lr_0002_control on the fixed 512 SurvFace train images."""
    return _diagnose_512_control_epochs(
        selection,
        expected_schema=FINETUNE_DETACH_MARGIN_LR_0002_SCHEMA,
        control_key="control_detach_margin_lr_0002",
        expected_ctrl_wd=0.0,
        expected_output_bn_wd=None,
        expected_freeze_bias=True,
        epochs=epochs,
        evaluate_live=evaluate_live,
    )


def launch_parent_control(base_dir, cfg, *, resume_checkpoint=None, background=False):
    """Same canonical training entry point for notebooks and detached agent execution."""
    plan = inspect_parent_control(cfg)
    validate_control_resume(cfg, resume_checkpoint)
    root = Path(cfg["parent_control"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    # A run's scientific plan is immutable. Epoch cap is an execution limit only.
    w.write_json(root / "control.json", cfg["parent_control"], immutable=True)
    guide = root / "HOW_TO_READ.md"
    if not (root / "plan.json").exists():
        w.write_json(root / "plan.json", {k:v for k,v in plan.items() if k not in ("free_bytes", "stop_after_epoch")}, immutable=True)
    attempt_plan = root / "attempt_plans" / f"{uuid.uuid4().hex}.json"
    w.write_json(attempt_plan, plan, immutable=True)
    source_archive = root / "provenance" / f"parent_controls_{compute_sha256(__file__)}.py"
    source_archive.parent.mkdir(exist_ok=True)
    if not source_archive.exists():
        shutil.copyfile(__file__, source_archive)
    elif compute_sha256(str(source_archive)) != compute_sha256(__file__):
        raise ValueError("Archived control source was modified")
    if not guide.exists():
        shutil.copyfile(GUIDE, guide)
    resume = {RUN_ID: str(resume_checkpoint)} if resume_checkpoint else {}
    if not background:
        from .process_runner import launch_campaign
        result = launch_campaign(base_dir, cfg, "pretrain", resume)
        return dict(results=result, summary=str(summarize_parent_control(cfg)))
    ticket = Path(cfg["paths"]["launch_dir"]) / uuid.uuid4().hex
    ticket.mkdir(parents=True)
    effective = copy.deepcopy(cfg)
    effective["runtime"] = {**effective.get("runtime", {}), "stop_file": str(ticket / "stop.request"),
        "log_file": str(ticket / "console.log"), "result_file": str(ticket / "result.json"),
        "attempt_plan":str(attempt_plan)}
    request = dict(settings=effective, stage="pretrain", resume=resume, smoke=False)
    request_path = ticket / "request.json"
    w.write_json(request_path, request, immutable=True)
    command = [sys.executable, "-u", str(Path(base_dir) / "fine-tune/scripts/run_parent_control.py"),
               "--request", str(request_path)]
    with (ticket / "process_output.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=base_dir, stdout=output, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    receipt = dict(pid=process.pid, request=str(request_path), log=str(ticket / "console.log"),
        output=str(ticket / "process_output.log"), stop_file=effective["runtime"]["stop_file"],
        result=effective["runtime"]["result_file"], interpreter=sys.executable,
        command=command, created_at=datetime.now(timezone(timedelta(hours=9))).isoformat())
    w.write_json(ticket / "launch.json", receipt, immutable=True)
    return receipt


def checkpoint_health(path):
    payload = load_checkpoint(str(path))
    state = payload["backbone_state_dict"]
    gamma, variance = state["bn.weight"], state["bn.running_var"]
    return dict(path=str(path), sha256=compute_sha256(str(path)), epoch=payload["epoch"],
        completed=payload["completed"], config_hash=payload["config_hash"],
        weight_decay=payload["config"]["training"]["weight_decay"],
        active_gamma=int((gamma.abs() > 0.01).sum()), active_variance=int((variance > 0.001).sum()),
        linear_rows_gt_001=int((state["linear.weight"].norm(dim=1) > 0.01).sum()),
        loss=payload.get("extra_meta", {}).get("loss"),
        optimizer_steps=float(next(iter(payload["optimizer_state_dict"]["state"].values()))["step"]))


def summarize_parent_control(cfg):
    root = Path(cfg["parent_control"]["output_root"])
    reference = Path(cfg["parent_control"]["reference"]["path"])
    run_dir = w.checkpoint_path(cfg, "pretrain", "arcface", 8972).parent
    rows = []
    for epoch in range(1, 41):
        filename = f"pretrain_mobilefacenet_1024d_epoch_{epoch}.pt"
        candidate = run_dir / filename
        if candidate.is_file():
            rows.append(dict(epoch=epoch, reference=checkpoint_health(reference.with_name(filename)),
                             control=checkpoint_health(candidate)))
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    out = root / "summaries" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    summary = dict(schema=SCHEMA, purpose="diagnostic_control", parent_control=cfg["parent_control"],
        completed=bool(rows and rows[-1]["epoch"] == 40 and rows[-1]["control"]["completed"]),
        available_paired_epochs=[r["epoch"] for r in rows], paired_checkpoint_health=rows,
        active_criteria=dict(abs_bn_gamma_gt=0.01, bn_running_variance_gt=0.001),
        interpretation="Counts are diagnostics, not accuracy or exact effective rank. Recognition evaluation is separate.",
        keep_raw_results=cfg["project"]["keep_raw_results"], raw_location=str(run_dir),
        reference_results_immutable=True, exact_ucface_reproduction=False)
    w.write_json(out / "summary.json", summary, immutable=True)
    shutil.copyfile(GUIDE, out / "HOW_TO_READ.md")
    data = (out / "summary.json").read_text(encoding="utf-8")
    w.write_json(out / "bundle.json", dict(files=["summary.json", "HOW_TO_READ.md"],
        summary_bytes=(out / "summary.json").stat().st_size,
        estimated_summary_tokens=len(data), token_estimate_method="Conservative one Unicode character per token; not an exact tokenizer count",
        suggested_upload_order=["HOW_TO_READ.md", "summary.json"]), immutable=True)
    return out / "summary.json"
