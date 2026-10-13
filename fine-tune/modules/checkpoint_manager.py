"""
Checkpoint Management, Parent Hash Verification, and Model Export Utilities.
Enforces the pretrain -> finetune contract by requiring parent weights and hash verification.
"""

import os
import json
import copy
import random
import hashlib
import logging
import collections
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("fine_tune.checkpoint")


def compute_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    if not os.path.exists(file_path):
        logger.error(f"Cannot compute hash. File does not exist: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error computing SHA-256 for {file_path}: {e}", exc_info=True)
        raise


def load_checkpoint(file_path: str, map_location: Any = "cpu", **kwargs) -> Any:
    """Load a PyTorch checkpoint with backwards compatibility across PyTorch versions.

    In PyTorch 2.6+, the weights_only parameter defaults to True. This causes
    UnpicklingError when loading checkpoints containing full training states
    (e.g., optimizer_state_dict, scheduler_state_dict, numpy RNG state arrays,
    custom metadata dictionaries).
    
    This function automatically supplies weights_only=False if supported by the
    installed PyTorch version, with graceful fallback for older versions.
    """
    if not os.path.exists(file_path):
        logger.error(f"Cannot load checkpoint. File does not exist: {file_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    load_kwargs = dict(map_location=map_location, **kwargs)
    try:
        import inspect
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters and "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = False
    except Exception as e:
        logger.warning(f"Could not inspect torch.load signature: {e}")

    try:
        return torch.load(file_path, **load_kwargs)
    except TypeError as te:
        if "weights_only" in str(te) and "weights_only" in load_kwargs:
            load_kwargs.pop("weights_only", None)
            return torch.load(file_path, **load_kwargs)
        logger.error(f"TypeError loading checkpoint from {file_path}: {te}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {file_path}: {e}", exc_info=True)
        raise


def verify_parent_checkpoint(
    parent_path: str,
    expected_hash: Optional[str] = None,
    require_hash: bool = True,
    target_epoch: Optional[int] = None,
    allow_partial: bool = False,
    expected_seed: Optional[int] = None,
    require_completed: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Verify and load parent pre-trained backbone checkpoint.

    Rules:
    1. Parent checkpoint MUST exist. If missing, raises FileNotFoundError.
       Random initialization during fine-tuning is strictly disallowed.
    2. Smoke-test checkpoints cannot be used as parent for formal training.
    3. If require_hash is True, expected_hash MUST be provided and match.
    4. Completed state & epoch verification:
       - If completed is False, rejects unless allow_partial=True.
       - If require_completed is True, strictly requires completed=True.
       - If target_epoch is provided, requires actual_epoch >= target_epoch.
       - If expected_seed is provided, requires seed match.

    Args:
        parent_path (str): Path to parent backbone checkpoint.
        expected_hash (str, optional): Expected SHA-256 hash string.
        require_hash (bool): If True, requires expected_hash to be provided. Default: True.
        target_epoch (int, optional): Expected completed epoch (e.g. 40).
        allow_partial (bool): If True, allows intermediate/partial checkpoint inspection.
        expected_seed (int, optional): Expected random seed of the parent run.
        require_completed (bool): If True, strictly requires completed flag to be True.

    Returns:
        Tuple[dict, str]: (loaded_state_dict, computed_sha256)
    """
    if not os.path.exists(parent_path):
        logger.error(
            f"[Pretrain -> Finetune Violation] Parent checkpoint not found at: {parent_path}. "
            "Fine-tuning strictly requires a validated pre-trained backbone checkpoint."
        )
        raise FileNotFoundError(
            f"[Pretrain -> Finetune Violation] Parent checkpoint not found at: {parent_path}. "
            "Fine-tuning strictly requires a validated pre-trained backbone checkpoint."
        )

    # Smoke checkpoint guard: prevent temporary/dummy checkpoints from becoming formal parents
    basename = os.path.basename(parent_path).lower()
    dirname = os.path.dirname(parent_path).lower()
    if "smoke" in basename or "smoke" in dirname:
        logger.error(
            f"[Integrity Violation] Smoke-test checkpoint '{parent_path}' cannot be used as "
            "parent checkpoint for formal fine-tuning!"
        )
        raise ValueError(
            f"Smoke-test checkpoint cannot be used as formal pre-trained parent: {parent_path}"
        )

    actual_hash = compute_sha256(parent_path)
    logger.info(f"Verified parent checkpoint: {parent_path} (SHA-256: {actual_hash})")

    if require_hash:
        if not expected_hash or not expected_hash.strip():
            logger.error(
                "[Integrity Violation] Formal fine-tuning strictly requires expected_hash. "
                "Bypassing hash verification is disallowed."
            )
            raise ValueError(
                "expected_hash is required for formal fine-tuning. "
                "Ensure pretrain hash JSON exists and provides a valid SHA-256 hash."
            )

    if expected_hash is not None and expected_hash.strip():
        if actual_hash.lower() != expected_hash.strip().lower():
            logger.error(
                f"[Integrity Error] Parent checkpoint hash mismatch! "
                f"Expected: {expected_hash}, Actual: {actual_hash}"
            )
            raise ValueError(
                f"[Integrity Error] Parent checkpoint hash mismatch! "
                f"Expected: {expected_hash}, Actual: {actual_hash}"
            )
        logger.info("Parent checkpoint SHA-256 integrity check PASSED.")

    # Load and deeply inspect checkpoint payload
    checkpoint = load_checkpoint(parent_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        # Inspect top-level, extra_meta, and metadata for smoke flags
        is_smoke = (
            checkpoint.get("is_smoke") is True
            or checkpoint.get("extra_meta", {}).get("is_smoke") is True
            or checkpoint.get("metadata", {}).get("is_smoke") is True
        )
        if is_smoke:
            logger.error(
                f"[Integrity Violation] Checkpoint payload at '{parent_path}' is explicitly marked as a smoke test!"
            )
            raise ValueError(
                f"Smoke-test checkpoint cannot be used as formal pre-trained parent: {parent_path}"
            )

        # Ensure valid backbone weights are present
        has_backbone = "backbone_state_dict" in checkpoint or "state_dict" in checkpoint
        if not has_backbone:
            logger.error(f"[Integrity Error] Checkpoint at '{parent_path}' contains no valid backbone state dict!")
            raise ValueError(f"Checkpoint contains no valid backbone weights: {parent_path}")

        meta = checkpoint.get("extra_meta", checkpoint.get("metadata", {}))

        # Check formal training completion
        is_completed = checkpoint.get("completed", meta.get("completed", None))
        actual_epoch = checkpoint.get("epoch", meta.get("epoch", None))

        if not allow_partial:
            if is_completed is False:
                logger.error(f"[Parent Violation] Parent checkpoint '{parent_path}' is marked completed=False.")
                raise ValueError(
                    f"[Parent Verification Error] Checkpoint '{parent_path}' is incomplete (completed=False). "
                    "Formal fine-tuning requires a completed pre-trained parent checkpoint."
                )
            if require_completed and is_completed is not True:
                logger.error(f"[Parent Violation] Parent checkpoint '{parent_path}' is not marked completed=True.")
                raise ValueError(
                    f"[Parent Verification Error] Checkpoint '{parent_path}' is incomplete or lacks completed=True. "
                    "Formal fine-tuning requires a verified completed pre-trained parent."
                )
            if target_epoch is not None:
                if is_completed is not True:
                    logger.error(f"[Parent Violation] Parent checkpoint '{parent_path}' is not marked completed=True.")
                    raise ValueError(
                        f"[Parent Verification Error] Checkpoint '{parent_path}' is incomplete or lacks completed=True. "
                        "Formal fine-tuning requires a verified completed pre-trained parent."
                    )
                if actual_epoch is None:
                    logger.error(f"[Parent Violation] Parent checkpoint '{parent_path}' lacks recorded epoch.")
                    raise ValueError(
                        f"[Parent Verification Error] Parent checkpoint '{parent_path}' lacks recorded epoch (expected target_epoch {target_epoch})."
                    )
                if int(actual_epoch) < int(target_epoch):
                    logger.error(
                        f"[Parent Violation] Parent epoch {actual_epoch} < target_epoch {target_epoch}."
                    )
                    raise ValueError(
                        f"[Parent Verification Error] Parent checkpoint '{parent_path}' reached epoch {actual_epoch}, "
                        f"which is less than target epoch {target_epoch}."
                    )

            # Check seed consistency if expected_seed is provided
            if expected_seed is not None:
                ckpt_seed = checkpoint.get("seed", meta.get("seed", None))
                if ckpt_seed is None:
                    logger.error(f"[Parent Violation] Parent checkpoint '{parent_path}' lacks seed metadata.")
                    raise ValueError(
                        f"[Parent Verification Error] Parent checkpoint '{parent_path}' lacks seed metadata (expected {expected_seed})."
                    )
                if int(ckpt_seed) != int(expected_seed):
                    logger.error(
                        f"[Parent Violation] Seed mismatch: expected {expected_seed}, found {ckpt_seed}."
                    )
                    raise ValueError(
                        f"[Parent Verification Error] Seed mismatch! Expected seed {expected_seed}, "
                        f"but parent checkpoint specifies seed {ckpt_seed}."
                    )

            # Check stage consistency if target_epoch is specified
            if target_epoch is not None:
                stage = meta.get("stage", checkpoint.get("stage", None))
                if stage is None:
                    logger.error(f"[Parent Violation] Parent checkpoint '{parent_path}' lacks stage metadata.")
                    raise ValueError(
                        f"[Parent Verification Error] Parent checkpoint '{parent_path}' lacks stage metadata (expected 'pretrain')."
                    )
                if str(stage).lower() != "pretrain":
                    logger.error(f"[Parent Violation] Expected stage 'pretrain', but found '{stage}'.")
                    raise ValueError(
                        f"[Parent Verification Error] Expected stage 'pretrain', but parent checkpoint has stage '{stage}'."
                    )

    return checkpoint, actual_hash


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash for a config dictionary."""
    clean_dict = {k: v for k, v in config_dict.items() if k not in ("config_hash", "sha256")}

    def _json_default(obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    canonical_json = json.dumps(clean_dict, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def extract_model_config(model: nn.Module) -> Dict[str, Any]:
    """Extract architectural configuration from model backbone and head."""
    raw_model = model.module if hasattr(model, "module") else model
    config: Dict[str, Any] = {}

    # Backbone attributes
    backbone = getattr(raw_model, "backbone", raw_model)
    backbone_type = backbone.__class__.__name__
    feat_dim = None
    if hasattr(backbone, "linear") and hasattr(backbone.linear, "out_features"):
        feat_dim = int(backbone.linear.out_features)
    elif hasattr(backbone, "embedding_size"):
        feat_dim = int(backbone.embedding_size)

    dropout_rate = getattr(backbone, "dropout_rate", 0.0)
    out_h = getattr(backbone, "out_h", None)
    out_w = getattr(backbone, "out_w", None)

    config["backbone"] = {
        "type": backbone_type,
        "feat_dim": feat_dim,
        "dropout_rate": round(float(dropout_rate), 4),
    }
    if out_h is not None:
        config["backbone"]["out_h"] = int(out_h)
    if out_w is not None:
        config["backbone"]["out_w"] = int(out_w)

    # Head attributes
    head = getattr(raw_model, "head", None)
    if head is not None:
        head_type = head.__class__.__name__
        head_dict: Dict[str, Any] = {"type": head_type}

        weight = getattr(head, "kernel", getattr(head, "weight", None))
        if weight is not None and hasattr(weight, "shape"):
            head_dict["feat_dim"] = int(weight.shape[0])
            head_dict["num_class"] = int(weight.shape[1])
        elif hasattr(head, "num_class"):
            head_dict["num_class"] = int(head.num_class)

        # Head specific hyperparameters
        if hasattr(head, "margin_arc"):
            head_dict["margin_arc"] = round(float(head.margin_arc), 4)
        if hasattr(head, "margin_am"):
            head_dict["margin_am"] = round(float(head.margin_am), 4)
        if hasattr(head, "margin"):
            head_dict["margin"] = round(float(head.margin), 4)
        if hasattr(head, "scale"):
            head_dict["scale"] = round(float(head.scale), 4)
        elif hasattr(head, "s"):
            head_dict["scale"] = round(float(head.s), 4)

        if hasattr(head, "m"):
            head_dict["m"] = round(float(head.m), 4)
        if hasattr(head, "h"):
            head_dict["h"] = round(float(head.h), 4)
        if hasattr(head, "t_alpha"):
            head_dict["t_alpha"] = round(float(head.t_alpha), 4)

        if hasattr(head, "l_a"):
            head_dict["l_a"] = round(float(head.l_a), 4)
        if hasattr(head, "u_a"):
            head_dict["u_a"] = round(float(head.u_a), 4)
        if hasattr(head, "l_margin"):
            head_dict["l_margin"] = round(float(head.l_margin), 4)
        if hasattr(head, "u_margin"):
            head_dict["u_margin"] = round(float(head.u_margin), 4)
        if hasattr(head, "lamda"):
            head_dict["lamda"] = round(float(head.lamda), 4)

        config["head"] = head_dict

    return config


def extract_train_config(
    config: Optional[Dict[str, Any]] = None,
    train_loader: Optional[Any] = None,
    val_loader: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    seed: Optional[int] = None,
    stage: Optional[str] = None,
    parent_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract and bind training, dataset, preprocessing, and scheduler hyperparameters.
    Captures live DataLoader batch sizes, data roots, index file SHA-256 hashes,
    milestones, and decay ratios to guarantee complete execution identity.
    """
    cfg = config or {}
    train_part: Dict[str, Any] = {}

    # 1. Dataset & Preprocessing
    ds_cfg = cfg.get("dataset", {})
    train_ds = getattr(train_loader, "dataset", None) if train_loader is not None else None
    val_ds = getattr(val_loader, "dataset", None) if val_loader is not None else None

    # Unwrap dataset wrappers (Subset, ConcatDataset, etc.)
    curr_tds = train_ds
    while curr_tds is not None and hasattr(curr_tds, "dataset"):
        curr_tds = curr_tds.dataset
    curr_vds = val_ds
    while curr_vds is not None and hasattr(curr_vds, "dataset"):
        curr_vds = curr_vds.dataset

    # Data paths
    eff_data_root = ""
    if curr_tds is not None and hasattr(curr_tds, "data_root") and curr_tds.data_root:
        eff_data_root = str(curr_tds.data_root)
    elif ds_cfg:
        eff_data_root = str(ds_cfg.get("data_root", ds_cfg.get("raw_root", "")))

    train_list = ""
    if curr_tds is not None and hasattr(curr_tds, "index_file") and curr_tds.index_file:
        train_list = str(curr_tds.index_file)
    elif ds_cfg:
        train_list = str(ds_cfg.get("train_list", ""))

    val_list = ""
    if curr_vds is not None and hasattr(curr_vds, "index_file") and curr_vds.index_file:
        val_list = str(curr_vds.index_file)
    elif ds_cfg:
        val_list = str(ds_cfg.get("val_list") or "")

    # Resolve paths if relative to eff_data_root
    resolved_train_list = train_list
    if resolved_train_list and not os.path.isabs(resolved_train_list) and not os.path.isfile(resolved_train_list) and eff_data_root:
        cand_t = os.path.join(eff_data_root, resolved_train_list)
        if os.path.isfile(cand_t):
            resolved_train_list = cand_t

    resolved_val_list = val_list
    if resolved_val_list and not os.path.isabs(resolved_val_list) and not os.path.isfile(resolved_val_list) and eff_data_root:
        cand_v = os.path.join(eff_data_root, resolved_val_list)
        if os.path.isfile(cand_v):
            resolved_val_list = cand_v

    # Compute index file hashes deterministically
    train_list_hash = None
    if resolved_train_list and os.path.isfile(resolved_train_list):
        try:
            train_list_hash = compute_sha256(resolved_train_list)
        except Exception as e:
            logger.warning(f"Could not compute sha256 for train_list '{resolved_train_list}': {e}")
            train_list_hash = None
    elif ds_cfg and ds_cfg.get("train_list_hash"):
        train_list_hash = str(ds_cfg["train_list_hash"])

    val_list_hash = None
    if resolved_val_list and os.path.isfile(resolved_val_list):
        try:
            val_list_hash = compute_sha256(resolved_val_list)
        except Exception as e:
            logger.warning(f"Could not compute sha256 for val_list '{resolved_val_list}': {e}")
            val_list_hash = None
    elif ds_cfg and ds_cfg.get("val_list_hash"):
        val_list_hash = str(ds_cfg["val_list_hash"])

    # Preprocessing & Dataset
    has_ds = bool(ds_cfg or eff_data_root or train_list or (curr_tds is not None and hasattr(curr_tds, "target_size")))
    if has_ds:
        input_size = None
        if curr_tds is not None and hasattr(curr_tds, "target_size") and curr_tds.target_size:
            input_size = list(curr_tds.target_size)
        elif ds_cfg and "input_size" in ds_cfg:
            input_size = list(ds_cfg.get("input_size", [112, 112]))
        else:
            input_size = [112, 112]

        pixel_mean = None
        if curr_tds is not None and hasattr(curr_tds, "pixel_mean"):
            pixel_mean = float(curr_tds.pixel_mean)
        elif ds_cfg and "pixel_mean" in ds_cfg:
            pixel_mean = float(ds_cfg["pixel_mean"])
        else:
            pixel_mean = 127.5

        pixel_std = None
        if curr_tds is not None and hasattr(curr_tds, "pixel_std"):
            pixel_std = float(curr_tds.pixel_std)
        elif ds_cfg and "pixel_std" in ds_cfg:
            pixel_std = float(ds_cfg["pixel_std"])
        else:
            pixel_std = 128.0

        train_part["preprocessing"] = {
            "sampling": "epoch_index_rng_v1",
            "input_size": input_size if input_size is not None else [112, 112],
            "pixel_mean": pixel_mean if pixel_mean is not None else 127.5,
            "pixel_std": pixel_std if pixel_std is not None else 128.0,
        }
        train_part["dataset"] = {
            "name": str(ds_cfg.get("name", "VGGFace2")),
            "num_classes": int(ds_cfg.get("num_classes", 8631)),
            "data_root": eff_data_root,
            "raw_root": str(ds_cfg.get("raw_root", eff_data_root)),
            "train_list": train_list,
            "train_list_hash": train_list_hash,
            "val_list": val_list,
            "val_list_hash": val_list_hash,
        }

    # 2. Training & optimization
    train_cfg = cfg.get("training", {})
    eff_batch_size = None
    if train_loader is not None:
        if getattr(train_loader, "batch_size", None) is not None:
            eff_batch_size = int(train_loader.batch_size)
        elif hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "batch_size"):
            eff_batch_size = int(train_loader.batch_sampler.batch_size)
    if eff_batch_size is None and "batch_size" in train_cfg:
        eff_batch_size = int(train_cfg["batch_size"])

    eff_milestones = None
    if scheduler is not None and hasattr(scheduler, "milestones"):
        ms = scheduler.milestones
        if isinstance(ms, (dict, collections.abc.Mapping)):
            eff_milestones = sorted([int(m) for m in ms.keys()])
        else:
            eff_milestones = sorted([int(m) for m in ms])
    elif "milestones" in train_cfg:
        eff_milestones = sorted([int(m) for m in train_cfg["milestones"]])

    eff_lr_decay_ratio = None
    if scheduler is not None and hasattr(scheduler, "gamma"):
        eff_lr_decay_ratio = round(float(scheduler.gamma), 4)
    elif "lr_decay_ratio" in train_cfg:
        eff_lr_decay_ratio = round(float(train_cfg["lr_decay_ratio"]), 4)

    opt_name = None
    if optimizer is not None:
        opt_name = optimizer.__class__.__name__.lower()
    elif "optimizer" in train_cfg:
        opt_name = str(train_cfg["optimizer"]).lower()

    lr = None
    if optimizer is not None and hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0 and "lr" in optimizer.param_groups[0]:
        lr = float(optimizer.param_groups[0]["lr"])
    elif "lr" in train_cfg:
        lr = float(train_cfg["lr"])

    weight_decay = None
    if optimizer is not None and hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0 and "weight_decay" in optimizer.param_groups[0]:
        weight_decay = float(optimizer.param_groups[0]["weight_decay"])
    elif "weight_decay" in train_cfg:
        weight_decay = float(train_cfg["weight_decay"])

    total_epochs = None
    if "epoches" in train_cfg:
        total_epochs = int(train_cfg["epoches"])
    elif "total_epochs" in train_cfg:
        total_epochs = int(train_cfg["total_epochs"])

    if train_cfg or train_loader is not None or scheduler is not None or optimizer is not None:
        training_dict: Dict[str, Any] = {
            "optimizer": opt_name if opt_name is not None else "adam",
            "lr": lr if lr is not None else 0.001,
            "weight_decay": weight_decay if weight_decay is not None else 0.0001,
            "total_epochs": total_epochs if total_epochs is not None else 40,
            "batch_size": eff_batch_size if eff_batch_size is not None else 64,
            "milestones": eff_milestones if eff_milestones is not None else [10, 20, 30],
            "lr_decay_ratio": eff_lr_decay_ratio if eff_lr_decay_ratio is not None else 0.1,
        }
        train_part["training"] = training_dict

    # Identifiers
    eff_seed = seed
    if eff_seed is None and "project" in cfg and "seed" in cfg["project"]:
        eff_seed = cfg["project"]["seed"]
    if eff_seed is not None:
        train_part["seed"] = int(eff_seed)

    eff_stage = stage
    if eff_stage is None and "stage" in cfg:
        eff_stage = cfg["stage"]
    elif eff_stage is None and "project" in cfg and "stage" in cfg["project"]:
        eff_stage = cfg["project"]["stage"]
    if eff_stage is not None:
        train_part["stage"] = str(eff_stage)

    if parent_hash is not None:
        train_part["parent_hash"] = str(parent_hash)

    return train_part


def extract_model_and_train_config(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    stage: Optional[str] = None,
    head_type: Optional[str] = None,
    parent_hash: Optional[str] = None,
    train_loader: Optional[Any] = None,
    val_loader: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    optimizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Combine live model architecture with training hyperparameter config."""
    full_cfg = extract_model_config(model)

    if head_type is not None and "head" in full_cfg:
        full_cfg["head"].setdefault("type", str(head_type))

    train_part = extract_train_config(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        seed=seed,
        stage=stage,
        parent_hash=parent_hash,
    )

    # Sync num_classes into dataset section if present
    if "dataset" in train_part and "head" in full_cfg and "num_class" in full_cfg["head"]:
        train_part["dataset"].setdefault("num_classes", int(full_cfg["head"]["num_class"]))

    full_cfg.update(train_part)
    return full_cfg


def compare_configs(
    ckpt_cfg: Dict[str, Any],
    current_cfg: Dict[str, Any],
    full_strict: bool = False,
) -> List[str]:
    """Compare checkpoint configuration against current configuration.
    Returns a list of human-readable mismatch descriptions.
    
    If full_strict is True (e.g. when formal expected_config is provided),
    any section or parameter present in one but missing or differing in the other is a mismatch.
    """
    mismatches: List[str] = []

    def _diff_val(name: str, v1: Any, v2: Any):
        if (v1 is None) != (v2 is None):
            mismatches.append(f"{name} mismatch: checkpoint={v1}, current={v2}")
            return
        if v1 is None and v2 is None:
            return
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(float(v1) - float(v2)) > 1e-4:
                mismatches.append(f"{name} mismatch: checkpoint={v1}, current={v2}")
        elif isinstance(v1, list) and isinstance(v2, list):
            if v1 != v2:
                mismatches.append(f"{name} mismatch: checkpoint={v1}, current={v2}")
        elif str(v1).lower() != str(v2).lower():
            mismatches.append(f"{name} mismatch: checkpoint={v1}, current={v2}")

    def _diff_path(name: str, p1: Any, p2: Any):
        if (not p1) != (not p2):
            mismatches.append(f"{name} mismatch: checkpoint={p1}, current={p2}")
            return
        if not p1 and not p2:
            return
        norm1 = os.path.normcase(os.path.normpath(str(p1)))
        norm2 = os.path.normcase(os.path.normpath(str(p2)))
        if norm1 != norm2:
            mismatches.append(f"{name} mismatch: checkpoint={p1}, current={p2}")

    # 1. Backbone comparison
    ckpt_bb = ckpt_cfg.get("backbone", {})
    curr_bb = current_cfg.get("backbone", {})
    for key in ["type", "feat_dim", "dropout_rate", "out_h", "out_w"]:
        if full_strict:
            if key in ckpt_bb or key in curr_bb:
                _diff_val(f"Backbone {key}", ckpt_bb.get(key), curr_bb.get(key))
        else:
            if key in ckpt_bb and key in curr_bb:
                _diff_val(f"Backbone {key}", ckpt_bb[key], curr_bb[key])

    # 2. Head comparison
    ckpt_hd = ckpt_cfg.get("head", {})
    curr_hd = current_cfg.get("head", {})
    head_keys = [
        "type", "feat_dim", "num_class", "margin_arc", "margin_am", "margin", "scale",
        "m", "h", "t_alpha", "l_a", "u_a", "l_margin", "u_margin", "lamda"
    ]
    for key in head_keys:
        if full_strict:
            if key in ckpt_hd or key in curr_hd:
                _diff_val(f"Head {key}", ckpt_hd.get(key), curr_hd.get(key))
        else:
            if key in ckpt_hd and key in curr_hd:
                _diff_val(f"Head {key}", ckpt_hd[key], curr_hd[key])

    # 3. Preprocessing comparison
    ckpt_pre = ckpt_cfg.get("preprocessing", {})
    curr_pre = current_cfg.get("preprocessing", {})
    if full_strict and (ckpt_pre or curr_pre):
        for key in ["input_size", "pixel_mean", "pixel_std", "sampling"]:
            _diff_val(f"Preprocessing {key}", ckpt_pre.get(key), curr_pre.get(key))
    else:
        for key in ["input_size", "pixel_mean", "pixel_std", "sampling"]:
            if key in ckpt_pre and key in curr_pre:
                _diff_val(f"Preprocessing {key}", ckpt_pre[key], curr_pre[key])

    # 4. Dataset comparison
    ckpt_ds = ckpt_cfg.get("dataset", {})
    curr_ds = current_cfg.get("dataset", {})
    if full_strict and (ckpt_ds or curr_ds):
        for key in ["name", "num_classes"]:
            _diff_val(f"Dataset {key}", ckpt_ds.get(key), curr_ds.get(key))
        for key in ["data_root", "raw_root", "train_list", "val_list"]:
            _diff_path(f"Dataset {key}", ckpt_ds.get(key), curr_ds.get(key))
        for key in ["train_list_hash", "val_list_hash"]:
            _diff_val(f"Dataset {key}", ckpt_ds.get(key), curr_ds.get(key))
    else:
        for key in ["name", "num_classes"]:
            if key in ckpt_ds and key in curr_ds:
                _diff_val(f"Dataset {key}", ckpt_ds[key], curr_ds[key])
        for key in ["data_root", "raw_root", "train_list", "val_list"]:
            if key in ckpt_ds and key in curr_ds:
                _diff_path(f"Dataset {key}", ckpt_ds[key], curr_ds[key])
        for key in ["train_list_hash", "val_list_hash"]:
            if key in ckpt_ds and key in curr_ds:
                _diff_val(f"Dataset {key}", ckpt_ds[key], curr_ds[key])

    # 5. Training / Optimization comparison
    ckpt_opt = ckpt_cfg.get("training", {})
    curr_opt = current_cfg.get("training", {})
    if full_strict and (ckpt_opt or curr_opt):
        for key in ["optimizer", "lr", "weight_decay"]:
            _diff_val(f"Optimizer {key}", ckpt_opt.get(key), curr_opt.get(key))
        for key in ["batch_size", "milestones", "lr_decay_ratio", "total_epochs"]:
            _diff_val(f"Training {key}", ckpt_opt.get(key), curr_opt.get(key))
    else:
        for key in ["optimizer", "lr", "weight_decay"]:
            if key in ckpt_opt and key in curr_opt:
                _diff_val(f"Optimizer {key}", ckpt_opt[key], curr_opt[key])
        for key in ["batch_size", "milestones", "lr_decay_ratio", "total_epochs"]:
            if key in ckpt_opt and key in curr_opt:
                _diff_val(f"Training {key}", ckpt_opt[key], curr_opt[key])

    # 6. Global identifiers
    for key in ["seed", "stage", "parent_hash"]:
        if full_strict:
            if key in ckpt_cfg or key in current_cfg:
                _diff_val(f"Global {key}", ckpt_cfg.get(key), current_cfg.get(key))
        else:
            if key in ckpt_cfg and key in current_cfg:
                _diff_val(f"Global {key}", ckpt_cfg[key], current_cfg[key])

    return mismatches


class CheckpointManager:
    """Manages saving, resuming, and exporting checkpoints for training stages."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        filename: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        batch_idx: int = 0,
        extra_meta: Optional[Dict[str, Any]] = None,
        allow_overwrite: bool = False,
    ) -> str:
        """Save full training checkpoint with model, optimizer, scheduler, RNG state, head stats, and config."""
        save_path = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(save_path):
            # If the existing file was a completed formal checkpoint, block overwrite
            try:
                existing = load_checkpoint(save_path, map_location="cpu")
                if isinstance(existing, dict):
                    ext_meta = existing.get("extra_meta", existing.get("metadata", {}))
                    if existing.get("completed") is True or ext_meta.get("completed") is True:
                        if not allow_overwrite:
                            logger.error(
                                f"[Save Blocked] Completed checkpoint already exists at {save_path}. "
                                "Overwriting completed runs is strictly prohibited."
                            )
                            raise FileExistsError(
                                f"[Save Blocked] Completed checkpoint already exists at {save_path}. "
                                "Overwriting completed runs is strictly prohibited."
                            )
            except FileExistsError:
                raise
            except Exception as e:
                logger.warning(f"Could not inspect existing checkpoint {save_path} for completion status: {e}")

            if not allow_overwrite:
                logger.error(f"[Save Blocked] Checkpoint already exists at: {save_path} and allow_overwrite=False.")
                raise FileExistsError(
                    f"[Save Blocked] Checkpoint already exists at: {save_path} and allow_overwrite=False."
                )

        # Handle DataParallel if wrapped
        raw_model = model.module if hasattr(model, "module") else model

        # Extract AdaFace buffer stats if present
        head_stats = {}
        if hasattr(raw_model.head, "batch_mean"):
            head_stats["batch_mean"] = raw_model.head.batch_mean.detach().cpu()
        if hasattr(raw_model.head, "batch_std"):
            head_stats["batch_std"] = raw_model.head.batch_std.detach().cpu()

        meta = extra_meta or {}

        # Extract full training configuration if not provided in extra_meta
        train_config = meta.get("config")
        if train_config is None:
            train_config = extract_model_and_train_config(
                model=raw_model,
                config=meta.get("raw_config"),
                seed=meta.get("seed"),
                stage=meta.get("stage"),
                head_type=meta.get("head", meta.get("head_type")),
                parent_hash=meta.get("parent_hash"),
                scheduler=scheduler,
                optimizer=optimizer,
            )
        config_hash = compute_config_hash(train_config)
        meta["config"] = train_config
        meta["config_hash"] = config_hash

        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "backbone_state_dict": raw_model.backbone.state_dict(),
            "head_state_dict": raw_model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "head_stats": head_stats,
            "rng_state": {
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            "completed": meta.get("completed", None),
            "is_smoke": meta.get("is_smoke", False),
            "seed": meta.get("seed", None),
            "stage": meta.get("stage", None),
            "head": meta.get("head", meta.get("head_type", None)),
            "total_epochs": meta.get("total_epochs", None),
            "parent_hash": meta.get("parent_hash", None),
            "config": train_config,
            "config_hash": config_hash,
            "extra_meta": meta,
        }

        tmp_save_path = save_path + ".tmp"
        try:
            torch.save(state, tmp_save_path)
            os.replace(tmp_save_path, save_path)
            logger.info(f"Saved full checkpoint to {save_path}")
            return save_path
        except Exception as e:
            if os.path.exists(tmp_save_path):
                os.remove(tmp_save_path)
            logger.error(f"Failed to save checkpoint to {save_path}: {e}", exc_info=True)
            raise

    def resume_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        expected_seed: Optional[int] = None,
        expected_head: Optional[str] = None,
        expected_parent_hash: Optional[str] = None,
        expected_stage: Optional[str] = None,
        expected_config: Optional[Dict[str, Any]] = None,
        return_meta: bool = False,
        strict_config: bool = True,
        allow_unverified_resume: bool = False,
    ) -> Any:
        """Resume training state from a checkpoint file with origin & settings validation."""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        raw_model = model.module if hasattr(model, "module") else model

        meta = checkpoint.get("extra_meta", checkpoint.get("metadata", {}))

        # Check smoke status: do not allow resuming a smoke checkpoint for formal training
        is_smoke = (
            checkpoint.get("is_smoke") is True
            or meta.get("is_smoke") is True
        )
        if is_smoke:
            logger.error(f"[Resume Error] Cannot resume from smoke checkpoint: {checkpoint_path}")
            raise ValueError(f"Cannot resume formal training from smoke checkpoint: {checkpoint_path}")

        # Validate stage if requested
        if expected_stage is not None:
            ckpt_stage = meta.get("stage", checkpoint.get("stage", None))
            if ckpt_stage is None:
                logger.error(f"[Resume Error] Resume checkpoint lacks stage metadata (expected '{expected_stage}')")
                raise ValueError(
                    f"[Resume Error] Resume checkpoint lacks required stage metadata (expected '{expected_stage}')."
                )
            if str(ckpt_stage).lower() != str(expected_stage).lower():
                logger.error(f"[Resume Error] Stage mismatch! Expected {expected_stage}, found {ckpt_stage}")
                raise ValueError(
                    f"[Resume Error] Stage mismatch! Expected stage '{expected_stage}', "
                    f"but resume checkpoint specifies stage '{ckpt_stage}'."
                )

        # Validate origin, seed, head, parent_hash against current run configuration
        if expected_seed is not None:
            ckpt_seed = meta.get("seed", checkpoint.get("seed", None))
            if ckpt_seed is None:
                logger.error(f"[Resume Error] Resume checkpoint lacks seed metadata (expected {expected_seed})")
                raise ValueError(
                    f"[Resume Error] Resume checkpoint lacks required seed metadata (expected {expected_seed})."
                )
            if int(ckpt_seed) != int(expected_seed):
                logger.error(f"[Resume Error] Seed mismatch! Expected {expected_seed}, found {ckpt_seed}")
                raise ValueError(
                    f"[Resume Error] Seed mismatch! Expected seed {expected_seed}, "
                    f"but resume checkpoint specifies seed {ckpt_seed}."
                )

        if expected_head is not None:
            ckpt_head = meta.get("head", meta.get("head_type", checkpoint.get("head", None)))
            if ckpt_head is None:
                logger.error(f"[Resume Error] Resume checkpoint lacks head metadata (expected '{expected_head}')")
                raise ValueError(
                    f"[Resume Error] Resume checkpoint lacks required head metadata (expected '{expected_head}')."
                )
            if str(ckpt_head).lower() != str(expected_head).lower():
                logger.error(f"[Resume Error] Head mismatch! Expected {expected_head}, found {ckpt_head}")
                raise ValueError(
                    f"[Resume Error] Head mismatch! Expected head '{expected_head}', "
                    f"but resume checkpoint specifies head '{ckpt_head}'."
                )

        if expected_parent_hash is not None:
            ckpt_phash = meta.get("parent_hash", checkpoint.get("parent_hash", None))
            if ckpt_phash is None:
                logger.error(f"[Resume Error] Resume checkpoint lacks parent_hash metadata (expected '{expected_parent_hash}')")
                raise ValueError(
                    f"[Resume Error] Resume checkpoint lacks required parent_hash metadata (expected '{expected_parent_hash}')."
                )
            if str(ckpt_phash).lower() != str(expected_parent_hash).lower():
                logger.error(f"[Resume Error] Parent hash mismatch! Expected {expected_parent_hash}, found {ckpt_phash}")
                raise ValueError(
                    f"[Resume Error] Parent hash mismatch! Expected parent_hash '{expected_parent_hash}', "
                    f"but resume checkpoint specifies '{ckpt_phash}'."
                )

        # Ensure optimizer_state_dict exists if resuming optimizer
        if optimizer is not None:
            if "optimizer_state_dict" not in checkpoint or checkpoint["optimizer_state_dict"] is None:
                logger.error(f"[Resume Error] Checkpoint does not contain optimizer_state_dict: {checkpoint_path}")
                raise ValueError(
                    f"[Resume Error] Checkpoint '{checkpoint_path}' does not contain optimizer_state_dict. "
                    "Cannot resume formal training without optimizer state."
                )

        # Validate configuration metadata presence (Fail-Closed on missing config/hash)
        ckpt_cfg = checkpoint.get("config", meta.get("config", None))
        ckpt_hash = checkpoint.get("config_hash", meta.get("config_hash", ckpt_cfg.get("config_hash", None) if isinstance(ckpt_cfg, dict) else None))

        is_strict = bool(strict_config and not allow_unverified_resume)

        if is_strict:
            if ckpt_cfg is None or not isinstance(ckpt_cfg, dict) or not ckpt_cfg or "backbone" not in ckpt_cfg or "head" not in ckpt_cfg:
                logger.error(
                    f"[Resume Config Error] Checkpoint '{checkpoint_path}' lacks required configuration metadata ('config'). "
                    "Formal resume strictly requires verifiable configuration metadata. "
                    "To resume unverified legacy checkpoints, specify allow_unverified_resume=True or strict_config=False."
                )
                raise ValueError(
                    f"[Resume Config Error] Checkpoint '{checkpoint_path}' lacks required configuration metadata ('config'). "
                    "Formal resume strictly requires verifiable configuration metadata. "
                    "To resume unverified legacy checkpoints, specify allow_unverified_resume=True or strict_config=False."
                )
            if ckpt_hash is None or not str(ckpt_hash).strip():
                logger.error(
                    f"[Resume Config Error] Checkpoint '{checkpoint_path}' lacks required config hash metadata ('config_hash'). "
                    "Formal resume strictly requires verifiable configuration metadata. "
                    "To resume unverified legacy checkpoints, specify allow_unverified_resume=True or strict_config=False."
                )
                raise ValueError(
                    f"[Resume Config Error] Checkpoint '{checkpoint_path}' lacks required config hash metadata ('config_hash'). "
                    "Formal resume strictly requires verifiable configuration metadata. "
                    "To resume unverified legacy checkpoints, specify allow_unverified_resume=True or strict_config=False."
                )
            # Verify internal integrity: config_hash must match SHA-256 of checkpoint's config
            computed_ckpt_hash = compute_config_hash(ckpt_cfg)
            if ckpt_hash != computed_ckpt_hash:
                logger.error(
                    f"[Resume Config Error] Checkpoint configuration internal hash mismatch! "
                    f"Recorded config_hash='{ckpt_hash}', but computed hash='{computed_ckpt_hash}'."
                )
                raise ValueError(
                    f"[Resume Config Error] Checkpoint configuration internal hash mismatch! "
                    f"Checkpoint metadata appears tampered or corrupted. "
                    f"Recorded config_hash='{ckpt_hash}', but computed hash='{computed_ckpt_hash}'."
                )
        else:
            if ckpt_cfg is None or ckpt_hash is None:
                logger.warning(
                    f"[Resume Warning] Resuming from unverified checkpoint '{checkpoint_path}' without configuration metadata "
                    f"(allow_unverified_resume={allow_unverified_resume}, strict_config={strict_config}). "
                    "Training conditions (margins, dropout, optimizer, scheduler, dataset) cannot be verified against current run."
                )

        # Validate training configuration (backbone, head, dropout, margins, scale, preprocessing, dataset, optimizer)
        current_model_cfg = extract_model_config(raw_model)

        if expected_config is not None:
            effective_curr_cfg = copy.deepcopy(expected_config)
            if "backbone" in current_model_cfg:
                effective_curr_cfg.setdefault("backbone", {}).update(current_model_cfg["backbone"])
            if "head" in current_model_cfg:
                effective_curr_cfg.setdefault("head", {}).update(current_model_cfg["head"])
            full_strict = True
        else:
            effective_curr_cfg = copy.deepcopy(current_model_cfg)
            live_train_cfg = extract_train_config(
                config=None,
                scheduler=scheduler,
                optimizer=optimizer,
                seed=expected_seed,
                stage=expected_stage,
                parent_hash=expected_parent_hash,
            )
            for sect_k, sect_v in live_train_cfg.items():
                if sect_k not in effective_curr_cfg:
                    effective_curr_cfg[sect_k] = sect_v
            full_strict = False

        if ckpt_cfg is not None and isinstance(ckpt_cfg, dict):
            mismatches = compare_configs(ckpt_cfg, effective_curr_cfg, full_strict=full_strict)
            if mismatches:
                error_msg = (
                    f"[Resume Config Error] Training configuration mismatch between checkpoint and current run:\n"
                    + "\n".join(f"  - {m}" for m in mismatches)
                    + "\nResuming with altered training conditions is strictly prohibited. Start a new run instead."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if ckpt_hash and expected_config is not None:
                expected_hash = effective_curr_cfg.get("config_hash") or compute_config_hash(effective_curr_cfg)
                if ckpt_hash != expected_hash:
                    logger.error(f"[Resume Config Error] Config hash mismatch! Checkpoint: {ckpt_hash}, Current: {expected_hash}")
                    raise ValueError(
                        f"[Resume Config Error] Config hash mismatch! Checkpoint has hash '{ckpt_hash}', "
                        f"but current configuration produces hash '{expected_hash}'."
                    )

        try:
            # Load backbone & head
            if "backbone_state_dict" in checkpoint:
                raw_model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
                logger.info("Loaded backbone state_dict.")
            if "head_state_dict" in checkpoint:
                raw_model.head.load_state_dict(checkpoint["head_state_dict"])
                logger.info("Loaded head state_dict.")

            # Restore AdaFace EMA buffers if present
            if "head_stats" in checkpoint and checkpoint["head_stats"]:
                if hasattr(raw_model.head, "batch_mean") and "batch_mean" in checkpoint["head_stats"]:
                    raw_model.head.batch_mean.copy_(checkpoint["head_stats"]["batch_mean"])
                if hasattr(raw_model.head, "batch_std") and "batch_std" in checkpoint["head_stats"]:
                    raw_model.head.batch_std.copy_(checkpoint["head_stats"]["batch_std"])
                logger.info("Restored head EMA buffers (batch_mean, batch_std).")

            # Load optimizer & scheduler
            if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state_dict.")
            if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Loaded scheduler state_dict.")
        except Exception as e:
            logger.error(f"Failed loading model/optimizer state during resume from {checkpoint_path}: {e}", exc_info=True)
            raise

        # Restore RNG states for exact reproducibility
        if "rng_state" in checkpoint and checkpoint["rng_state"]:
            rng = checkpoint["rng_state"]
            if "torch_rng" in rng and rng["torch_rng"] is not None:
                torch.set_rng_state(rng["torch_rng"])
            if "cuda_rng" in rng and rng["cuda_rng"] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng["cuda_rng"])
                except Exception as e:
                    logger.warning(f"Could not restore CUDA RNG states: {e}")
            if "numpy_rng" in rng and rng["numpy_rng"] is not None:
                np.random.set_state(rng["numpy_rng"])
            if "python_rng" in rng and rng["python_rng"] is not None:
                random.setstate(rng["python_rng"])
            logger.info("Restored RNG states.")

        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Successfully resumed from {checkpoint_path}. Next epoch: {start_epoch}")
        if return_meta:
            return start_epoch, meta
        return start_epoch

    def export_backbone_only(
        self,
        model: Any,
        output_filename: str,
        metadata: Dict[str, Any],
        target_feat_dim: int = 1024,
        allow_overwrite: bool = False,
    ) -> str:
        """Export clean inference backbone weights (for ronbun and evaluation benchmarks).
        
        Guarantees:
        1. Rejects any checkpoint declared or marked as is_smoke=True.
        2. Blocks overwriting completed exports (unless allow_overwrite=True).
        3. Filters out all classifier/head keys (e.g. head.*, classifier.*).
        4. Verifies strict loadability into a standalone MobileFaceNet before publishing.
        5. Atomically publishes using os.replace without prior destructive deletion.
        """
        if metadata.get("is_smoke") is True:
            logger.error("[Export Violation] Cannot export smoke-test checkpoint as formal release!")
            raise ValueError("Cannot export smoke-test checkpoint as formal release!")

        export_pt = os.path.join(self.checkpoint_dir, output_filename)
        if os.path.exists(export_pt) and not allow_overwrite:
            logger.error(
                f"[Export Blocked] Exported model already exists at: {export_pt}. "
                "Overwriting completed exports is strictly forbidden to preserve experimental immutability."
            )
            raise FileExistsError(
                f"[Export Blocked] Exported model already exists at: {export_pt}. "
                "Overwriting completed exports is strictly forbidden to preserve experimental immutability. "
                "Specify a unique run export directory or set allow_overwrite=True if intentionally replacing."
            )

        # Extract weights from module or dict
        if isinstance(model, dict):
            if "backbone_state_dict" in model:
                raw_weights = model["backbone_state_dict"]
            elif "state_dict" in model:
                raw_weights = model["state_dict"]
            else:
                raw_weights = model
        else:
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "backbone"):
                raw_weights = raw_model.backbone.state_dict()
            else:
                raw_weights = raw_model.state_dict()

        # Clean prefixes and filter out classifier/head layers
        clean_weights = {}
        for k, v in raw_weights.items():
            clean_k = k.replace("module.", "").replace("backbone.", "")
            if clean_k.startswith("head.") or clean_k.startswith("classifier."):
                continue
            clean_weights[clean_k] = v.cpu()

        tmp_export_pt = export_pt + ".tmp"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        try:
            # 1. Save to temporary file first
            torch.save({"state_dict": clean_weights, "metadata": metadata}, tmp_export_pt)

            # 2. Strict loading verification with standalone MobileFaceNet
            try:
                from backbone.MobileFaceNets import MobileFaceNet
            except ImportError:
                from ..backbone.MobileFaceNets import MobileFaceNet

            validator_model = MobileFaceNet(
                embedding_size=target_feat_dim,
                out_h=7,
                out_w=7,
                dropout_rate=0.0,
            )
            # Must strictly match MobileFaceNet keys (no missing, no unexpected)
            validator_model.load_state_dict(clean_weights, strict=True)
            validator_model.eval()

            # Dummy forward pass verification
            with torch.no_grad():
                dummy_out = validator_model(torch.randn(2, 3, 112, 112))
                assert dummy_out.shape == (2, target_feat_dim), f"Shape mismatch: {dummy_out.shape}"

            # Compute SHA-256 for the verified exported checkpoint before moving
            export_hash = compute_sha256(tmp_export_pt)
            metadata["sha256"] = export_hash
            metadata["exported_file"] = output_filename

            # 3. Validation passed: Write metadata JSON to temporary file
            meta_json_path = os.path.splitext(export_pt)[0] + "_meta.json"
            tmp_meta_json = meta_json_path + ".tmp"
            with open(tmp_meta_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 4. Atomically publish both export weights and metadata JSON
            os.replace(tmp_export_pt, export_pt)
            os.replace(tmp_meta_json, meta_json_path)

            logger.info(f"Verified & exported clean backbone weights to {export_pt} (SHA-256: {export_hash})")
            logger.info(f"Exported metadata to {meta_json_path}")
            return export_pt
        except Exception as e:
            if os.path.exists(tmp_export_pt):
                os.remove(tmp_export_pt)
            meta_json_path = os.path.splitext(export_pt)[0] + "_meta.json"
            tmp_meta_json = meta_json_path + ".tmp"
            if os.path.exists(tmp_meta_json):
                os.remove(tmp_meta_json)
            logger.error(f"Failed to export backbone weights: {e}", exc_info=True)
            raise


def validate_checkpoint_for_export(
    checkpoint_data: Dict[str, Any],
    target_epoch: int = 40,
    allow_smoke: bool = False,
    allow_partial: bool = False,
    expected_seed: Optional[int] = None,
    expected_head: Optional[str] = None,
    expected_stage: Optional[str] = None,
    expected_parent_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate that a fine-tuned checkpoint is a legitimate, completed formal run before export.
    
    Checks:
    1. Rejects if is_smoke is True (unless allow_smoke=True).
    2. Validates backbone presence.
    3. Validates completed status and target epoch completeness (unless allow_partial=True).
    4. Validates seed match if expected_seed is provided.
    5. Validates head match if expected_head is provided.
    6. Validates stage match if expected_stage is provided.
    7. Validates parent hash match if expected_parent_hash is provided.
    8. Returns parsed metadata dictionary.
    """
    if not allow_smoke:
        is_smoke = (
            checkpoint_data.get("is_smoke") is True
            or checkpoint_data.get("extra_meta", {}).get("is_smoke") is True
            or checkpoint_data.get("metadata", {}).get("is_smoke") is True
        )
        if is_smoke:
            raise ValueError(
                "[Export Blocked] Checkpoint is marked as a smoke test! "
                "Smoke checkpoints cannot be exported as formal releases."
            )

    has_backbone = "backbone_state_dict" in checkpoint_data or "state_dict" in checkpoint_data
    if not has_backbone:
        raise ValueError("[Export Blocked] Checkpoint contains no backbone weights.")

    meta = checkpoint_data.get("extra_meta", checkpoint_data.get("metadata", {}))
    actual_epoch = checkpoint_data.get("epoch", meta.get("epoch", 0))
    if actual_epoch < 1:
        raise ValueError(f"[Export Blocked] Checkpoint has invalid completed epoch: {actual_epoch}")

    is_completed = checkpoint_data.get("completed", meta.get("completed", None))
    if not allow_partial:
        if is_completed is not True:
            raise ValueError(
                f"[Export Blocked] Checkpoint is not marked completed (completed={is_completed}). "
                "Formal export strictly requires a completed fine-tuning run. "
                "Set allow_partial=True if analyzing an intermediate checkpoint."
            )
        if actual_epoch < target_epoch:
            raise ValueError(
                f"[Export Blocked] Checkpoint reached epoch {actual_epoch}, which is less than "
                f"target_epoch {target_epoch}. Formal export requires scheduled completion. "
                "Set allow_partial=True if analyzing an intermediate checkpoint."
            )

        # Validate seed if expected_seed is provided
        if expected_seed is not None:
            ckpt_seed = meta.get("seed", checkpoint_data.get("seed", None))
            if ckpt_seed is None:
                raise ValueError(
                    f"[Export Blocked] Checkpoint lacks required seed metadata (expected {expected_seed})."
                )
            if int(ckpt_seed) != int(expected_seed):
                raise ValueError(
                    f"[Export Blocked] Seed mismatch! Expected seed {expected_seed}, "
                    f"but checkpoint specifies seed {ckpt_seed}."
                )

        # Validate head if expected_head is provided
        if expected_head is not None:
            ckpt_head = meta.get("head", meta.get("head_type", checkpoint_data.get("head", None)))
            if ckpt_head is None:
                raise ValueError(
                    f"[Export Blocked] Checkpoint lacks required head metadata (expected '{expected_head}')."
                )
            if str(ckpt_head).lower() != str(expected_head).lower():
                raise ValueError(
                    f"[Export Blocked] Head mismatch! Expected head '{expected_head}', "
                    f"but checkpoint specifies head '{ckpt_head}'."
                )

        # Validate stage if expected_stage is provided
        if expected_stage is not None:
            ckpt_stage = meta.get("stage", checkpoint_data.get("stage", None))
            if ckpt_stage is None:
                raise ValueError(
                    f"[Export Blocked] Checkpoint lacks required stage metadata (expected '{expected_stage}')."
                )
            if str(ckpt_stage).lower() != str(expected_stage).lower():
                raise ValueError(
                    f"[Export Blocked] Stage mismatch! Expected stage '{expected_stage}', "
                    f"but checkpoint specifies stage '{ckpt_stage}'."
                )

        # Validate parent_hash if expected_parent_hash is provided
        if expected_parent_hash is not None:
            ckpt_phash = meta.get("parent_hash", checkpoint_data.get("parent_hash", None))
            if ckpt_phash is None:
                raise ValueError(
                    f"[Export Blocked] Checkpoint lacks required parent_hash metadata (expected '{expected_parent_hash}')."
                )
            if str(ckpt_phash).lower() != str(expected_parent_hash).lower():
                raise ValueError(
                    f"[Export Blocked] Parent hash mismatch! Expected parent_hash '{expected_parent_hash}', "
                    f"but checkpoint specifies '{ckpt_phash}'."
                )

    return meta


__all__ = [
    "compute_sha256",
    "compute_config_hash",
    "extract_model_config",
    "extract_train_config",
    "extract_model_and_train_config",
    "compare_configs",
    "load_checkpoint",
    "verify_parent_checkpoint",
    "validate_checkpoint_for_export",
    "CheckpointManager",
]

