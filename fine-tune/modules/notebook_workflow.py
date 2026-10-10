"""Notebook-first configuration and sequential, restartable baseline campaigns.

YAML defaults < saved notebook settings < the current notebook's overrides.
No training is triggered by resolving settings or building a plan.
"""
from __future__ import annotations

import copy
import gc
import json
import logging
import os
import platform
import shutil
import subprocess
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .checkpoint_manager import (
    CheckpointManager, compute_sha256, compute_config_hash, load_checkpoint,
    verify_parent_checkpoint, validate_checkpoint_for_export,
)
from .dataset import VGGFace2ArchiveInspector, build_dataloader, validate_dataset_files, resolve_image_path
from .model_builder import build_model
from .trainer import Trainer, set_seed

HEAD_NAMES = ("arcface", "adaface", "magface")


def merge_settings(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_settings(base_dir, settings_path=None, overrides=None, require_saved=False):
    root = Path(base_dir)
    cfg = yaml.safe_load((root / "fine-tune/configs/common.yaml").read_text(encoding="utf-8"))
    cfg["heads"] = {
        head: yaml.safe_load((root / f"fine-tune/configs/{head}.yaml").read_text(encoding="utf-8"))["head"]
        for head in HEAD_NAMES
    }
    cfg["campaign"] = {"heads": list(HEAD_NAMES), "reuse_completed": True}
    cfg["paths"]["report_dir"] = str(root / "fine-tune/reports/campaigns")
    cfg["training"]["num_workers"] = 0
    if settings_path and Path(settings_path).is_file():
        cfg = merge_settings(cfg, json.loads(Path(settings_path).read_text(encoding="utf-8")))
    elif require_saved:
        raise FileNotFoundError(f"Run notebook 00 to save settings first: {settings_path}")
    cfg = merge_settings(cfg, overrides or {})
    validate_settings(cfg)
    return cfg


def validate_settings(cfg):
    seeds = cfg["project"]["seed_list"]
    heads = cfg["campaign"]["heads"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("SEEDS must contain unique integers in [0, 2**32)")
    if not heads or len(heads) != len(set(heads)) or any(h not in HEAD_NAMES for h in heads):
        raise ValueError(f"HEADS must contain unique choices from {HEAD_NAMES}")
    # Notebook execution deliberately uses single-process loading on every OS.
    if int(cfg["training"]["num_workers"]) != 0:
        raise ValueError("Notebook workflow requires NUM_WORKERS=0 (Windows/Jupyter safe mode).")
    if list(cfg["dataset"]["input_size"]) != [112, 112]:
        raise ValueError("Current MobileFaceNet/export architecture requires INPUT_SIZE=(112,112).")
    if cfg["backbone"]["type"] != "MobileFaceNet" or (cfg["backbone"]["out_h"], cfg["backbone"]["out_w"]) != (7, 7):
        raise ValueError("This notebook workflow supports MobileFaceNet with out_h=out_w=7.")
    if cfg["training"]["batch_size"] < 2:
        raise ValueError("BATCH_SIZE must be >= 2 for training BatchNorm.")
    if cfg["training"]["epoches"] < 1 or cfg["training"]["save_freq"] < 1:
        raise ValueError("EPOCHS and SAVE_FREQ must be positive")
    if not np.isfinite(cfg["dataset"]["pixel_std"]) or cfg["dataset"]["pixel_std"] <= 0:
        raise ValueError("PIXEL_STD must be positive and finite")
    if not np.isfinite(cfg["dataset"]["pixel_mean"]):
        raise ValueError("PIXEL_MEAN must be finite")


def write_json(path, payload, immutable=False):
    path = Path(path)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing record differs; select a new output path: {path}")
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)
    return str(path)


def save_settings(path, cfg):
    validate_settings(cfg)
    return write_json(path, cfg)


def inspect_environment(base_dir, cfg):
    """Report only observed checks; existence is never an archive integrity claim."""
    ds = cfg["dataset"]
    checks = {}
    checks["python"] = platform.python_version()
    checks["torch"] = torch.__version__
    checks["cuda_available"] = torch.cuda.is_available()
    checks["gpu"] = torch.cuda.get_device_name(0) if checks["cuda_available"] else None
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=base_dir, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        branch = None
    checks["branch"] = branch
    checks["branch_matches"] = branch == "add-adaface"
    for key in ("train_archive", "test_archive", "meta_csv"):
        p = Path(ds[key])
        checks[key] = {"exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else None}
    root = Path(ds["raw_root"])
    checks["free_bytes"] = shutil.disk_usage(root).free if root.exists() else None
    try:
        ids = VGGFace2ArchiveInspector(str(root)).parse_identity_meta(train_only=True)
        checks["training_identities"] = len(ids)
        checks["identity_count_matches"] = len(ids) == ds["num_classes"]
    except (OSError, ValueError, StopIteration) as exc:
        checks["identity_count_matches"] = False
        checks["metadata_error"] = str(exc)
    checks["archive_integrity"] = "not_checked"
    checks["all_images_decode"] = "not_checked"
    checks["readiness"] = "partial_environment_check_only"
    return checks


def prepare_indices(cfg, val_every=20):
    """Preserve the existing deterministic 95:5 split; never replace differing lists."""
    if val_every < 2:
        raise ValueError("VAL_EVERY must be >= 2")
    ds = cfg["dataset"]
    ids = VGGFace2ArchiveInspector(ds["raw_root"]).parse_identity_meta(train_only=True)
    if len(ids) != ds["num_classes"]:
        raise ValueError("Metadata class count differs from NUM_CLASSES")
    labels = {x["class_id"]: i for i, x in enumerate(ids)}
    root = Path(ds["raw_root"])
    candidates = [root / "meta/train_list.txt", root / "data/train_list.txt"]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError("Original VGGFace2 train_list.txt is missing")
    paths = [Path(ds["train_list"]), Path(ds["val_list"])]
    if paths[0].resolve() == paths[1].resolve() or source.resolve() in [p.resolve() for p in paths]:
        raise ValueError("Input list and two output lists must have distinct paths")
    temps, counts = [], [0, 0]
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            temps.append(path.with_name(path.name + f".{uuid.uuid4().hex}.tmp"))
        with source.open(encoding="utf-8") as src, temps[0].open("w", encoding="utf-8") as train, temps[1].open("w", encoding="utf-8") as val:
            for i, line in enumerate(src):
                rel = line.strip().replace("\\", "/")
                if not rel:
                    continue
                identity = rel.removeprefix("train/").split("/")[0]
                if identity not in labels:
                    raise ValueError(f"Unknown training identity: {identity}")
                index = int(i % val_every == 0)
                (train, val)[index].write(f"{rel} {labels[identity]}\n")
                counts[index] += 1
        if min(counts) < 1:
            raise ValueError("Both train and validation lists must be nonempty")
        for path, temp in zip(paths, temps):
            if path.exists() and compute_sha256(str(path)) != compute_sha256(str(temp)):
                raise FileExistsError(f"Refusing to change an existing split: {path}. Use new list paths.")
        for path, temp in zip(paths, temps):
            if not path.exists():
                os.replace(temp, path)
        return {"train_count": counts[0], "val_count": counts[1],
                "train_sha256": compute_sha256(str(paths[0])), "val_sha256": compute_sha256(str(paths[1])),
                "split": f"every {val_every}th source row is validation; not identity-disjoint"}
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)


def make_loaders(cfg):
    ds, train = cfg["dataset"], cfg["training"]
    loaders = []
    for key, is_train in (("train_list", True), ("val_list", False)):
        index = ds.get(key)
        if not index or not Path(index).is_file() or Path(index).stat().st_size == 0:
            raise FileNotFoundError(f"Required nonempty {key}: {index}")
        _, missing = validate_dataset_files(ds["raw_root"], index, max_check=20)
        if missing:
            raise FileNotFoundError(f"Missing image: {missing[0]}")
        loader = build_dataloader(
            data_root=ds["raw_root"], index_file=index,
            batch_size=train["batch_size"], num_workers=train["num_workers"], is_train=is_train,
            target_size=tuple(ds["input_size"]), pixel_mean=ds["pixel_mean"], pixel_std=ds["pixel_std"],
        )
        if len(loader) == 0:
            raise ValueError(f"No usable batches in {key}; training drops incomplete batches")
        if any(label < 0 or label >= ds["num_classes"] for _, label in loader.dataset.samples):
            raise ValueError(f"Label outside NUM_CLASSES in {key}")
        loaders.append(loader)
    return tuple(loaders)


def plan_matrix(cfg, stage):
    validate_settings(cfg)
    if stage not in ("pretrain", "finetune", "export"):
        raise ValueError(stage)
    heads = ["arcface"] if stage == "pretrain" else cfg["campaign"]["heads"]
    return [{"stage": stage, "head": h, "seed": seed,
             "run_id": f"{stage}_{h}_seed{seed}"}
            for seed in cfg["project"]["seed_list"] for h in heads]


def checkpoint_path(cfg, stage, head, seed):
    run_dir = Path(cfg["paths"]["checkpoint_dir"]) / f"{stage}_{head}_seed{seed}"
    dim = cfg["backbone"]["feat_dim"]
    filename = (f"pretrain_mobilefacenet_{dim}d.pt" if stage == "pretrain" else
                f"finetune_mobilefacenet_{dim}d_{head}_epoch_{cfg['training']['epoches']}.pt")
    return run_dir / filename


def _formal_device(cfg):
    device = torch.device(cfg["project"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal training requires an available CUDA device; no CPU fallback")
    torch.cuda.get_device_name(device)
    return device


def _run_one(cfg, row, resume_path=None, smoke=False):
    cfg = copy.deepcopy(cfg)
    cfg["project"]["seed"] = row["seed"]
    set_seed(row["seed"])
    stage, head, seed = row["stage"], row["head"], row["seed"]
    final_path = checkpoint_path(cfg, stage, head, seed)
    run_dir = final_path.parent if not smoke else Path(cfg["paths"]["checkpoint_dir"]) / "smoke" / row["run_id"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if smoke else _formal_device(cfg)
    parent_hash = None
    parent = None
    model = trainer = optimizer = None
    try:
        if stage == "finetune" and not smoke:
            p = checkpoint_path(cfg, "pretrain", "arcface", seed)
            hp = p.with_name(p.stem + "_hash.json")
            expected_hash = json.loads(hp.read_text(encoding="utf-8"))["sha256"]
            parent, parent_hash = verify_parent_checkpoint(
                str(p), expected_hash=expected_hash, require_hash=True, require_completed=True,
                expected_seed=seed, target_epoch=cfg["training"]["epoches"],
            )
        head_cfg = dict(cfg["heads"][head], num_class=cfg["dataset"]["num_classes"])
        model = build_model(cfg["backbone"], head_cfg).to(device)
        if parent is not None:
            model.backbone.load_state_dict(parent["backbone_state_dict"], strict=True)
        manager = CheckpointManager(str(run_dir))
        if smoke:
            model.train()
            x = torch.randn(4, 3, *cfg["dataset"]["input_size"], device=device)
            y = torch.arange(4, device=device) % cfg["dataset"]["num_classes"]
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
            logits, aux = model(x, y)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if aux is not None:
                loss = loss + aux.mean()
            loss.backward()
            optimizer.step()
            # A smoke run produces diagnostics only; it cannot become a formal parent.
            return {**row, "status": "smoke", "completed": False, "is_smoke": True,
                    "loss": float(loss.detach()), "device": str(device)}
        train_loader, val_loader = make_loaders(cfg)
        trainer = Trainer(model, train_loader, val_loader, cfg, manager, device,
                          parent_hash=parent_hash, head_type=head, stage=stage)
        if final_path.is_file():
            if not cfg["campaign"]["reuse_completed"]:
                raise FileExistsError(str(final_path))
            payload = load_checkpoint(str(final_path))
            validate_checkpoint_for_export(payload, target_epoch=trainer.total_epochs,
                                           expected_seed=seed, expected_head=head, expected_stage=stage,
                                           expected_parent_hash=parent_hash)
            trainer.resume_from_checkpoint(str(final_path))  # verifies effective data/config too
            status = "reused_completed"
            actual_retention = payload.get("extra_meta", {}).get("execution_settings", {}).get("project", {}).get("keep_raw_results")
        else:
            existing = list(run_dir.glob("*.pt"))
            if existing and resume_path is None:
                raise FileExistsError(f"Incomplete run exists. Set RESUME_CHECKPOINTS['{row['run_id']}'] explicitly.")
            start = trainer.resume_from_checkpoint(str(resume_path)) if resume_path else 1
            snapshot = run_dir / "notebook_settings.json"
            if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) != cfg:
                # Scientific equality was verified by resume_from_checkpoint above.
                # Preserve the original selection/retention settings and record this attempt separately.
                snapshot = run_dir / f"resume_settings_{uuid.uuid4().hex}.json"
            write_json(snapshot, cfg, immutable=True)
            prefix = f"{stage}_mobilefacenet_{cfg['backbone']['feat_dim']}d" + (f"_{head}" if stage == "finetune" else "")
            trainer.train(start_epoch=start, model_prefix=prefix)
            trainer.save_state_checkpoint(final_path.name, epoch=trainer.cur_epoch,
                                          extra_meta={"execution_settings": cfg})
            status = "completed"
            actual_retention = trainer.keep_raw_results
        if stage == "pretrain":
            hp = final_path.with_name(final_path.stem + "_hash.json")
            checksum = compute_sha256(str(final_path))
            if hp.exists():
                if json.loads(hp.read_text(encoding="utf-8")).get("sha256") != checksum:
                    raise ValueError(f"Parent hash record does not match: {hp}")
            else:
                write_json(hp, {"sha256": checksum, "seed": seed, "completed": True}, immutable=True)
        return {**row, "status": status, "completed": True, "checkpoint": str(final_path),
                "sha256": compute_sha256(str(final_path)), "config_hash": trainer.config_hash,
                "parent_hash": parent_hash, "keep_raw_results": actual_retention,
                "requested_keep_raw_results": trainer.keep_raw_results,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    finally:
        del trainer, optimizer, model, parent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _matrix(cfg, stage, operation):
    plan = plan_matrix(cfg, stage)
    rows = []
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    report = Path(cfg["paths"]["report_dir"]) / f"{stage}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    try:
        for row in plan:
            print(f"[{len(rows)+1}/{len(plan)}] {row['run_id']}", flush=True)
            try:
                result = operation(row)
            except Exception as exc:
                rows.append({**row, "status": "failed", "completed": False, "error": str(exc)})
                raise
            rows.append(result)
        return rows
    finally:
        done = sum(r.get("completed") is True for r in rows)
        write_json(report, {"planned": len(plan), "completed": done, "rows": rows,
                            "status": "completed" if done == len(plan) else "partial_or_smoke",
                            "not_started": plan[len(rows):], "effective_settings": cfg,
                            "interpretation": "fine-tune/README.md; smoke is not formal training or FIQA evaluation"})
        print(f"Campaign summary: {report} (formal completed {done}/{len(plan)})")


def run_training_matrix(cfg, stage, resume_checkpoints=None, smoke=False):
    if stage not in ("pretrain", "finetune"):
        raise ValueError(stage)
    resume = resume_checkpoints or {}
    valid = {r["run_id"] for r in plan_matrix(cfg, stage)}
    if set(resume) - valid:
        raise ValueError(f"Unknown resume run IDs: {set(resume) - valid}")
    if smoke and resume:
        raise ValueError("Smoke mode cannot resume formal checkpoints")
    return _matrix(cfg, stage, lambda row: _run_one(cfg, row, resume.get(row["run_id"]), smoke))


def _export_one(cfg, row):
    head, seed = row["head"], row["seed"]
    path = checkpoint_path(cfg, "finetune", head, seed)
    payload = load_checkpoint(str(path))
    validate_checkpoint_for_export(payload, target_epoch=cfg["training"]["epoches"],
                                   expected_seed=seed, expected_head=head, expected_stage="finetune")
    stored = payload["config"]
    if compute_config_hash(stored) != payload["config_hash"]:
        raise ValueError("Source checkpoint config hash mismatch")
    source_hash = compute_sha256(str(path))
    dim = stored["backbone"]["feat_dim"]
    out_dir = Path(cfg["paths"]["export_dir"]) / row["run_id"]
    filename = f"exported_mobilefacenet_{dim}d_{head}_seed{seed}.pt"
    out = out_dir / filename
    metadata_path = out.with_name(out.stem + "_meta.json")
    if out.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not cfg["campaign"]["reuse_completed"] or meta.get("source_sha256") != source_hash or meta.get("sha256") != compute_sha256(str(out)):
            raise FileExistsError(f"Existing export cannot be verified for this source: {out}")
        status = "reused_completed"
    else:
        metadata = {**row, "completed": True, "is_smoke": False, "feat_dim": dim,
                    "source_checkpoint": str(path), "source_sha256": source_hash,
                    "source_config": stored, "source_config_hash": payload["config_hash"],
                    "execution_settings": payload.get("extra_meta", {}).get("execution_settings"),
                    "preprocessing": {**stored.get("preprocessing", {}), "color": "BGR", "alignment": "resize_only"},
                    "parent_hash": payload.get("parent_hash"), "epochs": payload["epoch"],
                    "ronbun_integration": "not_executed", "fiqa_evaluation": "not_executed"}
        CheckpointManager(str(out_dir)).export_backbone_only(payload, filename, metadata,
                                                           target_feat_dim=dim, allow_overwrite=False)
        status = "completed"
    return {**row, "status": status, "completed": True, "export": str(out),
            "sha256": compute_sha256(str(out)), "source_sha256": source_hash}


def export_matrix(cfg):
    return _matrix(cfg, "export", lambda row: _export_one(cfg, row))
