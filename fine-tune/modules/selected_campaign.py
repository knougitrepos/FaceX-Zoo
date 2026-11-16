"""Explicit, extensible fine-tuning selections. Completed artifacts stay immutable."""
import copy
import csv
import gc
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from . import notebook_workflow as w
from .checkpoint_manager import (CheckpointManager, compare_configs, compute_config_hash,
    compute_sha256, extract_model_and_train_config, load_checkpoint,
    validate_checkpoint_for_export, verify_parent_checkpoint)
from .shared_storage import artifact_lock
from .ucface_reference import selection_scope
from .ucface_controls import ALIGNMENT_CONTROL, PARENT, WEIGHTS, check_file, read_json, export_alignment_control

SCHEMA = "facex-selection-v2"
INVENTORY_SCHEMA = "facex-model-inventory-v2"
RECIPES = {"resize_only": {"scface", "tinyface", "survface"},
           "two_eye_alignment": {"scface"}, "pfld_five_point_alignment": {"tinyface"},
           "arcface_continuous_margin": {"survface"}}
HEADS = {"arcface", "cosface", "magface", "adaface"}


def default_plan():
    return [dict(dataset="scface", recipe="two_eye_alignment", heads=sorted(HEADS), seeds=[8972]),
            dict(dataset="tinyface", recipe="pfld_five_point_alignment", heads=sorted(HEADS), seeds=[8972]),
            dict(dataset="survface", recipe="resize_only", heads=sorted(HEADS), seeds=[8972])]


def expand_plan(plan):
    rows, seen = [], set()
    for group in plan:
        if set(group) != {"dataset", "recipe", "heads", "seeds"}:
            raise ValueError("Each selection requires dataset, recipe, heads and seeds only")
        ds, recipe = group["dataset"], group["recipe"]
        if recipe not in RECIPES or ds not in RECIPES[recipe]:
            raise ValueError(f"Unsupported dataset/recipe: {ds}/{recipe}")
        if not group["heads"] or not group["seeds"]:
            raise ValueError("Empty heads or seeds")
        for head in group["heads"]:
            if head not in HEADS or (recipe == "arcface_continuous_margin" and head != "arcface"):
                raise ValueError(f"Unsupported head for {recipe}: {head}")
            for seed in group["seeds"]:
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError("Seeds must be integers in [0, 2**32)")
                alias = f"{ds}_{head}_{recipe}_s{seed}"
                if alias in seen:
                    raise ValueError(f"Duplicate selection: {alias}")
                seen.add(alias)
                rows.append(dict(run_id=alias, dataset=ds, head=head, recipe=recipe, seed=seed))
    if not rows:
        raise ValueError("Select at least one model")
    return rows


def make_selection(baseline, plan, root, diagnostic_root, references=None, reuse_completed=True,
                   *, purpose="local_protocol_baseline"):
    w.validate_settings(baseline)
    rows = expand_plan(plan)
    references = references or {}
    if set(references) - {r["run_id"] for r in rows}:
        raise ValueError("Reference keys must belong to the selected plan")
    settings = copy.deepcopy(baseline)
    settings.setdefault("data_integrity", {})["force_reverify"] = False  # per-attempt switch, not a new scientific plan
    return dict(schema=SCHEMA, settings=settings, rows=rows, purpose=purpose,
                reproduction_scope=selection_scope(settings, rows, purpose),
                root=str(Path(root).resolve()), diagnostic_root=str(Path(diagnostic_root).resolve()),
                references=copy.deepcopy(references), reuse_completed=bool(reuse_completed))


def save_selection(path, selection):
    """04 reads this exact snapshot, never a newest-file heuristic."""
    path = Path(path)
    with artifact_lock(path):
        w.write_json(path, selection, immutable=True)
    return path


def load_selection(path):
    selection = read_json(path)
    return validate_selection(selection)


def validate_selection(selection):
    if selection.get("schema") not in (SCHEMA, "facex-selection-v1"):
        raise ValueError("Unsupported selection schema")
    legacy = selection["schema"] == "facex-selection-v1"
    rebuilt = make_selection(selection["settings"], [dict(dataset=r["dataset"], recipe=r["recipe"],
        heads=[r["head"]], seeds=[r["seed"]]) for r in selection["rows"]], selection["root"],
        selection["diagnostic_root"], selection["references"], selection["reuse_completed"],
        purpose="diagnostic_control" if legacy else selection["purpose"])
    if legacy:
        # Retain byte-for-byte historical plans for reading, never promote them.
        rebuilt["schema"] = "facex-selection-v1"
        rebuilt.pop("purpose")
        rebuilt.pop("reproduction_scope")
    if rebuilt != selection:
        raise ValueError("Non-canonical selection")
    return selection


def validate_selection_execution(selection):
    validate_selection(selection)
    if selection["schema"] != SCHEMA:
        raise ValueError("Legacy selection_v1 is read-only. Create a new v2 selection with an explicit purpose; "
                         "preserve the old plan and checkpoints.")


def _baseline_reference(base, dataset, head):
    path = w.checkpoint_path(base, "finetune", head, 8972, dataset)
    export = Path(base["paths"]["export_dir"]) / f"export_{dataset}_{head}_seed8972"
    metadata = export / f"exported_{w.backbone_slug(base['backbone'])}_1024d_{head}_seed8972_meta.json"
    return dict(path=str(path), sha256=read_json(metadata)["source_sha256"])


def _recipe_source(selection, row):
    if row["recipe"] in ("two_eye_alignment", "pfld_five_point_alignment"):
        ds = row["dataset"]
        path = Path(selection["diagnostic_root"]) / "checkpoints" / f"finetune_{ds}_arcface_seed8972" / "finetune_mobilefacenet_1024d_arcface_epoch_40.pt"
        return dict(path=str(path), sha256=WEIGHTS[ds])
    return _baseline_reference(selection["settings"], row["dataset"], "arcface")


def default_references(selection):
    """Known completed sources only; no directory scan or score-based choice."""
    refs = {}
    for row in selection["rows"]:
        if row["seed"] != 8972:
            continue
        if row["recipe"] == "resize_only":
            refs[row["run_id"]] = _baseline_reference(selection["settings"], row["dataset"], row["head"])
        elif row["head"] == "arcface" and row["recipe"] in ("two_eye_alignment", "pfld_five_point_alignment"):
            refs[row["run_id"]] = _recipe_source(selection, row)
    return refs


def settings_for(selection, row, *, reverify=False, allow_verification_retry=False):
    base = selection["settings"]
    cfg = w.stage_settings(base, "finetune")
    source = _recipe_source(selection, row)
    check_file(source["path"], source["sha256"])
    payload = load_checkpoint(source["path"])
    stored = payload["config"]
    validate_checkpoint_for_export(payload, expected_seed=8972, expected_head="arcface",
                                   expected_stage="finetune", expected_parent_hash=PARENT)
    if compute_config_hash(stored) != payload["config_hash"]:
        raise ValueError("Recipe source config hash mismatch")
    profile = cfg["finetune_profiles"][row["dataset"]]
    protocol = w.verify_protocol(profile)
    for key, expected in (("name", row["dataset"]), ("role_manifest_sha256", protocol["roles_sha256"]),
                          ("training_identity_sha256", protocol["training_identity_sha256"])):
        if stored["dataset"][key] != expected:
            raise ValueError(f"Recipe data contract differs: {key}")
    cfg["dataset"] = {**cfg["dataset"], **profile, **stored["dataset"]}
    cfg["dataset"]["val_list"] = None
    prep = stored["preprocessing"]
    for key in ("input_size", "pixel_mean", "pixel_std"):
        if base["dataset"][key] != prep[key]:
            raise ValueError(f"Selected normalization differs from the recipe: {key}")
    cfg["dataset"].update({k: prep[k] for k in ("alignment", "alignment_policy", "input_size", "pixel_mean", "pixel_std")})
    aligned = row["recipe"] in ("two_eye_alignment", "pfld_five_point_alignment")
    report_path = (Path(selection["diagnostic_root"]) / "prepared" / row["dataset"] / "report.json"
                   if aligned else Path(profile["alignment_report"]))
    cfg["dataset"]["alignment_report"] = str(report_path)
    check_file(report_path, stored["dataset"]["alignment_report_sha256"])
    check_file(cfg["dataset"]["train_list"], stored["dataset"]["train_list_hash"])
    report = read_json(report_path)
    verification = Path(selection["root"]) / "data_verification" / (stored["dataset"]["alignment_report_sha256"] + ".json")
    if verification.exists() and read_json(verification).get("status") != "completed" and not (reverify or allow_verification_retry):
        raise ValueError("A forced data verification failed/interrupted; repeat FORCE_DATA_REVERIFY=True")
    if aligned:
        manifest = report_path.parent / "images.csv"
        check_file(manifest, stored["dataset"]["alignment_manifest_sha256"])
        if (report["status"] != "completed" or report["alignment_policy"] != prep["alignment_policy"]
                or report["image_count"] != profile["expected_images"]
                or report["original_role_manifest_sha256"] != protocol["roles_sha256"]):
            raise ValueError("Incomplete or mismatched aligned data")
        cfg["alignment"]["low_resolution"] = prep["alignment_policy"]
        if reverify:
            import cv2
            import numpy as np
            with artifact_lock(verification):
                w.write_json(verification, dict(status="verifying", report_sha256=stored["dataset"]["alignment_report_sha256"]))
                count = 0
                with manifest.open(encoding="utf-8", newline="") as stream:
                    for item in csv.DictReader(stream):
                        image = report_path.parent / item["png"]
                        check_file(image, item["png_sha256"])
                        check_file(Path(profile["raw_root"]) / item["source"], item["source_sha256"])
                        decoded = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
                        if decoded is None or decoded.shape != (112, 112, 3):
                            raise ValueError(f"Invalid prepared image: {image}")
                        count += 1
                if count != report["image_count"]:
                    raise ValueError("Aligned manifest count differs")
                w.write_json(verification, dict(status="completed", report_sha256=stored["dataset"]["alignment_report_sha256"], verified_images=count))
        cfg["data_integrity_verification"] = dict(mode="controlled_alignment_all_verified" if reverify else "immutable_verified_reuse",
            receipt_path=str(report_path), receipt_sha256=compute_sha256(str(report_path)), verified_images=report["image_count"],
            assumption="Prepared images remain immutable after the completed full verification")
    else:
        from .alignment import alignment_policy
        policy = alignment_policy(Path(__file__).resolve().parents[2], cfg["alignment"], row["dataset"])
        if policy != prep["alignment_policy"]:
            raise ValueError("Selected resize policy differs from the prepared recipe")
        if (report.get("completed") is not True or report.get("failed") != 0
                or report.get("prepared") != profile["expected_images"] or report["policy"] != prep["alignment_policy"]
                or not (report_path.parent / "COMPLETED").is_file()):
            raise ValueError("Incomplete resize preprocessing")
        check_file(report["manifest"], stored["dataset"]["alignment_manifest_sha256"])
        if reverify:
            with artifact_lock(verification):
                w.write_json(verification, dict(status="verifying"))
                cfg["dataset"]["train_list"] = profile["train_list"]
                cfg["data_integrity"] = dict(mode="full", force_reverify=True)
                cfg = w.bind_aligned_dataset(cfg)
                w.write_json(verification, dict(status="completed", verified_images=report["prepared"]))
        else:
            cfg["data_integrity_verification"] = dict(mode="immutable_verified_reuse",
                receipt_path=str(report_path), receipt_sha256=stored["dataset"]["alignment_report_sha256"],
                assumption="Completed source checkpoint verified this exact immutable prepared dataset")
    cfg["project"].update(seed=row["seed"], seed_list=[row["seed"]])
    parent = base["campaign"].get("parent_checkpoint") or w.checkpoint_path(base, "pretrain", "arcface", base["campaign"]["parent_seed"])
    check_file(parent, PARENT)
    cfg["campaign"].update(heads=[row["head"]], finetune_datasets=[row["dataset"]], parent_checkpoint=str(parent), parent_sha256=PARENT)
    if row["recipe"] == "arcface_continuous_margin":
        cfg["heads"]["arcface"]["margin_am"] = 1 - math.cos(cfg["heads"]["arcface"]["margin_arc"])
    cfg["paths"] = {k: str(Path(selection["root"]) / k.removesuffix("_dir") / row["run_id"]) for k in cfg["paths"]}
    cfg["selection_provenance"] = dict(row=row, recipe_source=source, driver_sha256=compute_sha256(__file__))
    w.validate_paper_parameters(cfg)
    if cfg["training"]["drop_last"] is not False or cfg["dataset"]["val_list"]:
        raise ValueError("Selected training requires full FR split")
    return cfg


def _expected(cfg, row):
    with torch.random.fork_rng(devices=[]):
        model = w.build_model(cfg["backbone"], dict(cfg["heads"][row["head"]], num_class=cfg["dataset"]["num_classes"]))
    return extract_model_and_train_config(model, cfg, seed=row["seed"], stage="finetune", head_type=row["head"], parent_hash=PARENT)


def _validate(path, cfg, row, *, completed=True, digest=None):
    if digest:
        check_file(path, digest)
    payload = load_checkpoint(str(path))
    if compute_config_hash(payload["config"]) != payload["config_hash"]:
        raise ValueError("Checkpoint config hash mismatch")
    if completed:
        validate_checkpoint_for_export(payload, target_epoch=cfg["training"]["epoches"], expected_seed=row["seed"],
            expected_head=row["head"], expected_stage="finetune", expected_parent_hash=PARENT)
    differences = compare_configs(payload["config"], _expected(cfg, row), full_strict=True)
    if differences:
        raise ValueError("Selected checkpoint differs: " + "; ".join(differences))
    return payload


def final_path(cfg, row):
    return w.checkpoint_path(cfg, "finetune", row["head"], row["seed"], row["dataset"])


def inspect_selection(selection, resume_checkpoints=None, *, force_reverify=False):
    resume = resume_checkpoints or {}
    if set(selection.get("references", {})) - {r["run_id"] for r in selection["rows"]}:
        raise ValueError("Reference keys must be selected run_id values")
    if set(resume) - {r["run_id"] for r in selection["rows"]}:
        raise ValueError("Resume keys must be selected run_id values")
    result = []
    for row in selection["rows"]:
        try:
            if selection.get("root"):
                lock = Path(selection["root"]) / "campaign.lock"
                if lock.exists() and read_json(lock).get("pid") != os.getpid():
                    raise FileExistsError(f"Campaign lock exists; inspect recorded PID before removal: {lock}")
            cfg = settings_for(selection, row, allow_verification_retry=force_reverify)
            target = final_path(cfg, row)
            ref = selection["references"].get(row["run_id"])
            path = Path(ref["path"]) if ref else target
            if ref and (set(ref) != {"path", "sha256"} or len(ref["sha256"]) != 64):
                raise ValueError("Explicit references require path and SHA-256")
            if path.is_file():
                _validate(path, cfg, row, digest=ref["sha256"] if ref else None)
                if not selection["reuse_completed"]:
                    raise FileExistsError("Completed model exists; enable REUSE_COMPLETED")
                status = "reuse_completed"
            elif ref:
                raise FileNotFoundError(f"Explicit reuse source is missing: {path}")
            elif row["run_id"] in resume:
                path = Path(resume[row["run_id"]])
                _validate(path, cfg, row, completed=False)
                status = "resume"
            elif list(target.parent.glob("*.pt")):
                raise FileExistsError("Interrupted run: specify RESUME_CHECKPOINTS explicitly")
            else:
                status = "train_new"
            result.append({**row, "status": status, "checkpoint": str(path), "target": str(target)})
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            result.append({**row, "status": "blocked", "reason": str(exc)})
    return result


def _train(cfg, row, resume=None):
    """Use the canonical Trainer, optimizer, sampler, recovery and numerical policy."""
    import cv2
    from .trainer import Trainer, set_seed
    device = w._formal_device(cfg)
    torch.set_num_threads(cfg["training"].get("torch_threads", 2))
    cv2.setNumThreads(1)
    set_seed(row["seed"])
    parent, digest = verify_parent_checkpoint(cfg["campaign"]["parent_checkpoint"], expected_hash=PARENT,
        require_hash=True, require_completed=True, expected_seed=cfg["campaign"]["parent_seed"], target_epoch=40)
    w.validate_parent_preprocessing(parent["config"], cfg["dataset"], cfg["alignment"])
    fingerprints = parent["config"]["code_fingerprints"]
    if fingerprints != w.code_fingerprints() and not w.source_transition(fingerprints, w.code_fingerprints()):
        raise ValueError("Parent source fingerprints differ")
    target = final_path(cfg, row)
    model = trainer = train_loader = val_loader = None
    try:
        model = w.build_model(cfg["backbone"], dict(cfg["heads"][row["head"]], num_class=cfg["dataset"]["num_classes"])).to(device)
        model.backbone.load_state_dict(parent["backbone_state_dict"], strict=True)
        train_loader, val_loader = w.make_loaders(cfg)
        trainer = Trainer(model, train_loader, val_loader, cfg, CheckpointManager(str(target.parent)), device,
                          parent_hash=digest, head_type=row["head"], stage="finetune")
        start = trainer.resume_from_checkpoint(str(resume)) if resume else 1
        snapshot = target.parent / (f"resume_settings_{uuid.uuid4().hex}.json" if resume else "notebook_settings.json")
        w.write_json(snapshot, cfg, immutable=True)
        trainer.train(start_epoch=start, model_prefix=target.stem.rsplit("_epoch_", 1)[0])
        trainer.save_state_checkpoint(target.name, epoch=trainer.cur_epoch, extra_meta={"execution_settings": cfg})
        _validate(target, cfg, row)
        return target
    finally:
        for loader in (train_loader, val_loader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
        del trainer, model, parent, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


def _verify_export(output, payload, source_hash):
    meta = read_json(output.with_name(output.stem + "_meta.json"))
    check_file(output, meta["sha256"])
    if meta["source_sha256"] != source_hash or meta["source_config"] != payload["config"]:
        raise ValueError("Existing selected export belongs to different training")
    published = torch.load(output, map_location="cpu", weights_only=True)
    if published["metadata"] != {k: v for k, v in meta.items() if k not in ("sha256", "exported_file")}:
        raise ValueError("Selected export sidecar differs from embedded metadata")
    if meta.get("input_geometry"):
        check_file(output.parent / "affine.csv", meta["input_geometry"]["sha256"])
    weights = published["state_dict"]
    original = payload["backbone_state_dict"]
    if weights.keys() != original.keys() or any(not torch.equal(weights[k], v.cpu()) for k, v in original.items()):
        raise ValueError("Export weights differ from trained backbone")
    return meta


def _inventory_row(row, output, meta, path, source_hash):
    return {**row, "completed": True, "export": str(output), "sha256": meta["sha256"],
            "source_checkpoint": str(path), "source_sha256": source_hash,
            "evaluation_datasets": [row["dataset"]], "training_dataset": row["dataset"]}


def export_selected(selection, row, cfg, path):
    payload = _validate(path, cfg, row)
    source_hash = compute_sha256(str(path))
    known = default_references(selection).get(row["run_id"])
    if known and source_hash == known["sha256"]:
        # Keep existing model UIDs and compatible embedding caches reusable.
        base = selection["settings"]
        if row["recipe"] == "resize_only":
            export_dir = Path(base["paths"]["export_dir"]) / f"export_{row['dataset']}_{row['head']}_seed{row['seed']}"
            output = export_dir / f"exported_{w.backbone_slug(cfg['backbone'])}_1024d_{row['head']}_seed{row['seed']}.pt"
        else:
            result = export_alignment_control(base, row["dataset"], selection["diagnostic_root"],
                Path(base["paths"]["export_dir"]).parent / ALIGNMENT_CONTROL)
            output = Path(result["export"])
        return _inventory_row(row, output, _verify_export(output, payload, source_hash), path, source_hash)
    destination = Path(selection["root"]) / "exports" / row["run_id"]
    filename = row["run_id"] + ".pt"
    with artifact_lock(destination):
        if not destination.exists():
            geometry = None
            if row["recipe"] in ("two_eye_alignment", "pfld_five_point_alignment"):
                base_export = Path(selection["settings"]["paths"]["export_dir"]).parent / ALIGNMENT_CONTROL
                control = export_alignment_control(selection["settings"], row["dataset"], selection["diagnostic_root"], base_export)
                control_path = Path(control["export"])
                control_meta = read_json(control_path.with_name(control_path.stem + "_meta.json"))
                geometry = (control_path.parent / "affine.csv", control_meta["input_geometry"])
            with TemporaryDirectory(dir=destination.parent) as temp:
                staging = Path(temp) / row["run_id"]
                staging.mkdir()
                metadata = {**row, "stage": "export", "completed": True, "is_smoke": False, "feat_dim": 1024,
                    "epochs": payload["epoch"], "parent_hash": PARENT, "common_parent_seed": cfg["campaign"]["parent_seed"],
                    "source_checkpoint": str(path), "source_sha256": source_hash, "source_config": payload["config"],
                    "source_config_hash": payload["config_hash"], "preprocessing": payload["config"]["preprocessing"],
                    "execution_settings": payload.get("extra_meta", {}).get("execution_settings"),
                    "protocol_record": read_json(cfg["finetune_profiles"][row["dataset"]]["protocol_record"]),
                    "selection_provenance": cfg["selection_provenance"],
                    "reproduction_scope": "Selected local recipe; no claim of exact UCFace reproduction"}
                if geometry:
                    shutil.copyfile(geometry[0], staging / "affine.csv")
                    metadata["input_geometry"] = geometry[1]
                CheckpointManager(str(staging)).export_backbone_only(payload, filename, metadata)
                staging.rename(destination)
        output = destination / filename
        meta = _verify_export(output, payload, source_hash)
    return _inventory_row(row, output, meta, path, source_hash)


def _estimated_checkpoint_bytes(selection, plan, stage):
    estimated = 0
    for item in plan:
        if stage == "finetune" and item["status"] != "reuse_completed":
            size = Path(_recipe_source(selection, item)["path"]).stat().st_size
            training = selection["settings"]["training"]
            retained = (training["epoches"] // training["save_freq"]
                        if selection["settings"]["project"]["keep_raw_results"] else 0)
            estimated += size * (retained + 4)
    return estimated


def execute_selection(selection, stage, resume=None, runtime=None, force_reverify=False):
    if stage not in ("finetune", "export"):
        raise ValueError("Selected campaign supports finetune/export only")
    validate_selection_execution(selection)
    root = Path(selection["root"])
    # Non-immutable modes request a fresh check; the aligned recipe uses the
    # stronger hash+decode check for both full and content_verified_reuse.
    force_reverify = force_reverify or selection.get("settings", {}).get("data_integrity", {}).get(
        "mode", "immutable_verified_reuse") != "immutable_verified_reuse"
    attempt = root / "attempts" / uuid.uuid4().hex
    results = []
    with artifact_lock(root / "campaign"):
        plan = inspect_selection(selection, resume, force_reverify=force_reverify)
        w.write_json(attempt / "plan.json", dict(selection=selection, rows=plan), immutable=True)
        if any(r["status"] == "blocked" for r in plan):
            raise ValueError(f"Blocked selection; inspect {attempt / 'plan.json'}")
        if stage == "export" and any(r["status"] != "reuse_completed" for r in plan):
            raise ValueError("04 requires every selected model to be completed; export an explicit smaller plan if desired")
        estimated = _estimated_checkpoint_bytes(selection, plan, stage)
        free = shutil.disk_usage(root).free
        w.write_json(attempt / "storage.json", dict(free_bytes=free, estimated_training_checkpoint_bytes=estimated,
            basis="Reference full-checkpoint size times retention count plus four working copies; excludes unrelated writers and exports"))
        if free < estimated:
            raise OSError(f"Insufficient estimated checkpoint space: need {estimated} bytes; free {free}")
        verified = set()
        try:
            for item in plan:
                row = {k: item[k] for k in ("run_id", "dataset", "head", "recipe", "seed")}
                if runtime and Path(runtime["stop_file"]).exists():
                    raise InterruptedError("Selected campaign stop requested")
                data_key = (row["dataset"], row["recipe"])
                cfg = settings_for(selection, row, reverify=force_reverify and data_key not in verified)
                verified.add(data_key)
                cfg["runtime"] = runtime or {}
                path = Path(item["checkpoint"])
                with artifact_lock(Path(item["target"]).parent):
                    if stage == "finetune" and item["status"] != "reuse_completed":
                        path = _train(cfg, row, path if item["status"] == "resume" else None)
                    result = (export_selected(selection, row, cfg, path) if stage == "export" else
                              {**row, "completed": True, "status": "reused_completed" if item["status"] == "reuse_completed" else "completed",
                               "action": item["status"], "checkpoint": str(path), "sha256": compute_sha256(str(path))})
                results.append(result)
                w.write_json(attempt / "progress.json", dict(status="running", results=results, remaining=plan[len(results):]))
        except BaseException as exc:
            w.write_json(attempt / "result.json", dict(status="failed", error=repr(exc), results=results,
                failed=plan[len(results)], not_started=plan[len(results)+1:]))
            raise
        w.write_json(attempt / "result.json", dict(status="completed", results=results), immutable=True)
        if stage == "export":
            inventory = dict(schema=INVENTORY_SCHEMA, status="completed", selection=selection, models=results)
            inventory_path = root / "inventories" / (compute_config_hash(inventory) + ".json")
            w.write_json(inventory_path, inventory, immutable=True)
            print(f"MODEL_INVENTORY = {str(inventory_path)!r}", flush=True)
        return results
