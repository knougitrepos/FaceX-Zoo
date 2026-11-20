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
