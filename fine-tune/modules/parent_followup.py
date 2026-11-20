"""Finish one explicitly approved Parent control through the Origin comparison.

An immutable plan pins the existing Parent process, source files, downstream
reference and consumer settings. Failed stages stop; no checkpoint is chosen
automatically after an interruption.
"""
from __future__ import annotations

import copy
import gc
import json
import logging
from pathlib import Path
import shutil
import sys
import traceback
from datetime import datetime, timezone, timedelta

from .checkpoint_manager import compute_sha256, compute_config_hash
from .notebook_workflow import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def checked_file(record):
    path = Path(record["path"])
    if compute_sha256(path) != record["sha256"]:
        raise ValueError(f"Pinned artifact changed: {path}")
    return path


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=compute_sha256(path))


def verify_sources(plan):
    for name, digest in plan["sources"].items():
        checked_file(dict(path=name, sha256=digest))
    for name in ("launch", "request", "followup_reference", "baseline_yaml", "validation_protocol", "interpretation_guide"):
        checked_file(plan[name])


def prepare_plan(launch_path, ronbun_root, baseline_yaml, output_root, registry, reference_report,
                 reference_model_uid, *, keep_raw_results=True):
    """Pin a plan only; this function does not start training or inference."""
    from . import notebook_workflow as w
    from .parent_controls import GUIDE
    if type(keep_raw_results) is not bool:
        raise ValueError("KEEP_RAW_RESULTS must be boolean")
    ronbun_root = Path(ronbun_root).resolve()
    if str(ronbun_root) not in sys.path:
        sys.path.insert(0, str(ronbun_root))
    from research.runtime.source_fingerprints import source_fingerprints
    from research.experiments.facex_workflow import _config
    import yaml
    launch = read_json(launch_path)
    request = read_json(launch["request"])
    cfg = request["settings"]
    if request["stage"] != "pretrain" or cfg["runtime"].get("stop_after_epoch") is not None:
        raise ValueError("Expected the explicit full Parent continuation")
    root = Path(cfg["parent_control"]["output_root"])
    destination = root / "followup_execution"
    facex = Path(__file__).resolve().parents[2]
    sources = {str(facex / name): digest for name, digest in w.code_fingerprints().items()}
    for path in (facex / "fine-tune/modules").glob("*.py"):
        sources[str(path)] = compute_sha256(path)
    for name in ("run_campaign.py", "run_parent_control.py", "watch_parent_followup.py"):
        path = facex / "fine-tune/scripts" / name
        sources[str(path)] = compute_sha256(path)
    for name, digest in source_fingerprints(
            "research.experiments.facex_workflow", "research.experiments.facex_inventory",
            "research.experiments.facex_comparison", "research.embeddings.pytorch.facex_zoo").items():
        sources[str(ronbun_root / name)] = digest
    resolved = _config(baseline_yaml)
    baseline = destination / "baseline_resolved.yaml"
    content = yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False)
    baseline.parent.mkdir(parents=True, exist_ok=True)
    if baseline.exists() and baseline.read_text(encoding="utf-8") != content:
        raise FileExistsError("A different frozen consumer baseline already exists")
    if not baseline.exists():
        with baseline.open("x", encoding="utf-8") as stream:
            stream.write(content)
    guide = destination / "HOW_TO_READ.md"
    if guide.exists() and compute_sha256(guide) != compute_sha256(GUIDE):
        raise FileExistsError("A different frozen interpretation guide already exists")
    if not guide.exists():
        shutil.copyfile(GUIDE, guide)
    plan = dict(schema="facex-parent-followup-v1", created_at=now(),
                root=str(root), facex_root=str(facex), ronbun_root=str(ronbun_root),
                launch=pin(launch_path), request=pin(launch["request"]),
                followup_reference=pin(root / "followup_reference.json"),
                baseline_yaml=pin(baseline), baseline_source=pin(baseline_yaml),
                interpretation_guide=pin(guide),
                validation_protocol=pin(root / "validation/vggface2_fixed_200x10_v1/protocol.json"),
                output_root=str(Path(output_root).resolve()), registry=str(Path(registry).resolve()),
                reference_report=str(Path(reference_report).resolve()), reference_model_uid=reference_model_uid,
                reference_report_manifest=pin(Path(reference_report) / "run_manifest.json"),
                keep_raw_results=keep_raw_results, sources=dict(sorted(sources.items())),
                scope="One full-FR SurvFace MagFace; uncompressed test ROC comparison only. No FR-unseen calibration claim.")
    path = destination / "plans" / (compute_config_hash(plan) + ".json")
    write_json(path, plan, immutable=True)
    return path


def require_parent_result(launch, request, root):
    from .parent_controls import verify_control_parent
    from .notebook_workflow import checkpoint_path
    results = read_json(launch["result"])
    if (len(results) != 1 or results[0].get("completed") is not True
            or results[0].get("status") not in ("completed", "reused_completed")):
        raise ValueError("Parent has not completed its full schedule")
    result = results[0]
    expected = checkpoint_path(request["settings"], "pretrain", "arcface", 8972).resolve()
    path = checked_file(dict(path=result["checkpoint"], sha256=result["sha256"])).resolve()
    if path != expected or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Result is outside the explicitly selected Parent run")
    binding = dict(path=str(path), sha256=result["sha256"],
                   reference=request["settings"]["parent_control"]["reference"])
    verify_control_parent(binding)
    return binding


def require_same_finetune(reference, current, new_parent_hash):
    """Use exact scientific values; an optimizer tolerance must not waive drift."""
    from .notebook_workflow import source_transition
    old, new = copy.deepcopy(reference), copy.deepcopy(current)
    if (old["code_fingerprints"] != new["code_fingerprints"]
            and not source_transition(old["code_fingerprints"], new["code_fingerprints"])):
        raise ValueError("Unreviewed downstream source transition")
    old["parent_hash"] = new_parent_hash
    for cfg in (old, new):
        cfg.pop("config_hash", None)
        cfg.pop("code_fingerprints", None)
    if old != new:
        raise ValueError("SurvFace fine-tune differs beyond the approved Parent replacement")


def origin_rows(result, model_uid=None):
    """Select descriptive test ROC fields, never the Global-safe acceptance rate."""
    rows = result["metrics"]
    rows = rows.loc[rows.dataset.eq("survface") & rows.compression_profile.eq("origin_1024")
                    & rows.method.eq("global_safe")].copy()
    if model_uid is not None:
        rows = rows.loc[rows.model_uid.eq(model_uid)]
    if (rows.model_uid.nunique() != 1 or rows.target_fpir.duplicated().any()
            or set(rows.target_fpir) != {.01, .05, .10, .20}):
        raise ValueError("Expected exactly one Origin model at all four Table I operating points")
    columns = ["target_fpir", "model_uid", "mated_count", "non_mated_count",
               "origin_test_roc_tpir_at_20", "origin_test_roc_fpir", "unknown_cosine_ge_095_count"]
    if rows[columns].isna().any().any():
        raise ValueError("Missing Origin test ROC evidence")
    if ((rows.origin_test_roc_fpir > rows.target_fpir).any()
            or (rows.origin_test_roc_fpir < 0).any()
            or not rows.origin_test_roc_tpir_at_20.between(0, 1).all()):
        raise ValueError("Descriptive test ROC does not meet its claimed operating point")
    return rows[columns]


def comparison_table(old_result, new_result, reference_uid):
    from research.experiments.facex_comparison import UCFACE_TABLE_I_REFERENCE
    old, new = origin_rows(old_result, reference_uid), origin_rows(new_result)
    result = old.merge(new, on="target_fpir", suffixes=("_reference", "_control"), validate="one_to_one")
    for key in ("mated_count", "non_mated_count"):
        # Mixed-dataset historical CSVs infer float counts because rank rows
        # contain NaN; a SurvFace-only CSV can infer the same counts as integers.
        if (result[f"{key}_reference"].to_numpy() != result[f"{key}_control"].to_numpy()).any():
            raise ValueError("Reference and control test cohort counts differ")
    result["reference_tpir_pct"] = result.origin_test_roc_tpir_at_20_reference * 100
    result["control_tpir_pct"] = result.origin_test_roc_tpir_at_20_control * 100
    result["control_minus_reference_pp"] = result.control_tpir_pct - result.reference_tpir_pct
    result["ucface_table_i_tpir_pct"] = result.target_fpir.map(UCFACE_TABLE_I_REFERENCE["magface"]["survface"])
    result["control_minus_ucface_pp"] = result.control_tpir_pct - result.ucface_table_i_tpir_pct
    return result.sort_values("target_fpir", ascending=False)


def execute_plan(plan_path):
    import psutil
    from . import parent_controls as controls, selected_campaign as selected
    from .parent_validation import evaluate_parent, compare_parents
    from .process_runner import launch_campaign
    from .shared_storage import artifact_lock
    plan = read_json(plan_path)
    if plan.get("schema") != "facex-parent-followup-v1":
        raise ValueError("Unsupported followup plan")
    verify_sources(plan)
    root = Path(plan["root"])
    work = root / "followup_execution"
    if (work / "COMPLETED").exists():
        raise FileExistsError("This followup is already completed; inspect its immutable summary")
    launch, request = read_json(plan["launch"]["path"]), read_json(plan["request"]["path"])
    state = dict(phase="waiting_for_parent", plan=pin(plan_path), updated_at=now())

    def phase(name, **values):
        state.update(phase=name, updated_at=now(), **values)
        write_json(work / "status.json", state)
        logging.info("Parent followup phase: %s", name)

    with artifact_lock(work / "worker"):
        try:
            phase("waiting_for_parent", parent_pid=launch["pid"])
            try:
                process = psutil.Process(launch["pid"])
                if (launch["request"] not in process.cmdline()
                        or "run_parent_control.py" not in " ".join(process.cmdline())):
                    raise ValueError("Parent PID belongs to a different command")
                while True:
                    try:
                        code = process.wait(timeout=30)
                        if code != 0:
                            raise RuntimeError(f"Parent process exited with {code}")
                        break
                    except psutil.TimeoutExpired:
                        phase("waiting_for_parent", parent_pid=process.pid)
            except psutil.NoSuchProcess:
                logging.info("Parent no longer running; checking the explicit terminal result")
            verify_sources(plan)
            binding = require_parent_result(launch, request, root)
            phase("evaluating_completed_parent", parent=binding)
            validation = root / "validation/vggface2_fixed_200x10_v1"
            before = evaluate_parent(binding["reference"]["path"], validation, root / "validation/evaluations",
                                     keep_raw_results=plan["keep_raw_results"])
            after = evaluate_parent(binding["path"], validation, root / "validation/evaluations",
                                    keep_raw_results=plan["keep_raw_results"])
            paired = root / "validation/comparisons" / f"epoch_40_{after.parent.name}.json"
            compare_parents(before, after, paired)
            write_json(root / "final_parent_report.json", dict(status="parent_completed_and_evaluated",
                parent=binding, comparison=read_json(paired), comparison_file=pin(paired)), immutable=True)
            import torch
            gc.collect()
            torch.cuda.empty_cache()

            frozen = read_json(checked_file(plan["followup_reference"]))
            checked_file(dict(path=frozen["reference_model"]["source_checkpoint"],
                              sha256=frozen["reference_model"]["source_sha256"]))
            base = copy.deepcopy(frozen["baseline_settings"])
            base["project"]["keep_raw_results"] = plan["keep_raw_results"]
            selection_path = controls.make_followup_selection(base, binding["path"], binding["sha256"], root)
            selection = selected.load_selection(selection_path)
            if len(selection["rows"]) != 1 or selection["rows"] != frozen["planned_rows"]:
                raise ValueError("Followup must contain the exact single approved MagFace")
            cfg = selected.settings_for(selection, selection["rows"][0])
            expected = selected._expected(cfg, selection["rows"][0])
            require_same_finetune(frozen["reference_config"], expected, binding["sha256"])
            write_json(work / "finetune_preflight.json", dict(status="verified", selection=pin(selection_path),
                parent=binding, reference_checkpoint=frozen["reference_model"], expected_config=expected,
                only_scientific_change="parent_hash"), immutable=True)
            verify_sources(plan)
            phase("finetuning_survface_magface", selection=str(selection_path))
            trained = launch_campaign(plan["facex_root"], selection["settings"], "finetune", {}, selection=selection)
            if len(trained) != 1 or trained[0].get("completed") is not True:
                raise ValueError("The single MagFace did not complete")
            write_json(work / "finetune_result.json", trained, immutable=True)
            verify_sources(plan)
            phase("exporting_survface_magface")
            exports = launch_campaign(plan["facex_root"], selection["settings"], "export", {}, selection=selection)
            inventory = dict(schema=selected.INVENTORY_SCHEMA, status="completed", selection=selection, models=exports)
            inventory_path = root / "followup/inventories" / (compute_config_hash(inventory) + ".json")
            if read_json(inventory_path) != inventory:
                raise ValueError("Export did not publish the expected immutable inventory")

            if plan["ronbun_root"] not in sys.path:
                sys.path.insert(0, plan["ronbun_root"])
            from research.experiments.facex_inventory import write_inventory_config
            from research.experiments.facex_workflow import prepare_inputs, run_workflow, read_results
            verify_sources(plan)
            phase("preparing_origin_inputs", inventory=str(inventory_path))
            config = write_inventory_config(inventory_path, plan["baseline_yaml"]["path"], plan["registry"],
                work / "origin_config.yaml", purpose="diagnostic_control", origin_only=True)
            inputs = prepare_inputs(config, datasets=["survface"], output_root=plan["output_root"])
            write_json(work / "origin_inputs.json", inputs, immutable=True)
            verify_sources(plan)
            phase("evaluating_origin", config=str(config))
            result = run_workflow(config, datasets=["survface"], output_root=plan["output_root"],
                                  keep_raw_results=plan["keep_raw_results"], study_stage="baseline")
            new = read_results(result["report_dir"])
            checked_file(plan["reference_report_manifest"])
            old = read_results(plan["reference_report"])
            table = comparison_table(old, new, plan["reference_model_uid"])
            # Check the actual test role contract, beyond equal sample counts.
            new_handoff = new["manifest"]["contract"]["config"]["handoff"]
            old_handoff = old["manifest"]["contract"]["config"]["handoff"]
            if compute_sha256(new_handoff) != compute_sha256(old_handoff):
                raise ValueError("Reference/control protocol handoffs differ")
            table_path = work / "origin_comparison.csv"
            if table_path.exists():
                raise FileExistsError("Comparison already exists; verify the existing completed outcome")
            table.to_csv(table_path, index=False)
            summary = dict(status="completed", plan=pin(plan_path), parent=binding,
                parent_report=pin(root / "final_parent_report.json"), inventory=pin(inventory_path),
                config=pin(config), input_index=pin(inputs["input_index"]),
                report_dir=result["report_dir"], reference_report=plan["reference_report"],
                comparison=pin(table_path), metrics=table.to_dict("records"), keep_raw_results=plan["keep_raw_results"],
                interpretation="Compare Origin test ROC TPIR@20 at the same FPIR. Ancillary calibrated fields of the existing runner are not this diagnostic's evidence. Full-FR calibration is not FR-unseen.",
                limitations="One local seed versus the paper's five-run mean. No guarantee or exact author-implementation claim.")
            write_json(work / "summary.json", summary, immutable=True)
            bundle_files = ["HOW_TO_READ.md", "summary.json", "origin_comparison.csv"]
            write_json(work / "chatgpt_bundle.json", dict(files=bundle_files,
                bytes=sum((work / name).stat().st_size for name in bundle_files),
                estimated_text_tokens=sum(len((work / name).read_text(encoding="utf-8")) for name in bundle_files),
                token_estimate_method="Conservative one Unicode character per token; not an exact tokenizer"), immutable=True)
            (work / "COMPLETED").write_text(compute_sha256(work / "summary.json") + "\n", encoding="utf-8")
            phase("completed", summary=str(work / "summary.json"))
            return summary
        except BaseException as exc:
            write_json(work / "failure.json", dict(failed_phase=state["phase"], error=repr(exc), traceback=traceback.format_exc(), at=now()))
            phase("failed", error=repr(exc))
            raise
