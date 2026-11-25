"""Parent comparison, reviewed CosFace probe, and completed-only Origin handoff."""

import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .alignment_resume import recovery_lock
from .backbone_factory import build_backbone
from .checkpoint_manager import CheckpointManager, compute_sha256, load_checkpoint
from .clean_benchmarks import LFW_ALIGNMENT, checked, evaluate_aligned, evaluate_survface, monitor, prepare_panels
from .clean_protocols import (
    DATASETS,
    HEADS,
    METRICS,
    PARENT_SOURCES,
    SCOPES,
    TABLE_I,
    TABLE_XIII,
    geometry_source,
)
from .clean_training import (
    compatible_completed_implementation,
    final_artifact,
    files_for_protocol,
    load_parent,
    matrix_recipe,
    publish_scope_reuse,
    publish_summary,
    run_stage,
)
from .notebook_workflow import write_json
from .shared_storage import artifact_lock


def validate_choices(datasets, heads, scope, parent_source):
    if scope not in SCOPES or parent_source not in PARENT_SOURCES:
        raise ValueError("Invalid protocol scope or parent source")
    for values, allowed in ((datasets, DATASETS), (heads, HEADS)):
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f"Select unique values from {allowed}")


def model_dir(run, scope, source, dataset, head):
    validate_choices([dataset], [head], scope, source)
    return Path(run) / "models" / scope / source / dataset / head


def evaluation_dir(folder, dataset, *, probe=False):
    return folder / ("probe_origin" if probe and dataset == "tinyface" else "origin")


def read_result(folder):
    summary = json.loads(
        checked(folder / "summary.json", (folder / "COMPLETED").read_text().strip()).read_bytes()
    )
    if summary["status"] != "completed":
        raise ValueError("Incomplete result")
    checked(folder / "plan.json", summary["plan_sha256"])
    bundle = json.loads((folder / "bundle.json").read_bytes())
    for name in ("summary.json", "plan.json", "HOW_TO_READ.md"):
        checked(folder / name, bundle["sha256"][name])
    return summary


def _record(folder, plan, calculate):
    with artifact_lock(folder):
        plan_path = folder / "plan.json"
        if (folder / "COMPLETED").exists():
            result = read_result(folder)
            saved_plan = json.loads(plan_path.read_bytes())
            # Downstream metrics are unchanged by the v3 Parent and LFW corrections.
            if "head" in plan and compatible_completed_implementation(saved_plan["implementation"]):
                if saved_plan["implementation"]["clean_protocols"] == plan["implementation"]["clean_protocols"]:
                    saved_plan["implementation"] = plan["implementation"]
            if saved_plan != plan:
                raise ValueError(f"Completed evaluation differs from requested plan: {folder}")
            return result
        write_json(plan_path, plan, immutable=True)
        if (folder / "summary.json").is_file():
            result = json.loads((folder / "summary.json").read_bytes())
            if result["status"] != "completed" or result["plan_sha256"] != compute_sha256(plan_path):
                raise ValueError("Invalid interrupted evaluation publication")
        else:
            result = dict(
                status="completed",
                guide="HOW_TO_READ.md",
                plan_sha256=compute_sha256(plan_path),
                **plan,
                **calculate(),
            )
        publish_summary(folder, result)
        return result


def _model(state):
    if not torch.cuda.is_available():
        raise RuntimeError("Clean evaluation requires CUDA")
    torch.set_num_threads(2)
    model = build_backbone(dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=0.0))
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def _percentages(dataset, report):
    if dataset == "survface":
        return {key: 100 * report[key]["tpir_at_20"] for key in METRICS[dataset]}
    if dataset == "scface":
        return {
            **{"d" + d: 100 * row["rank1"] for d, row in report["distances"].items()},
            "average": 100 * report["rank1"],
        }
    return {f"rank1_{key}": 100 * value["rank1"] for key, value in report.items()}


def _comparison(dataset, observed, reference):
    return [
        dict(
            dataset=dataset,
            metric=metric,
            observed_pct=value,
            paper_pct=reference.get(metric),
            difference_pp=value - reference[metric] if metric in reference else None,
        )
        for metric, value in observed.items()
    ]


def parent_validation(root, run, *, seed=8972, keep_raw_results=False):
    root, run = Path(root).resolve(), Path(run).resolve()
    _, current = final_artifact(run / "parent")
    stored_panel = current["epochs"][-1]["zero_shot"]["metrics"]
    folder = run / "parent_validation"
    parent_inputs = json.loads((run / "parent/plan.json").read_bytes())["monitor_inputs"]
    if parent_inputs.get("lfw_alignment") != LFW_ALIGNMENT and folder.exists():
        # Keep historical validation records readable at their original path.
        if (folder / "plan.json").is_file() and not json.loads((folder / "plan.json").read_bytes()).get("lfw_alignment"):
            folder = run / ("parent_validation_" + LFW_ALIGNMENT)
    from .clean_protocols import LEGACY_SHA

    plan = dict(
        parent_sha256=current["sha256"],
        legacy_parent_sha256=LEGACY_SHA,
        data_root=str(root),
        keep_raw_results=keep_raw_results,
        seed=seed,
        implementation=files_for_protocol(),
        lfw_alignment=LFW_ALIGNMENT,
    )

    def calculate():
        payload, legacy_sha = load_parent(run, "legacy_parent_wd0", seed)
        legacy = _model(payload["backbone_state_dict"])
        del payload
        panels, info = prepare_panels(root)
        if info["raw_images_sha256"] != parent_inputs["raw_images_sha256"]:
            raise ValueError("Parent monitoring pixels differ from the legacy comparison inputs")
        legacy_panel = monitor(
            legacy, panels, raw_file=folder / "legacy_panels.npz" if keep_raw_results else None
        )["metrics"]
        legacy_roc = evaluate_survface(
            legacy, root, folder / "legacy_survface.npz" if keep_raw_results else None
        )
        del legacy
        payload, clean_sha = load_parent(run, "clean_parent", seed)
        model = _model(payload["backbone_state_dict"])
        del payload
        clean_panel = stored_panel
        if parent_inputs.get("lfw_alignment") != info["lfw_alignment"]:
            # An earlier Parent used unaligned LFW inputs. Evaluate both weights on the same pixels.
            clean_panel = monitor(model, panels, raw_file=folder / "clean_panels.npz" if keep_raw_results else None)["metrics"]
        del panels
        clean_roc = evaluate_survface(
            model, root, folder / "clean_survface.npz" if keep_raw_results else None
        )
        table = []
        for ds in DATASETS:
            if ds == "survface":
                clean, old = _percentages(ds, clean_roc), _percentages(ds, legacy_roc)
                table.append(
                    dict(
                        dataset=ds,
                        metric="unknown_cos_ge_095",
                        observed_count=clean_roc["unknown_cos_ge_095"],
                        legacy_count=legacy_roc["unknown_cos_ge_095"],
                    )
                )
            elif ds == "scface":
                clean, old = (_percentages(ds, panel[ds]) for panel in (clean_panel, legacy_panel))
            else:
                clean, old = (
                    _percentages(ds, {"without_distractors": panel[ds]})
                    for panel in (clean_panel, legacy_panel)
                )
            table.extend(
                {
                    **row,
                    "legacy_pct": old[row["metric"]],
                    "vs_legacy_pp": row["observed_pct"] - old[row["metric"]],
                }
                for row in _comparison(ds, clean, TABLE_XIII[ds])
            )
        table.append(
            dict(
                dataset="lfw",
                metric="accuracy",
                observed_pct=100 * clean_panel["lfw"]["accuracy"],
                legacy_pct=100 * legacy_panel["lfw"]["accuracy"],
            )
        )
        if clean_sha != current["sha256"] or legacy_sha != plan["legacy_parent_sha256"]:
            raise ValueError("Parent comparison lineage mismatch")
        return dict(
            comparison=table,
            clean_roc=clean_roc,
            legacy_roc=legacy_roc,
            clean_panel=clean_panel,
            legacy_panel=legacy_panel,
        )

    return _record(folder, plan, calculate)


def verified_model(root, run, scope, source, dataset, head, seed, *, keep_raw_results=None, parent_sha=None):
    run = Path(run).resolve()
    folder = model_dir(run, scope, source, dataset, head)
    r = matrix_recipe(run, root, dataset, head, scope)
    checkpoint, receipt = final_artifact(folder, expected_recipe=r)
    plan = json.loads((folder / "plan.json").read_bytes())
    if parent_sha is None:
        _, parent_sha = load_parent(run, source, seed)
    if (
        plan["initial_weights"] != source
        or plan["seed"] != seed
        or plan["run_dir"] != str(run)
        or receipt["parent_sha256"] != parent_sha
        or not compatible_completed_implementation(plan["implementation"])
        or (keep_raw_results is not None and plan["keep_raw_results"] != keep_raw_results)
    ):
        raise ValueError("Completed model differs from selected Parent, seed or settings")
    return checkpoint, receipt, r, plan


def _train_or_reuse(root, run, scope, source, dataset, head, seed, parent_sha, keep_raw_results, resume):
    folder = model_dir(run, scope, source, dataset, head)
    shared = dataset in ("tinyface", "scface") and source == "clean_parent"
    lock = recovery_lock if resume else artifact_lock
    # Both scopes serialize selection/publication for the same training job.
    with lock(run / "models" / "shared_scope_locks" / dataset / head) if shared else nullcontext():
        if (folder / "COMPLETED").is_file():
            verified_model(
                root,
                run,
                scope,
                source,
                dataset,
                head,
                seed,
                keep_raw_results=keep_raw_results,
                parent_sha=parent_sha,
            )
            return "verified_reuse"

        r = matrix_recipe(run, root, dataset, head, scope)
        if shared:
            peer = next(candidate for candidate in SCOPES if candidate != scope)
            if (model_dir(run, peer, source, dataset, head) / "COMPLETED").is_file():
                checkpoint, receipt, _, plan = verified_model(
                    root, run, peer, source, dataset, head, seed, parent_sha=parent_sha
                )
                publish_scope_reuse(folder, r, checkpoint, receipt, plan, keep_raw_results)
                return "verified_cross_scope_reuse"

        run_stage(
            root,
            run,
            "finetune",
            seed=seed,
            keep_raw_results=keep_raw_results,
            resume=resume and folder.exists(),
            matrix=(folder, r),
            parent_source=source,
        )
        return "completed"


def train_models(
    root, run, scope, source, datasets, heads, *, seed=8972, keep_raw_results=False, resume=False
):
    validate_choices(datasets, heads, scope, source)
    run = Path(run).resolve()
    _, parent_sha = load_parent(run, source, seed)
    results = []
    for head in heads:
        for dataset in datasets:
            folder = model_dir(run, scope, source, dataset, head)
            status = _train_or_reuse(
                root, run, scope, source, dataset, head, seed, parent_sha, keep_raw_results, resume
            )
            row = dict(dataset=dataset, head=head, status=status, path=str(folder))
            results.append(row)
            print(row, flush=True)
    return results


def evaluate_model(
    root, run, scope, source, dataset, head, *, seed=8972, keep_raw_results=False, probe=False
):
    run = Path(run).resolve()
    checkpoint, trained, _, _ = verified_model(root, run, scope, source, dataset, head, seed)
    folder = evaluation_dir(checkpoint.parent, dataset, probe=probe)
    include_distractors = dataset == "tinyface" and not probe
    affine, geometry = geometry_source(run.parents[2], dataset) if dataset != "survface" else (None, None)
    plan = dict(
        dataset=dataset,
        head=head,
        source_sha256=trained["sha256"],
        parent_sha256=trained["parent_sha256"],
        protocol_scope=scope,
        origin_only=scope == "paper_table1",
        geometry=geometry,
        include_distractors=include_distractors,
        keep_raw_results=keep_raw_results,
        data_root=str(Path(root).resolve()),
        implementation=files_for_protocol(),
    )

    def calculate():
        model = _model(load_checkpoint(str(checkpoint))["backbone_state_dict"])
        raw = folder / "raw.npz" if keep_raw_results else None
        if dataset == "survface":
            result = evaluate_survface(model, root, raw)
        else:
            result = evaluate_aligned(
                model, root, dataset, affine, raw, include_distractors=include_distractors
            )
        values = result["without_distractors"] if dataset == "scface" else result
        reference = dict(zip(METRICS[dataset], TABLE_I[head][DATASETS.index(dataset)], strict=True))
        return dict(
            metrics=result,
            comparison=_comparison(dataset, _percentages(dataset, values), reference),
            reference="UCFace Table I, VGGFace2 fine-tune, five runs",
        )

    return _record(folder, plan, calculate)


def run_probe(root, run, source, datasets, *, seed=8972, keep_raw_results=False, resume=False):
    if source == "clean_parent":
        validation = read_result(Path(run) / "parent_validation")
        _, parent = final_artifact(Path(run) / "parent")
        if validation["parent_sha256"] != parent["sha256"]:
            raise ValueError("Parent validation does not match the clean Parent")
    validate_choices(datasets, ["cosface"], "paper_table1", source)
    results = []
    for ds in datasets:
        train_models(
            root,
            run,
            "paper_table1",
            source,
            [ds],
            ["cosface"],
            seed=seed,
            keep_raw_results=keep_raw_results,
            resume=resume,
        )
        evaluation = evaluate_model(
            root,
            run,
            "paper_table1",
            source,
            ds,
            "cosface",
            seed=seed,
            keep_raw_results=keep_raw_results,
            probe=True,
        )
        results.append(evaluation)
        print(ds, {row["metric"]: row["observed_pct"] for row in evaluation["comparison"]}, flush=True)
    return results


def run_full(
    root,
    run,
    scope,
    datasets,
    heads,
    *,
    approved,
    reviewed_datasets,
    seed=8972,
    keep_raw_results=False,
    resume=False,
):
    if approved is not True:
        raise ValueError("Review the clean Parent CosFace probe and set APPROVE_EXPANSION=True")
    validate_choices(reviewed_datasets, ["cosface"], "paper_table1", "clean_parent")
    for ds in reviewed_datasets:
        _, trained, _, _ = verified_model(root, run, "paper_table1", "clean_parent", ds, "cosface", seed)
        folder = model_dir(run, "paper_table1", "clean_parent", ds, "cosface")
        evaluation = read_result(evaluation_dir(folder, ds, probe=True))
        if (
            evaluation["source_sha256"] != trained["sha256"]
            or evaluation["parent_sha256"] != trained["parent_sha256"]
            or evaluation["protocol_scope"] != "paper_table1"
            or evaluation["dataset"] != ds
            or evaluation["head"] != "cosface"
            or not compatible_completed_implementation(evaluation["implementation"])
        ):
            raise ValueError(f"Reviewed probe does not match the completed CosFace for {ds}")
    return train_models(
        root,
        run,
        scope,
        "clean_parent",
        datasets,
        heads,
        seed=seed,
        keep_raw_results=keep_raw_results,
        resume=resume,
    )


def export_model(root, run, scope, source, dataset, head, *, seed=8972):
    run = Path(run).resolve()
    checkpoint, trained, r, _ = verified_model(root, run, scope, source, dataset, head, seed)
    state = load_checkpoint(str(checkpoint))
    name = f"{dataset}_{head}_s{seed}"
    folder = checkpoint.parent / "export"
    output = folder / (name + ".pt")
    meta = dict(
        schema="facex-clean-export-v1",
        run_id=f"{run.name}_{scope}_{source}_{name}",
        dataset=dataset,
        head=head,
        seed=seed,
        stage="export",
        completed=True,
        is_smoke=False,
        feat_dim=1024,
        epochs=state["epoch"],
        parent_hash=trained["parent_sha256"],
        parent_sha256=trained["parent_sha256"],
        source_checkpoint=str(checkpoint),
        source_sha256=trained["sha256"],
        source_config=state["config"],
        source_config_hash=state["config_hash"],
        protocol_record=r["protocol_record"],
        protocol_scope=scope,
        origin_only=r["origin_only"],
        preprocessing=r["preprocessing"],
        recipe=r["data_recipe"],
    )
    if "reused_from" in trained:
        meta["reused_from"] = trained["reused_from"]
    with artifact_lock(folder):
        if not folder.exists():
            with TemporaryDirectory(dir=folder.parent) as temp:
                staging = Path(temp) / "export"
                staging.mkdir()
                if dataset != "survface":
                    affine, geometry = geometry_source(run.parents[2], dataset)
                    shutil.copyfile(affine, staging / geometry["file"])
                    meta["input_geometry"] = geometry
                CheckpointManager(str(staging)).export_backbone_only(state, output.name, meta)
                staging.rename(folder)
        if dataset != "survface":
            _, geometry = geometry_source(run.parents[2], dataset)
            meta["input_geometry"] = geometry
            checked(folder / geometry["file"], geometry["sha256"])
        sidecar = json.loads(output.with_name(output.stem + "_meta.json").read_bytes())
        if sidecar != {**meta, "sha256": compute_sha256(output), "exported_file": output.name}:
            raise ValueError("Export sidecar, checksum or source provenance differs")
        published = torch.load(output, map_location="cpu", weights_only=True)
        if (
            published["metadata"] != meta
            or any(
                not torch.equal(w, state["backbone_state_dict"][k])
                for k, w in published["state_dict"].items()
            )
            or published["state_dict"].keys() != state["backbone_state_dict"].keys()
        ):
            raise ValueError("Export tensors or embedded metadata differ")
    return dict(
        path=str(output),
        sha256=sidecar["sha256"],
        run_id=sidecar["run_id"],
        seed=seed,
        origin_only=r["origin_only"],
        source_sha256=trained["sha256"],
        parent_sha256=trained["parent_sha256"],
    )


def origin_matrix(root, run, scope, datasets, heads, *, seed=8972, keep_raw_results=False, export=False):
    validate_choices(datasets, heads, scope, "clean_parent")
    records = []
    run = Path(run)
    for dataset in datasets:
        for head in heads:
            folder = model_dir(run, scope, "clean_parent", dataset, head)
            if not (folder / "COMPLETED").is_file():
                records.append(dict(dataset=dataset, head=head, status="pending"))
                continue
            result = evaluate_model(
                root, run, scope, "clean_parent", dataset, head, seed=seed, keep_raw_results=keep_raw_results
            )
            exported = (
                export_model(root, run, scope, "clean_parent", dataset, head, seed=seed) if export else None
            )
            records.append(
                dict(
                    dataset=dataset,
                    head=head,
                    status="completed",
                    source_sha256=result["source_sha256"],
                    evaluation_sha256=compute_sha256(folder / "origin/summary.json"),
                    comparison=result["comparison"],
                    metrics=result["metrics"],
                    export=exported,
                )
            )
    plan = dict(
        protocol_scope=scope,
        datasets=list(datasets),
        heads=list(heads),
        seed=seed,
        export=export,
        keep_raw_results=keep_raw_results,
        models=[
            {k: r.get(k) for k in ("dataset", "head", "status", "evaluation_sha256", "export")}
            for r in records
        ],
    )
    from hashlib import sha256

    key = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    folder = run / "origin_matrix" / scope / key[:20]
    result = _record(
        folder,
        plan,
        lambda: dict(
            matrix_status="completed" if all(r["status"] == "completed" for r in records) else "partial",
            records=records,
        ),
    )
    exported = [dict(dataset=r["dataset"], head=r["head"], **r["export"]) for r in records if r.get("export")]
    if exported:
        path = folder / "inventory.json"
        with artifact_lock(path):
            write_json(
                path,
                dict(
                    schema="facex-clean-inventory-v1",
                    protocol_scope=scope,
                    origin_only=scope == "paper_table1",
                    matrix_status=result["matrix_status"],
                    models=exported,
                    report_sha256=compute_sha256(folder / "summary.json"),
                ),
                immutable=True,
            )
        result = {**result, "inventory": str(path)}
    return {**result, "report_dir": str(folder)}
