"""Fresh VGGFace2 parent and SurvFace MagFace run, with explicit epoch-boundary recovery."""

import json
import math
import random
import re
import shutil
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import torch
import cv2
from torch import nn

from .alignment_resume import recovery_lock
from .baseline_policy import DEFAULT_AUGMENTATION, code_fingerprints
from .checkpoint_manager import (
    CheckpointManager,
    compute_config_hash,
    compute_sha256,
    load_checkpoint,
    verify_parent_checkpoint,
)
from .clean_benchmarks import (
    DATA,
    MILESTONES,
    checked,
    evaluate_survface,
    monitor,
    prepare_panels,
    reference_gaps,
)
from .dataset import build_dataloader
from .clean_protocols import HEAD_OPTIONS, LEGACY_PARENT, LEGACY_SHA, LOCAL_VARIANTS, downstream
from .model_builder import build_model
from .notebook_workflow import write_json
from .shared_storage import artifact_lock
from .survface_input_pilot import PilotMagFace
from .trainer import set_seed


# Completed refacto2 artifacts from before the scope reuse and report correction.
_PRE_SCOPE_REUSE = {
    "clean_training": "0c01d470c4677962bbd6d5357e30f413d11043f6b43342d789ea7e02b90165dd",
    "clean_campaign": "d96092ded63c47f834264b7443f4a93a3d98920ef8962f3b97d12488de743cc8",
    "clean_protocols": "ef40b55752ec3a52366102259d9417cd9720f54325b12d1cc2c1514077cc702f",
}


def matrix_recipe(run, root, dataset, head, scope):
    r = recipe("finetune")
    for key in HEAD_OPTIONS["magface"]:
        if key != "scale":
            r.pop(key)
    return {
        **r,
        **downstream(root, run.parents[2], dataset, scope),
        **HEAD_OPTIONS[head],
        "head": head,
        "local_head_variant": LOCAL_VARIANTS.get(head),
        "protocol_scope": scope,
        "origin_only": scope == "paper_table1",
    }


def recipe(stage):
    if stage == "parent":
        return dict(
            dataset="VGGFace2",
            prepared="VGGFace2/processed/shared_v1_boundary_mobilefacenet",
            train_hash="b99ac7567d262346e601aa57b91aeeaa3ef0cf02c61ec95434327781d45a34d4",
            manifest_hash="4d5fbaf67b0752f590e88af381f7ba9d538e422d22630d404c4a378d9da98ead",
            images=3141890,
            classes=8631,
            batch=256,
            epochs=24,
            dropout=0.0,
            optimizer="SGD",
            head="ArcFace",
            m=0.5,
            scale=64.0,
            lr=0.1,
            warmup_start=0.01,
            momentum=0.9,
            conv_linear_decay=0.0001,
            other_decay=0.0,
            milestones=[10, 18, 22],
            gamma=0.1,
            augmentation={**DEFAULT_AUGMENTATION, "affine_probability": 0.0},
        )
    if stage == "finetune":
        return dict(
            dataset="QMUL-SurvFace",
            prepared="QMUL-SurvFace/processed/paper_baseline_v1_resize_only",
            train_hash="5bb88f70d3981c2d4c8e4d6ba90c75ddc8d5fc775e0c0b189b397f12a863ece5",
            manifest_hash="af7a7954c5cd7474ee55d998c8d55fb3412a4d5a915c994bf35230fea11e4ab5",
            images=220888,
            classes=5319,
            batch=64,
            epochs=40,
            dropout=0.6,
            optimizer="Adam",
            head="MagFace",
            scale=64.0,
            l_a=10.0,
            u_a=110.0,
            l_margin=0.45,
            u_margin=0.8,
            lamda=20.0,
            detach_margin_norm=True,
            official_monotonic_fallback=True,
            warmup_epochs=3,
            head_lr=0.0002,
            bn_lr=0.0001,
            backbone_lr=0.0001,
            weight_decay=0.0,
            milestones=[10, 20, 30],
            gamma=0.1,
            augmentation=DEFAULT_AUGMENTATION.copy(),
        )
    raise ValueError(f"Unknown stage {stage}")


def input_list(root, r):
    folder = root / r["prepared"]
    report = json.loads((folder / "report.json").read_bytes())
    if not (
        (folder / "COMPLETED").is_file()
        and not folder.with_name(folder.name + ".lock").exists()
        and report["completed"]
        and report["failed"] == 0
        and report["prepared"] == r["images"]
        and report["train_list_sha256"] == r["train_hash"]
    ):
        raise ValueError(f"Incomplete prepared dataset: {folder}")
    checked(folder / "manifest.jsonl", r["manifest_hash"])
    return checked(folder / "train_list.txt", r["train_hash"])


def make_model(stage, r, parent_state=None):
    if (stage == "finetune") != (parent_state is not None):
        raise ValueError("Only fine-tuning accepts and requires completed parent weights")
    backbone = dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=r["dropout"])
    head = dict(type=r["head"], num_class=r["classes"], scale=r["scale"])
    if stage == "parent":
        head.update(margin_arc=r["m"], margin_am=r["m"] * math.sin(r["m"]))
    else:
        head.update({k: r[k] for k in HEAD_OPTIONS[r["head"].lower()] if k in r})
    model = build_model(backbone, head)
    if parent_state is not None:
        model.backbone.load_state_dict(parent_state, strict=True)
    if stage == "finetune" and r["head"].lower() == "magface":
        monotonic_head = PilotMagFace(
            1024, r["classes"], **{k: r[k] for k in ("scale", "l_a", "u_a", "l_margin", "u_margin", "lamda")}
        )
        monotonic_head.load_state_dict(model.head.state_dict(), strict=True)
        model.head = monotonic_head
    model.backbone.bn.bias.requires_grad_(False).zero_()
    return model.cuda()


def optimizer_for(model, stage, r):
    if stage == "parent":
        decay = [m.weight for m in model.backbone.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        ids = {id(p) for p in decay}
        other = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
        return torch.optim.SGD(
            [
                dict(params=decay, weight_decay=r["conv_linear_decay"]),
                dict(params=other, weight_decay=r["other_decay"]),
            ],
            lr=r["lr"],
            momentum=r["momentum"],
        )
    bn = [p for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()]
    bn.append(model.backbone.bn.weight)
    excluded = {id(p) for p in bn + [model.backbone.bn.bias]}
    features = [p for p in model.backbone.parameters() if id(p) not in excluded]
    return torch.optim.Adam(
        [
            dict(params=model.head.parameters(), lr=r["head_lr"]),
            dict(params=bn, lr=r["bn_lr"]),
            dict(params=features, lr=r["backbone_lr"]),
        ],
        weight_decay=r["weight_decay"],
    )


def configure_epoch(model, opt, stage, r, epoch):
    model.train()
    rates = [r["lr"]] * 2 if stage == "parent" else [r["head_lr"], r["bn_lr"], r["backbone_lr"]]
    decay = r["gamma"] ** sum(epoch > m for m in r["milestones"])
    for group, base in zip(opt.param_groups, rates, strict=True):
        group["lr"] = base * decay
    if stage == "finetune":
        warm = epoch <= r["warmup_epochs"]
        for p in opt.param_groups[2]["params"]:
            p.requires_grad_(not warm)
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train(warm).requires_grad_(warm)


def train_epoch(model, opt, stage, r, loader, epoch):
    configure_epoch(model, opt, stage, r, epoch)
    loader.batch_sampler.set_epoch(epoch, 0)
    total, count = 0.0, 0
    for i, (xs, ys) in enumerate(loader):
        if stage == "parent" and epoch == 1:
            lr = r["warmup_start"] + (r["lr"] - r["warmup_start"]) * i / max(1, len(loader) - 1)
            for group in opt.param_groups:
                group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        ys = ys.cuda(non_blocking=True)
        logits, reg = model(xs.cuda(non_blocking=True), ys)
        loss = nn.functional.cross_entropy(logits, ys)
        if reg is not None:
            loss = loss + reg.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {i + 1}")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.nn.utils.get_total_norm(grads, float("inf")).isfinite():
            raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}, batch {i + 1}")
        opt.step()
        total += float(loss.detach()) * len(ys)
        count += len(ys)
        if (i + 1) % 500 == 0:
            print(f"{stage} ep {epoch} batch {i + 1}/{len(loader)}, loss {total / count:.4f}", flush=True)
    if count != len(loader.dataset):
        raise ValueError("Incomplete training epoch")
    return total / count


def files_for_protocol():
    files = code_fingerprints()
    for name in (
        "clean_training",
        "clean_benchmarks",
        "clean_protocols",
        "clean_campaign",
        "control_geometry",
        "survface_input_pilot",
        "alignment_resume",
    ):
        path = Path(__file__).with_name(name + ".py")
        files[name] = compute_sha256(str(path))
    return files


def compatible_completed_implementation(recorded):
    current = files_for_protocol()
    return recorded == current or recorded == {**current, **_PRE_SCOPE_REUSE}


def receipt_checkpoint(folder, receipt):
    name = receipt["checkpoint"]
    if not re.fullmatch(r"epoch_\d{2}_[a-f0-9]{12}\.pt", name):
        raise ValueError("Invalid clean checkpoint name")
    return checked(folder / name, receipt["sha256"])


def final_artifact(stage_dir, *, expected_recipe=None):
    """Verify final receipt, checkpoint and fixed diagnostic schedule before handoff."""
    folder = Path(stage_dir)
    report = json.loads(
        checked(folder / "summary.json", (folder / "COMPLETED").read_text().strip()).read_bytes()
    )
    contract = json.loads(checked(folder / "plan.json", report["plan_sha256"]).read_bytes())
    r = expected_recipe if expected_recipe is not None else recipe(contract["stage"])
    if (
        (expected_recipe is None and contract["stage"] != folder.name)
        or contract["recipe"] != r
        or report["status"] != "completed"
        or len(report["epochs"]) != r["epochs"]
        or report["epoch"] != r["epochs"]
        or [row["epoch"] for row in report["epochs"]] != list(range(1, r["epochs"] + 1))
        or report["parent_sha256"] != contract["parent_sha256"]
        or report.get("reused_from") != contract.get("reused_from")
    ):
        raise ValueError("Incomplete clean training stage")
    if contract["stage"] == "parent" and [
        row["epoch"] for row in report["epochs"] if "zero_shot" in row
    ] != list(MILESTONES):
        raise ValueError("Missing parent milestone")
    return receipt_checkpoint(folder, report), report


def run_directory(model_root, run_id):
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id
    ):
        raise ValueError("Use letters, digits, underscore or hyphen in RUN_ID")
    return Path(model_root).resolve() / "facex/clean_ucface" / run_id


def publish_summary(folder, summary):
    guide = Path(__file__).resolve().parents[1] / "notebooks/clean/HOW_TO_READ.md"
    shutil.copyfile(guide, folder / summary["guide"])
    write_json(folder / "summary.json", summary, immutable=True)
    names = ["summary.json", summary["guide"], "plan.json"]
    sizes = {name: (folder / name).stat().st_size for name in names}
    write_json(
        folder / "bundle.json",
        dict(
            recommended_order=names,
            bytes_by_file=sizes,
            total_bytes=sum(sizes.values()),
            estimated_tokens=sum(len((folder / n).read_text(encoding="utf-8")) for n in names),
            estimate_basis="one token per character for upload planning",
            sha256={name: compute_sha256(folder / name) for name in names},
        ),
        immutable=True,
    )
    temporary = folder / "COMPLETED.tmp"
    temporary.write_text(compute_sha256(folder / "summary.json"), encoding="ascii")
    temporary.replace(folder / "COMPLETED")


def checkpoint_config(stage, r, train_list, *, matrix):
    config = dict(
        stage="pretrain" if stage == "parent" else "finetune",
        backbone=dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=r["dropout"]),
        head=dict(
            type=r["head"],
            num_class=r["classes"],
            **{k: r[k] for k in HEAD_OPTIONS[r["head"].lower()] if k in r},
        ),
        training=r,
        dataset=dict(name=r["dataset"], train_list=str(train_list)),
    )
    if matrix:
        config["preprocessing"] = r["preprocessing"]
    return config


def publish_scope_reuse(
    folder, target_recipe, source_checkpoint, source_summary, source_plan, keep_raw_results
):
    """Publish a scope-local receipt and an independent copy of the verified final weights."""
    original = source_plan["recipe"]
    scope_fields = {"protocol_scope", "origin_only"}
    if (
        source_plan["stage"] != "finetune"
        or source_plan["initial_weights"] != "clean_parent"
        or original["dataset"] not in ("tinyface", "scface")
        or original["protocol_scope"] == target_recipe["protocol_scope"]
        or {k: v for k, v in original.items() if k not in scope_fields}
        != {k: v for k, v in target_recipe.items() if k not in scope_fields}
        or source_summary.get("reused_from") is not None
    ):
        raise ValueError("Scopes do not share the same completed training contract")

    with artifact_lock(folder):
        if folder.exists():
            raise FileExistsError(f"Preserve existing scope artifacts before publishing reuse: {folder}")
        with TemporaryDirectory(dir=folder.parent) as temp:
            staging = Path(temp) / "model"
            staging.mkdir()
            copied = staging / source_summary["checkpoint"]
            shutil.copyfile(source_checkpoint, copied)
            checked(copied, source_summary["sha256"])
            state = load_checkpoint(str(copied))
            config = checkpoint_config("finetune", original, source_plan["train_list"], matrix=True)
            if (
                state["config"] != config
                or state["config_hash"] != compute_config_hash(config)
                or state["stage"] != "finetune"
                or state["completed"] is not True
                or state["epoch"] != original["epochs"]
                or state["seed"] != source_plan["seed"]
                or state["parent_hash"] != source_plan["parent_sha256"]
                or state["extra_meta"]["plan_sha256"] != source_summary["plan_sha256"]
            ):
                raise ValueError("Source checkpoint does not match the completed training contract")
            del state

            lineage = dict(
                protocol_scope=original["protocol_scope"],
                checkpoint=str(source_checkpoint),
                checkpoint_sha256=source_summary["sha256"],
                plan_sha256=source_summary["plan_sha256"],
                summary_sha256=compute_sha256(source_checkpoint.parent / "summary.json"),
                keep_raw_results=source_plan["keep_raw_results"],
            )
            plan = dict(
                source_plan, recipe=target_recipe, keep_raw_results=keep_raw_results, reused_from=lineage
            )
            plan_path = Path(write_json(staging / "plan.json", plan, immutable=True))
            summary = dict(
                source_summary,
                plan_sha256=compute_sha256(plan_path),
                protocol_scope=target_recipe["protocol_scope"],
                origin_only=target_recipe["origin_only"],
                keep_raw_results=keep_raw_results,
                reused_from=lineage,
            )
            publish_summary(staging, summary)
            final_artifact(staging, expected_recipe=target_recipe)
            staging.rename(folder)


def load_parent(run, source, seed):
    if source == "legacy_parent_wd0":
        path, digest, epoch, parent_seed = run.parents[2] / LEGACY_PARENT, LEGACY_SHA, 40, 8972
    elif source == "clean_parent":
        path, report = final_artifact(run / "parent")
        digest, epoch, parent_seed = report["sha256"], recipe("parent")["epochs"], seed
    else:
        raise ValueError(f"Unknown parent source: {source}")
    return verify_parent_checkpoint(
        str(path), expected_hash=digest, expected_seed=parent_seed, target_epoch=epoch, require_completed=True
    )


def run_stage(
    data_root,
    run_dir,
    stage,
    *,
    seed=8972,
    keep_raw_results=False,
    resume=False,
    matrix=None,
    parent_source="clean_parent",
):
    if not torch.cuda.is_available():
        raise RuntimeError("Clean training requires CUDA")
    torch.set_num_threads(2)
    root, run = Path(data_root).resolve(), Path(run_dir).resolve()
    if matrix is None:
        folder, r = run / stage, recipe(stage)
        train_list = input_list(root, r)
    else:
        folder, r = matrix
        train_list = Path(r["prepared"]) / "train_list.txt"
    parent_hash, parent_state = None, None
    if stage == "finetune":
        payload, parent_hash = load_parent(run, parent_source, seed)
        parent_state = payload["backbone_state_dict"]
        del payload
    lock = recovery_lock if resume else artifact_lock
    with lock(folder):
        if (folder / "COMPLETED").exists() or (folder.exists() and not resume):
            raise FileExistsError(
                f"Existing stage. Read its summary or explicitly resume a partial run: {folder}"
            )
        if resume and not (folder / "plan.json").is_file() and folder.exists() and any(folder.iterdir()):
            raise ValueError("No stage plan to validate the existing results")
        folder.mkdir(parents=True, exist_ok=True)
        panels, panel_info = prepare_panels(root) if stage == "parent" else (None, {})
        plan = dict(
            stage=stage,
            run_dir=str(run),
            seed=seed,
            recipe=r,
            train_list=str(train_list),
            parent_sha256=parent_hash,
            initial_weights="random" if stage == "parent" else parent_source,
            keep_raw_results=keep_raw_results,
            milestones=list(MILESTONES) if panels else [],
            monitor_inputs={k: v for k, v in panel_info.items() if k != "prepare_seconds"},
            implementation=files_for_protocol(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            opencv=cv2.__version__,
            numpy=np.__version__,
        )
        write_json(folder / "plan.json", plan, immutable=True)
        plan_hash = compute_sha256(folder / "plan.json")
        set_seed(seed)
        model = make_model(stage, r, parent_state)
        opt = optimizer_for(model, stage, r)
        loader = build_dataloader(
            str(train_list.parent),
            str(train_list),
            batch_size=r["batch"],
            num_workers=4,
            cache_dir=str(run.parent / "input_index"),
            seed=seed,
            prefetch_factor=4,
            augmentation=r["augmentation"],
        )
        if len(loader.dataset) != r["images"]:
            raise ValueError("Train list count mismatch")
        progress = folder / "progress.json"
        rows, previous = [], None
        preparation_seconds = panel_info.get("prepare_seconds")
        if resume and progress.exists():
            saved = json.loads(progress.read_bytes())
            preparation_seconds = saved["panel_prepare_seconds"]
            if saved["plan_sha256"] != plan_hash:
                raise ValueError("Resume contract mismatch")
            previous = receipt_checkpoint(folder, saved)
            state = load_checkpoint(str(previous))
            rows = saved["epochs"]
            if (
                [row["epoch"] for row in rows] != list(range(1, state["epoch"] + 1))
                or state["extra_meta"]["plan_sha256"] != plan_hash
                or state["parent_hash"] != parent_hash
            ):
                raise ValueError("Invalid epoch-boundary checkpoint")
            model.backbone.load_state_dict(state["backbone_state_dict"], strict=True)
            model.head.load_state_dict(state["head_state_dict"], strict=True)
            opt.load_state_dict(state["optimizer_state_dict"])
            rng = state["rng_state"]
            torch.set_rng_state(rng["torch_rng"])
            torch.cuda.set_rng_state_all(rng["cuda_rng"])
            np.random.set_state(rng["numpy_rng"])
            random.setstate(rng["python_rng"])
            del state
        manager = CheckpointManager(str(folder))
        config = checkpoint_config(stage, r, train_list, matrix=matrix is not None)
        try:
            for epoch in range(len(rows) + 1, r["epochs"] + 1):
                loss = train_epoch(model, opt, stage, r, loader, epoch)
                row = dict(epoch=epoch, loss=loss, ending_lrs=[g["lr"] for g in opt.param_groups])
                if stage == "finetune":
                    with torch.no_grad():
                        directions = nn.functional.normalize(model.head.weight, dim=0)
                        row["head_direction_norm"] = float(directions.mean(dim=1).norm())
                        row["output_bn_weight_l2"] = float(model.backbone.bn.weight.norm())
                    print(
                        f"{r['dataset']}/{r['head']} epoch {epoch}, loss {loss:.4f}, "
                        f"head direction {row['head_direction_norm']:.4f}",
                        flush=True,
                    )
                attempt = f"epoch_{epoch:02d}_{uuid4().hex[:12]}"
                if stage == "parent" and epoch in MILESTONES:
                    raw = folder / (attempt + "_vectors.npz") if keep_raw_results else None
                    row["zero_shot"] = monitor(model.backbone, panels, raw_file=raw)
                    row["reference"] = reference_gaps(row["zero_shot"]["metrics"], epoch)
                    if raw is not None:
                        row["vectors"] = raw.name
                    m = row["zero_shot"]["metrics"]
                    print(
                        f"Parent epoch {epoch} | LFW {100 * m['lfw']['accuracy']:.2f}% | "
                        f"SCFace {100 * m['scface']['rank1']:.2f}% | TinyFace {100 * m['tinyface']['rank1']:.2f}% | "
                        f"{row['zero_shot']['seconds']:.1f}s",
                        row["reference"],
                        flush=True,
                    )
                    print("SCFace by distance", m["scface"].get("distances", {}), flush=True)
                current = Path(
                    manager.save_checkpoint(
                        attempt + ".pt",
                        model,
                        opt,
                        None,
                        epoch,
                        extra_meta=dict(
                            completed=epoch == r["epochs"],
                            seed=seed,
                            stage=config["stage"],
                            head=r["head"],
                            parent_hash=parent_hash,
                            total_epochs=r["epochs"],
                            plan_sha256=plan_hash,
                            config=config,
                        ),
                    )
                )
                rows.append(row)
                saved = dict(
                    panel_prepare_seconds=preparation_seconds,
                    plan_sha256=plan_hash,
                    epoch=epoch,
                    checkpoint=current.name,
                    sha256=compute_sha256(current),
                    epochs=rows,
                )
                write_json(progress, saved)
                if previous and not keep_raw_results:
                    previous.unlink(missing_ok=True)
                previous = current
            summary = dict(
                **saved,
                status="completed",
                parent_sha256=parent_hash,
                keep_raw_results=keep_raw_results,
                guide="HOW_TO_READ.md",
            )
            if matrix is not None:
                summary.update(
                    protocol_scope=r["protocol_scope"],
                    origin_only=r["origin_only"],
                    local_head_variant=r["local_head_variant"],
                    dataset=r["dataset"],
                    head=r["head"],
                )
            publish_summary(folder, summary)
            return summary
        except BaseException:
            (folder / "last_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise


def final_evaluation(data_root, run_dir, *, keep_raw_results=False):
    run = Path(run_dir)
    checkpoint, trained = final_artifact(run / "finetune")
    _, parent = final_artifact(run / "parent")
    if parent["sha256"] != trained["parent_sha256"]:
        raise ValueError("Fine-tune and parent lineage differ")
    folder = run / "origin"
    with artifact_lock(folder):
        if (folder / "COMPLETED").exists():
            report = json.loads(
                checked(folder / "summary.json", (folder / "COMPLETED").read_text().strip()).read_bytes()
            )
            checked(folder / "plan.json", report["plan_sha256"])
            if (
                report["sha256"] != trained["sha256"]
                or report["keep_raw_results"] != keep_raw_results
                or report["data_root"] != str(Path(data_root).resolve())
            ):
                raise ValueError("Final evaluation request differs from completed report")
            return report
        if not torch.cuda.is_available():
            raise RuntimeError("Final evaluation requires CUDA")
        folder.mkdir(parents=True, exist_ok=True)
        from .backbone_factory import build_backbone

        model = build_backbone(dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=0.6))
        state = load_checkpoint(str(checkpoint))
        model.load_state_dict(state["backbone_state_dict"], strict=True)
        del state
        model.cuda().eval()
        plan = dict(
            sha256=trained["sha256"],
            parent_sha256=parent["sha256"],
            data_root=str(Path(data_root).resolve()),
            role_sha256=DATA["survface"][2],
            keep_raw_results=keep_raw_results,
            implementation=files_for_protocol(),
        )
        write_json(folder / "plan.json", plan, immutable=True)
        points = evaluate_survface(model, data_root, folder / "queries.npz" if keep_raw_results else None)
        summary = dict(
            status="completed",
            **plan,
            plan_sha256=compute_sha256(folder / "plan.json"),
            roc=points,
            guide="HOW_TO_READ.md",
            gap_to_table_i_pp=100 * points["0.10"]["tpir_at_20"] - 24.80,
            scope="Single-run Origin test ROC; target FPIR 10%; five-run paper reference",
        )
        publish_summary(folder, summary)
        return summary
