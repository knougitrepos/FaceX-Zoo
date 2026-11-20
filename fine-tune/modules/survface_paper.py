"""SurvFace full-training (5,319 IDs / 220,888 images) paper baseline protocol, preprocessing, and checkpoint diagnostics.

Separates the UCFace Table I / Supplement Table IX full-FR baseline from the
4,255-ID FR-excluded calibration protocol while keeping existing checkpoints
and source-fingerprinted modules immutable.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .alignment import alignment_policy, read_rows
from .checkpoint_manager import compute_sha256, load_checkpoint
from .control_geometry import TARGET, read_bgr, similarity, tiny_landmarks
from .dataset import resolve_image_path
from .shared_storage import artifact_lock, resolve_protocol, write_derivation

PAPER_SURVFACE_CLASSES = 5319
PAPER_SURVFACE_IMAGES = 220888
PAPER_PROTOCOL_POLICY = "paper_full_fr_training_v1"
PAPER_PFLD_VERSION = "survface_pfld_five_point_arc112_v1"
PFLD_SHA256 = "5cb8410672e89ab87047114ba2dd4491d0ef82c412d4952d66d7f98e23c578f7"

UCFACE_PARENT_TABLE_I = {
    "parent_active_dims": 79,
    "parent_total_dims": 1024,
    "finding": (
        "VGGFace2 parent checkpoint has 79/1024 active dimensions; on SurvFace resize_only, "
        "parent Origin test ROC TPIR@20 at FPIR=10% is 1.42% (0.04% unknown cos>=0.95), "
        "whereas 4,255-ID fine-tuning collapsed to 2.58% (ArcFace, 10.37% unknown cos>=0.95) "
        "or 0.23%–0.42% (CosFace/MagFace/AdaFace, 8.48%–10.02% unknown cos>=0.95 due to 4–7 BN spike dims)."
    ),
    "parent_survface_origin": {
        "10%": {"tpir20": 1.42, "unknown_cos_ge_095_pct": 0.04},
        "20%": {"tpir20": 2.76, "unknown_cos_ge_095_pct": 0.04},
    },
    "finetuned_4255_resize_only": {
        "arcface": {"tpir20_at_fpir10": 0.20, "test_roc_tpir20_at_fpir10": 2.58, "unknown_cos_ge_095_pct": 10.37},
        "cosface": {"tpir20_at_fpir10": 0.42, "test_roc_tpir20_at_fpir10": 0.42, "unknown_cos_ge_095_pct": 10.02},
        "magface": {"tpir20_at_fpir10": 0.23, "test_roc_tpir20_at_fpir10": 0.23, "unknown_cos_ge_095_pct": 8.48},
        "adaface": {"tpir20_at_fpir10": 0.29, "test_roc_tpir20_at_fpir10": 0.29, "unknown_cos_ge_095_pct": 9.95},
    },
    "tinyface_no_distractors_rank1_pct": {"paper_mean": 55.58, "paper_se": 0.19, "local_whole": 30.74, "local_pfld_aligned": 53.57},
    "tinyface_with_distractors_rank1_pct": {"paper_mean": 51.10, "paper_se": 0.09},
    "scface_avg_rank1_pct": {"paper_mean": 77.08, "paper_se": 0.42, "local_whole": 9.17, "local_two_eye_aligned": 65.75},
    "survface_tpir20_at_fpir01_pct": {"paper_mean": 1.12, "paper_se": 0.21, "local_resize_only": 1.42, "local_pfld_aligned": 0.79},
    "survface_tpir20_at_fpir02_pct": {"paper_mean": 5.01, "paper_se": 0.33, "local_resize_only": 2.76, "local_pfld_aligned": 1.83},
}


def _default_dataset_root() -> Path:
    return Path(os.environ.get("FACE_DATASET_ROOT", "F:/face_datasets" if os.name == "nt" else "/data/face_datasets")).expanduser().resolve()


def paper_protocol_dir(dataset_root: str | Path | None = None) -> Path:
    root = Path(dataset_root).expanduser().resolve() if dataset_root else _default_dataset_root()
    return root / "QMUL-SurvFace/protocols/paper_baseline_v1/5319_full_fr_v1"


def paper_handoff_path(dataset_root: str | Path | None = None) -> Path:
    root = Path(dataset_root).expanduser().resolve() if dataset_root else _default_dataset_root()
    return root / "protocols/paper_baseline_v1/handoff.json"


def paper_processed_dir(dataset_root: str | Path | None = None, recipe: str = "full_fr_resize_only") -> Path:
    root = Path(dataset_root).expanduser().resolve() if dataset_root else _default_dataset_root()
    if recipe == "full_fr_resize_only":
        return root / "QMUL-SurvFace/processed/paper_baseline_v1_resize_only"
    if recipe == "full_fr_pfld_five_point_alignment":
        return root / "QMUL-SurvFace/processed/paper_baseline_v1_pfld_aligned"
    raise ValueError(f"Unsupported SurvFace paper recipe: {recipe}")


def _write_immutable_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Immutable paper artifact differs: {path}")
        return
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def prepare_survface_paper_protocol(dataset_root: str | Path | None = None) -> dict:
    """Create the immutable 5,319-ID / 220,888-image SurvFace protocol and v2 handoff."""
    root = Path(dataset_root).expanduser().resolve() if dataset_root else _default_dataset_root()
    shared_handoff = root / "protocols/shared_v1/handoff.json"
    if not shared_handoff.is_file():
        raise FileNotFoundError(f"Shared v2 handoff is required at {shared_handoff}")
    shared_index = json.loads(shared_handoff.read_text(encoding="utf-8"))
    if shared_index.get("schema") != "facex-ronbun-handoff-v2":
        raise ValueError("Expected facex-ronbun-handoff-v2")
    surv_item = shared_index["protocols"]["survface"]
    surv_proto_path = (shared_handoff.parent / surv_item["path"]).resolve()
    if compute_sha256(str(surv_proto_path)) != surv_item["sha256"]:
        raise ValueError("Shared SurvFace protocol hash mismatch")
    base_record = resolve_protocol(surv_proto_path)
    if compute_sha256(base_record["roles_path"]) != base_record["roles_sha256"]:
        raise ValueError("Shared SurvFace roles.csv hash mismatch")

    out_dir = paper_protocol_dir(root)
    handoff_out = paper_handoff_path(root)
    with artifact_lock(out_dir):
        with Path(base_record["roles_path"]).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        train_rows, test_ids, train_ids = [], set(), set()
        for row in rows:
            if row["role"].startswith("test_"):
                row["fr_train"] = "0"
                if row["source_identity_id"]:
                    test_ids.add(row["source_identity_id"])
            else:
                row["fr_train"] = "1"
                train_rows.append(row)
                train_ids.add(row["source_identity_id"])
        if train_ids & test_ids:
            raise ValueError("Native SurvFace training and test identities overlap")
        if len(train_ids) != PAPER_SURVFACE_CLASSES or len(train_rows) != PAPER_SURVFACE_IMAGES:
            raise ValueError(
                f"SurvFace paper training counts must be {PAPER_SURVFACE_CLASSES} IDs / "
                f"{PAPER_SURVFACE_IMAGES} images; got {len(train_ids)} / {len(train_rows)}"
            )
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        roles_content = buf.getvalue()
        roles_sha = hashlib.sha256(roles_content.encode("utf-8")).hexdigest()
        roles_path = out_dir / "roles.csv"
        _write_immutable_text(roles_path, roles_content)

        labels = {cid: idx for idx, cid in enumerate(sorted({r["identity_id"] for r in train_rows}))}
        train_list_content = "".join(f"{r['image_path']} {labels[r['identity_id']]}\n" for r in train_rows)
        train_list_path = out_dir / "train_list.txt"
        _write_immutable_text(train_list_path, train_list_content)
        train_sha = compute_sha256(str(train_list_path))

        by_role = {
            role: dict(
                images=sum(r["role"] == role for r in rows),
                identities=len({r["identity_id"] for r in rows if r["role"] == role and r["identity_id"]}),
            )
            for role in sorted({r["role"] for r in rows})
        }
        sorted_native_ids = sorted(train_ids)
        train_id_sha = hashlib.sha256("\n".join(sorted_native_ids).encode("utf-8")).hexdigest()
        raw_root = Path(base_record["raw_root"]).resolve()
        rel_raw_root = os.path.relpath(raw_root, out_dir).replace("\\", "/")
        rel_source_roles = os.path.relpath(Path(base_record["roles_path"]).resolve(), out_dir).replace("\\", "/")

        protocol_payload = dict(
            schema="facex-ronbun-roles-v2",
            name="survface",
            policy=PAPER_PROTOCOL_POLICY,
            split_seed=int(base_record.get("split_seed", 8972)),
            raw_root=rel_raw_root,
            roles_path="roles.csv",
            roles_sha256=roles_sha,
            train_list="train_list.txt",
            train_sha256=train_sha,
            num_classes=PAPER_SURVFACE_CLASSES,
            images=PAPER_SURVFACE_IMAGES,
            identity_disjoint=True,
            calibration_unseen_in_fr=False,
            calibration_gallery_fr_seen=True,
            parent_identity_overlap="not_audited",
            source="UCFace Supplement II-A / Table IX full 5,319-ID training set",
            all_images_decode=True,
            role_counts=by_role,
            sources={
                "shared_v1_roles": {
                    "path": rel_source_roles,
                    "sha256": base_record["roles_sha256"],
                }
            },
            training_identity_ids=sorted_native_ids,
            training_identity_sha256=train_id_sha,
        )
        protocol_path = out_dir / "protocol.json"
        _write_immutable_text(
            protocol_path,
            json.dumps(protocol_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        protocol_sha = compute_sha256(str(protocol_path))

        handoff_protocols = {}
        for ds_name, item in shared_index["protocols"].items():
            if ds_name == "survface":
                rel_p = os.path.relpath(protocol_path, handoff_out.parent).replace("\\", "/")
                handoff_protocols[ds_name] = {"path": rel_p, "sha256": protocol_sha}
            else:
                target_p = (shared_handoff.parent / item["path"]).resolve()
                rel_p = os.path.relpath(target_p, handoff_out.parent).replace("\\", "/")
                handoff_protocols[ds_name] = {"path": rel_p, "sha256": item["sha256"]}
        handoff_payload = dict(schema="facex-ronbun-handoff-v2", protocols=handoff_protocols)
        _write_immutable_text(
            handoff_out,
            json.dumps(handoff_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
    return resolve_protocol(protocol_path)


def paper_survface_profile(base_profile: dict, dataset_root: str | Path | None = None, recipe: str = "full_fr_resize_only") -> dict:
    """Return a SurvFace finetune_profile bound to the 5,319-ID paper protocol."""
    root = Path(dataset_root or Path(base_profile["raw_root"]).parent).expanduser().resolve()
    proto_dir = paper_protocol_dir(root)
    proto_path = proto_dir / "protocol.json"
    proc_dir = paper_processed_dir(root, recipe)
    profile = dict(base_profile)
    profile.update(
        name="survface",
        raw_root=str(root / "QMUL-SurvFace"),
        num_classes=PAPER_SURVFACE_CLASSES,
        expected_images=PAPER_SURVFACE_IMAGES,
        train_list=str(proto_dir / "train_list.txt"),
        val_list=None,
        protocol_record=str(proto_path),
        training_manifest=str(proto_dir / "roles.csv"),
        evaluation_manifest=str(proto_dir / "roles.csv"),
        enabled=True,
        local_audit_required=False,
        alignment_report=str(proc_dir / "report.json"),
        split_policy="facex_ronbun_v1",
        split_seed=8972,
        research_output=str(proto_dir.parent),
        portable_protocol=True,
        input_size=[112, 112],
        pixel_mean=127.5,
        pixel_std=128.0,
    )
    if proto_path.is_file():
        rec = json.loads(proto_path.read_text(encoding="utf-8"))
        profile.update(
            shared_protocol_sha256=compute_sha256(str(proto_path)),
            role_manifest_sha256=rec["roles_sha256"],
            training_identity_sha256=rec["training_identity_sha256"],
        )
    return profile


def pfld_survface_policy(base_dir: str | Path) -> dict:
    base_dir = Path(base_dir).resolve()
    pfld_path = base_dir / "face_sdk/models/face_alignment/face_alignment_1.0/face_landmark_pfld.pkl"
    if compute_sha256(str(pfld_path)) != PFLD_SHA256:
        raise ValueError(f"PFLD weight hash mismatch: {pfld_path}")
    return {
        "version": PAPER_PFLD_VERSION,
        "method": "direct_similarity_transform",
        "landmark_source": str(pfld_path),
        "landmark_sha256": PFLD_SHA256,
        "target_points": TARGET.tolist(),
        "input_size": [112, 112],
        "color": "BGR",
        "output_format": "png",
        "interpolation": "bicubic",
        "border": "zero",
        "face_detection": "none",
        "failure_policy": "record_and_block",
        "adapter_sha256": compute_sha256(__file__),
        "opencv": cv2.__version__,
        "research_only": True,
    }


def _encode_resize_row(args):
    idx, source_path, label, out_dir = args
    data = source_path.read_bytes()
    src_sha = hashlib.sha256(data).hexdigest()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {source_path}")
    src_size = list(image.shape[:2])
    crop = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
    target = out_dir / "images" / f"{idx // 1000:06d}" / f"{idx:08d}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", crop)
    if not ok or crop.shape != (112, 112, 3):
        raise ValueError(f"PNG encode failed for {source_path}")
    png_bytes = encoded.tobytes()
    target.write_bytes(png_bytes)
    out_sha = hashlib.sha256(png_bytes).hexdigest()
    record = {
        "index": idx,
        "source": str(source_path),
        "label": label,
        "source_sha256": src_sha,
        "status": "prepared",
        "output": str(target),
        "output_sha256": out_sha,
        "transform": "resize_only",
        "source_size": src_size,
    }
    return record, f"{target.as_posix()} {label}\n"


def prepare_survface_paper_inputs(
    base_dir: str | Path,
    dataset_root: str | Path | None = None,
    recipe: str = "full_fr_resize_only",
    max_images: int | None = None,
) -> dict:
    """Prepare the 5,319-ID / 220,888-image SurvFace training inputs."""
    base_dir = Path(base_dir).resolve()
    root = Path(dataset_root).expanduser().resolve() if dataset_root else _default_dataset_root()
    proto = prepare_survface_paper_protocol(root)
    source_list = Path(proto["train_list"]).resolve()
    raw_root = Path(proto["raw_root"]).resolve()
    out_dir = paper_processed_dir(root, recipe)
    if max_images is not None:
        if type(max_images) is not int or max_images < 1:
            raise ValueError("max_images must be a positive integer or None")
        out_dir = out_dir.parent / "audits" / f"{recipe}_max{max_images}"

    if recipe == "full_fr_resize_only":
        policy = alignment_policy(base_dir, {"low_resolution": {"method": "resize_only", "interpolation": "bicubic"}}, "survface")
        report_path = out_dir / "report.json"
        if max_images is None and (out_dir / "COMPLETED").is_file() and report_path.is_file():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            if (rep.get("completed") is True and rep.get("prepared") == PAPER_SURVFACE_IMAGES
                    and rep.get("policy") == policy
                    and compute_sha256(str(source_list)) == rep.get("source_list_sha256")
                    and compute_sha256(rep["manifest"]) == rep.get("manifest_sha256")
                    and compute_sha256(rep["train_list"]) == rep.get("train_list_sha256")):
                return rep

        with artifact_lock(out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            shared_manifest = root / "QMUL-SurvFace/processed/shared_v1_boundary_mobilefacenet/manifest.jsonl"
            existing_by_source = {}
            if shared_manifest.is_file():
                with shared_manifest.open(encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            row = json.loads(line)
                            if row.get("status") == "prepared":
                                existing_by_source[os.path.normcase(str(Path(row["source"]).resolve()))] = row

            all_source_rows = list(read_rows(source_list))
            total = len(all_source_rows)
            if max_images is not None:
                all_source_rows = all_source_rows[:max_images]

            records = [None] * len(all_source_rows)
            target_lines = [None] * len(all_source_rows)
            missing_tasks = []
            reused = 0
            for idx, (rel_name, label) in enumerate(all_source_rows):
                src_path = Path(resolve_image_path(str(raw_root), rel_name)).resolve()
                key = os.path.normcase(str(src_path))
                prior = existing_by_source.get(key)
                if prior is not None and Path(prior["output"]).is_file():
                    records[idx] = {
                        "index": idx,
                        "source": str(src_path),
                        "label": label,
                        "source_sha256": prior["source_sha256"],
                        "status": "prepared",
                        "output": prior["output"],
                        "output_sha256": prior["output_sha256"],
                        "transform": "resize_only",
                        "source_size": prior["source_size"],
                    }
                    target_lines[idx] = f"{Path(prior['output']).as_posix()} {label}\n"
                    reused += 1
                else:
                    missing_tasks.append((idx, src_path, label, out_dir))

            if missing_tasks:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    for (idx, _, _, _), (rec, line) in zip(missing_tasks, pool.map(_encode_resize_row, missing_tasks)):
                        records[idx] = rec
                        target_lines[idx] = line

            manifest_path = out_dir / "manifest.jsonl"
            target_list_path = out_dir / "train_list.txt"
            with manifest_path.open("w", encoding="utf-8", newline="\n") as mf:
                for rec in records:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with target_list_path.open("w", encoding="utf-8", newline="\n") as tf:
                tf.writelines(target_lines)

            source_hash = compute_sha256(str(source_list))
            processed = len(records)
            report = {
                "completed": processed == total and max_images is None,
                "total": total,
                "processed": processed,
                "prepared": processed,
                "failed": 0,
                "reused_images": reused,
                "regenerated_images": len(missing_tasks),
                "policy": policy,
                "execution": {"device": "cpu"},
                "source_list": str(source_list),
                "source_list_sha256": source_hash,
                "raw_root": str(raw_root),
                "storage_schema": "shared-preprocessing-v1",
                "manifest": str(manifest_path),
                "manifest_sha256": compute_sha256(str(manifest_path)),
                "train_list": str(target_list_path),
                "train_list_sha256": compute_sha256(str(target_list_path)),
            }
            write_derivation(
                out_dir,
                "image_preprocessing",
                {"source_list": {"path": str(source_list), "sha256": source_hash}, "images": "Per-image source/output SHA-256 in manifest.jsonl"},
                {"raw_root": str(raw_root), "policy": policy, "max_images": max_images},
                {k: report[k] for k in ("total", "processed", "prepared", "failed")},
                "completed" if report["completed"] else "partial_or_failed",
            )
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if report["completed"]:
                (out_dir / "COMPLETED").touch()
            return report

    if recipe == "full_fr_pfld_five_point_alignment":
        policy = pfld_survface_policy(base_dir)
        report_path = out_dir / "report.json"
        manifest_path = out_dir / "images.csv"
        target_list_path = out_dir / "train_list.txt"
        if max_images is None and (out_dir / "COMPLETED").is_file() and report_path.is_file():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            if (rep.get("status") == "completed" and rep.get("image_count") == PAPER_SURVFACE_IMAGES
                    and rep.get("alignment_policy") == policy
                    and compute_sha256(str(manifest_path)) == rep.get("manifest_sha256")
                    and compute_sha256(str(target_list_path)) == rep.get("train_list_sha256")):
                return rep

        if not torch.cuda.is_available():
            raise RuntimeError("PFLD five-point alignment requires CUDA")
        sdk = base_dir / "face_sdk"
        sys.path[:0] = [str(sdk), str(sdk / "models/network_def")]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
                pfld_model = torch.load(policy["landmark_source"], map_location="cpu", weights_only=False).eval().cuda()
        finally:
            del sys.path[:2]

        with artifact_lock(out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            with Path(proto["roles_path"]).open(encoding="utf-8-sig", newline="") as stream:
                by_rel = {r["image_path"].replace("\\", "/"): r["image_id"] for r in csv.DictReader(stream) if r["fr_train"] == "1"}
            all_source_rows = list(read_rows(source_list))
            total = len(all_source_rows)
            if max_images is not None:
                all_source_rows = all_source_rows[:max_images]

            records, list_rows, failed = [], [], 0
            with ThreadPoolExecutor(max_workers=6) as pool:
                for start in range(0, len(all_source_rows), 64):
                    part = all_source_rows[start:start + 64]
                    paths = [Path(resolve_image_path(str(raw_root), rel)).resolve() for rel, _ in part]
                    decoded = list(pool.map(read_bgr, paths))
                    landmarks = tiny_landmarks(pfld_model, [im for im, _ in decoded])
                    for offset, ((rel, label), (im, src_sha), lm) in enumerate(zip(part, decoded, landmarks)):
                        idx = start + offset
                        try:
                            mat = similarity(lm)
                            warped = cv2.warpAffine(im, mat, (112, 112), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            rel_out = Path("images") / f"{idx // 1000:03d}" / f"{idx:06d}.png"
                            dst = out_dir / rel_out
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            ok, enc = cv2.imencode(".png", warped)
                            if not ok or warped.shape != (112, 112, 3):
                                raise ValueError("Invalid warped PFLD image")
                            png_bytes = enc.tobytes()
                            dst.write_bytes(png_bytes)
                            records.append({
                                "index": idx,
                                "label": label,
                                "image_id": by_rel[rel.replace("\\", "/")],
                                "source": rel.replace("\\", "/"),
                                "source_sha256": src_sha,
                                "png": rel_out.as_posix(),
                                "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
                            })
                            list_rows.append(f"{dst.as_posix()} {label}")
                        except (ValueError, cv2.error):
                            failed += 1

            with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
                writer = csv.DictWriter(stream, fieldnames=["index", "label", "image_id", "source", "source_sha256", "png", "png_sha256"])
                writer.writeheader()
                writer.writerows(records)
            target_list_path.write_text("\n".join(list_rows) + ("\n" if list_rows else ""), encoding="utf-8")
            completed = failed == 0 and len(records) == total and max_images is None
            report = {
                "status": "completed" if completed else "partial_or_failed",
                "completed": completed,
                "dataset": "survface",
                "source_list": str(source_list),
                "source_list_sha256": compute_sha256(str(source_list)),
                "original_role_manifest_sha256": proto["roles_sha256"],
                "alignment_policy": policy,
                "manifest_sha256": compute_sha256(str(manifest_path)),
                "train_list_sha256": compute_sha256(str(target_list_path)),
                "image_count": len(records),
                "total": total,
                "prepared": len(records),
                "failed": failed,
                "class_count": PAPER_SURVFACE_CLASSES,
                "fr_train_only": True,
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if completed:
                (out_dir / "COMPLETED").touch()
            return report

    raise ValueError(f"Unsupported SurvFace paper recipe: {recipe}")


def diagnose_survface_checkpoint(
    checkpoint_path: str | Path,
    *,
    sample_size: int = 256,
    seed: int = 8972,
    device: str | None = None,
) -> dict:
    """Inspect backbone active rank, BN spike dimensions, classifier concentration, and margin fallback."""
    path = Path(checkpoint_path).expanduser().resolve()
    payload = load_checkpoint(str(path))
    sd = payload["backbone_state_dict"]
    linear_w = sd["linear.weight"].detach().float().cpu()
    bn_w = sd["bn.weight"].detach().float().cpu()
    bn_b = sd["bn.bias"].detach().float().cpu()
    bn_rv = sd["bn.running_var"].detach().float().cpu()
    sv = torch.linalg.svdvals(linear_w)
    spike_mask = (bn_rv > 15.0) | (bn_w.abs() > 0.5)
    spike_dims = torch.nonzero(spike_mask).squeeze(-1).tolist()
    if isinstance(spike_dims, int):
        spike_dims = [spike_dims]

    history = payload.get("extra_meta", {}).get("history", {}) or payload.get("history", {})
    losses = history.get("train_loss", [])
    report = {
        "checkpoint": str(path),
        "sha256": compute_sha256(str(path)),
        "stage": payload.get("stage"),
        "head": payload.get("head"),
        "epoch": payload.get("epoch"),
        "completed": payload.get("completed"),
        "seed": payload.get("seed"),
        "train_loss_first": float(losses[0]) if losses else None,
        "train_loss_last": float(losses[-1]) if losses else None,
        "backbone": {
            "linear_active_rows_gt_1e2": int((linear_w.norm(dim=1) > 1e-2).sum().item()),
            "linear_singular_values_gt_1e2": int((sv > 1e-2).sum().item()),
            "bn_active_rvar_gt_1e3": int((bn_rv > 1e-3).sum().item()),
            "bn_active_weight_gt_1e2": int((bn_w.abs() > 1e-2).sum().item()),
            "bn_weight_norm": float(bn_w.norm().item()),
            "bn_bias_norm": float(bn_b.norm().item()),
            "bn_spike_dim_count": len(spike_dims),
            "bn_spike_dims": spike_dims[:20],
        },
    }

    hd = payload.get("head_state_dict") or {}
    w_tensor = hd.get("weight") if "weight" in hd else hd.get("kernel")
    if w_tensor is not None:
        wn = F.normalize(w_tensor.detach().float().cpu(), dim=0)
        n_cls = wn.shape[1]
        paircos = float(((wn.mean(1).square().sum() * n_cls - 1.0) / max(n_cls - 1, 1)).item())
        cos_bw = float((F.normalize(bn_b, dim=0) @ F.normalize(wn.mean(1), dim=0)).item()) if bn_b.norm() > 0 else 0.0
        report["classifier"] = {
            "num_classes": int(n_cls),
            "class_paircos": round(paircos, 6),
            "cos_bn_bias_head_mean": round(cos_bw, 6),
        }

    cfg = payload.get("config", {})
    train_list = cfg.get("dataset", {}).get("train_list")
    data_root = cfg.get("dataset", {}).get("data_root") or cfg.get("dataset", {}).get("raw_root")
    if sample_size and sample_size > 0 and train_list and Path(train_list).is_file() and w_tensor is not None:
        from .dataset import FaceImageDataset
        from .model_builder import build_model
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ds = FaceImageDataset(
            str(data_root),
            str(train_list),
            is_train=False,
            alignment=cfg.get("dataset", {}).get("alignment", "resize_only"),
            seed=seed,
        )
        if len(ds) > 0:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(ds), size=min(sample_size, len(ds)), replace=False).tolist()
            model = build_model(cfg["backbone"], dict(cfg["head"], num_class=w_tensor.shape[1])).to(dev)
            model.backbone.load_state_dict(sd, strict=True)
            model.eval()
            wn_dev = F.normalize(w_tensor.detach().float().to(dev), dim=0)
            feats_all, labels_all = [], []
            with torch.inference_mode():
                for start in range(0, len(indices), 64):
                    batch_idx = indices[start:start + 64]
                    imgs = torch.stack([ds[i][0] for i in batch_idx]).to(dev)
                    labs = torch.tensor([ds[i][1] for i in batch_idx], device=dev, dtype=torch.long)
                    feats_all.append(model.backbone(imgs))
                    labels_all.append(labs)
                feats = torch.cat(feats_all, dim=0)
                labs = torch.cat(labels_all, dim=0)
                norms = feats.norm(dim=1)
                fn = F.normalize(feats, dim=1)
                target_cos = (fn * wn_dev[:, labs].T).sum(1)
                sim = fn @ fn.T
                off_diag = sim[~torch.eye(len(fn), dtype=torch.bool, device=dev)]
                margin_arc = float(cfg.get("head", {}).get("margin_arc", 0.6))
                cutoff = math.cos(math.pi - margin_arc)
                fb_count = int((target_cos <= cutoff).sum().item())
                la = float(cfg.get("head", {}).get("l_a", 10.0))
                below_la = int((norms < la).sum().item())
                report["sample_eval"] = {
                    "sample_size": len(indices),
                    "target_cosine_median": round(float(target_cos.median().item()), 6),
                    "feature_paircos_median": round(float(off_diag.median().item()), 6),
                    "feature_norm_median": round(float(norms.median().item()), 4),
                    "arcface_fallback_cutoff": round(cutoff, 6),
                    "arcface_fallback_count": fb_count,
                    "arcface_fallback_rate": round(fb_count / len(indices), 4),
                    "magface_below_la_count": below_la,
                    "magface_below_la_rate": round(below_la / len(indices), 4),
                }
    return report


def write_survface_geometry(output: str | Path, profile: dict, policy: dict, prepared: str | Path) -> int:
    """Write raw_to_arc112_affine_v1 CSV for all 463,341 SurvFace rows when exporting a PFLD-aligned model."""
    output = Path(output)
    prepared = Path(prepared)
    if (policy.get("version") != PAPER_PFLD_VERSION
            or policy.get("landmark_sha256") != PFLD_SHA256
            or compute_sha256(policy["landmark_source"]) != PFLD_SHA256):
        raise ValueError("SurvFace PFLD landmark source or policy version mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("SurvFace PFLD geometry generation requires CUDA")
    base_dir = Path(policy["landmark_source"]).resolve().parents[4]
    sdk = base_dir / "face_sdk"
    sys.path[:0] = [str(sdk), str(sdk / "models/network_def")]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
            model = torch.load(policy["landmark_source"], map_location="cpu", weights_only=False).eval().cuda()
    finally:
        del sys.path[:2]

    with Path(profile["training_manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (prepared / "images.csv").open(encoding="utf-8-sig", newline="") as stream:
        train = list(csv.DictReader(stream))
    by_id = {r["image_id"]: r for r in rows}
    train_rows = [by_id[r["image_id"]] for r in train]
    if len(train_rows) != len(train) or {r["image_id"] for r in train_rows} != {r["image_id"] for r in rows if r["fr_train"] == "1"}:
        raise ValueError("SurvFace PFLD training cohort and role manifest differ")
    ordered_test = [r for r in rows if r["fr_train"] != "1"]
    raw = Path(profile["raw_root"])
    with output.open("w", newline="", encoding="utf-8") as stream, ThreadPoolExecutor(6) as pool:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "source_sha256", "a", "b", "tx", "c", "d", "ty"])
        for group in (train_rows, ordered_test):
            for start in range(0, len(group), 64):
                part = group[start:start + 64]
                paths = [raw / row["image_path"] for row in part]
                decoded = list(pool.map(read_bgr, paths))
                points = tiny_landmarks(model, [im for im, _ in decoded])
                for (row, (_, sha), lm) in zip(part, decoded, points):
                    matrix = similarity(lm)
                    writer.writerow([row["image_id"], sha, *matrix.ravel().tolist()])
    return len(rows)


def parent_origin_comparison() -> dict:
    """Return the verified VGGFace2 Parent-only Origin comparison against UCFace Table I."""
    return dict(UCFACE_PARENT_TABLE_I)
