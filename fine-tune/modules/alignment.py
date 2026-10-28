"""Dataset-specific preprocessing with complete, ordered source provenance.

Every source row gets a manifest row, including failures. Only a complete,
failure-free run is eligible for training. Raw datasets/protocols are immutable.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                name, label = line.strip().rsplit(" ", 1)
                yield name, int(label)


def alignment_policy(base_dir, settings, dataset="vggface2"):
    """Record the actual transform; no detector or inferred landmark model."""
    name = dataset.lower()
    policy = {"input_size": [112, 112], "color": "BGR", "output_format": "png",
              "adapter_sha256": sha256(__file__), "opencv": cv2.__version__,
              "face_detection": "none", "failure_policy": "record_and_block",
              "input_size_source": "UCFace supplement, Augmentations"}
    if name == "vggface2":
        import skimage
        options = settings["vggface2"]
        if options["method"] != "official_landmarks":
            raise ValueError("VGGFace2 requires provided five-point landmarks")
        sdk = Path(base_dir) / "face_sdk"
        files = ["core/image_cropper/arcface_cropper/FaceRecImageCropper.py",
                 "core/image_cropper/BaseImageCropper.py", "utils/lms_trans.py"]
        policy.update(version="vggface2_official5_arcface112_v2", method="landmark_alignment",
                      landmark_archive=str(Path(options["archive"]).resolve()),
                      landmark_archive_sha256=sha256(options["archive"]),
                      landmark_member=options["member"],
                      landmark_order=["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"],
                      coordinate_space="released_loose_crop_pixels",
                      interpolation="opencv_warpAffine_default_linear", border="constant_zero",
                      source_hashes={p: sha256(sdk / p) for p in files}, skimage=skimage.__version__)
    elif name in ("tinyface", "survface", "scface"):
        options = settings["low_resolution"]
        if options["method"] != "resize_only" or options["interpolation"] not in INTERPOLATIONS:
            raise ValueError("Low-resolution preprocessing requires resize_only and a supported interpolation")
        policy.update(version="lowres_resize112_v2", method="resize_only",
                      interpolation=options["interpolation"], geometry="full_crop_stretch",
                      provenance="user-approved low-resolution policy; not a verified UCFace alignment recipe")
    else:
        raise ValueError(f"Unsupported preprocessing dataset: {dataset}")
    return policy


INTERPOLATIONS = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR}


class ResizePreprocessor:
    def __init__(self, policy):
        self.policy = policy
        self.execution = {"device": "cpu"}

    def __call__(self, image):
        crop = cv2.resize(image, (112, 112), interpolation=INTERPOLATIONS[self.policy["interpolation"]])
        return crop, {"transform": "resize_only", "source_size": list(image.shape[:2])}


class VGGFace2Landmarks:
    """Disk-backed exact-name lookup; stream the archive without extracting it.

    A temporary SQLite index bounds RAM use for 3.14 million rows. The archive
    and selected member are fingerprinted in the policy; no source is modified.
    """
    def __init__(self, archive, member, cache_dir):
        import csv
        import sqlite3
        import tarfile
        import tempfile
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="landmarks_", dir=cache_dir)
        self.db = sqlite3.connect(str(Path(self.temp.name) / "index.sqlite"))
        self.count = 0
        try:
            self.db.execute("CREATE TABLE points (name TEXT PRIMARY KEY, coords TEXT NOT NULL) WITHOUT ROWID")
            found = False
            with tarfile.open(archive, "r|gz") as tar:
                for entry in tar:
                    if entry.name != member:
                        continue
                    if found or not entry.isfile():
                        raise ValueError("Duplicate/non-file landmark member")
                    found = True
                    reader = csv.reader(line.decode("utf-8-sig") for line in tar.extractfile(entry))
                    expected = ["NAME_ID"] + [f"P{i}{axis}" for i in range(1, 6) for axis in "XY"]
                    if next(reader, None) != expected:
                        raise ValueError("Unexpected five-point CSV header/order")
                    batch = []
                    for row in reader:
                        if len(row) != 11:
                            raise ValueError("Invalid landmark CSV row")
                        points = np.asarray(row[1:], dtype=np.float64)
                        if not np.isfinite(points).all():
                            raise ValueError(f"Non-finite provided landmarks: {row[0]}")
                        key = self.key(row[0])
                        batch.append((key, json.dumps(points.tolist())))
                        self.count += 1
                        if len(batch) == 10000:
                            self.db.executemany("INSERT INTO points VALUES (?, ?)", batch)
                            batch.clear()
                        if self.count % 500000 == 0:
                            print(f"Provided landmark rows indexed: {self.count}", flush=True)
                    self.db.executemany("INSERT INTO points VALUES (?, ?)", batch)
            if not found or not self.count:
                raise ValueError("Missing/empty landmark CSV member")
            self.db.commit()
        except Exception:
            self.close()
            raise

    @staticmethod
    def key(name):
        name = name.replace("\\", "/").removeprefix("train/")
        if name.lower().endswith(".jpg"):
            name = name[:-4]
        parts = name.split("/")
        if len(parts) != 2 or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"Unexpected VGGFace2 image key: {name}")
        return name

    def lookup(self, name):
        row = self.db.execute("SELECT coords FROM points WHERE name=?", (self.key(name),)).fetchone()
        if row is None:
            raise ValueError(f"Missing provided landmarks: {name}")
        return json.loads(row[0])

    def close(self):
        self.db.close()
        self.temp.cleanup()


class ProvidedLandmarkAligner:
    def __init__(self, base_dir, raw_root, policy, cache_dir):
        sdk = Path(base_dir) / "face_sdk"
        sys.path.insert(0, str(sdk))
        # Only the original cropper is imported, not detection/alignment models.
        from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
        self.cropper = FaceRecImageCropper()
        self.root = Path(raw_root).resolve()
        self.policy = policy
        self.landmarks = VGGFace2Landmarks(policy["landmark_archive"], policy["landmark_member"], cache_dir)
        if sha256(policy["landmark_archive"]) != policy["landmark_archive_sha256"]:
            self.close()
            raise ValueError("Landmark archive changed during indexing")
        self.execution = {"device": "cpu", "provided_landmark_rows": self.landmarks.count}

    def process(self, image, source):
        key = self.landmarks.key(Path(source).relative_to(self.root).as_posix())
        points = self.landmarks.lookup(key)
        crop = self.cropper.crop_image_by_mat(image, points)
        return crop, {"transform": "landmark_alignment", "landmark_key": key, "landmarks": points}

    def close(self):
        self.landmarks.close()


def create_preprocessor(base_dir, cfg, ds):
    policy = alignment_policy(base_dir, cfg["alignment"], ds["name"])
    if policy["method"] == "resize_only":
        return ResizePreprocessor(policy)
    return ProvidedLandmarkAligner(base_dir, ds["raw_root"], policy, cfg["paths"]["cache_dir"])


def prepare_alignment(source_list, raw_root, output_dir, aligner, max_images=None):
    from .shared_storage import artifact_lock
    with artifact_lock(output_dir):
        return _prepare_alignment(source_list, raw_root, output_dir, aligner, max_images)


def _prepare_alignment(source_list, raw_root, output_dir, aligner, max_images=None):
    """Stream PNGs and a JSONL manifest; refuse overwrites of previous attempts.

    A bounded audit is explicitly partial. Retry failures using a new output
    directory after reviewing the failure policy, never a silent resize fallback.
    """
    from .dataset import resolve_image_path
    source_list, root = Path(source_list).resolve(), Path(raw_root).resolve()
    out = Path(output_dir).resolve()
    if max_images is not None and (type(max_images) is not int or max_images < 1):
        raise ValueError("max_images must be positive or None")
    out.mkdir(parents=True, exist_ok=False)
    manifest, target_list = out / "manifest.jsonl", out / "train_list.txt"
    source_hash = sha256(source_list)
    total = sum(1 for _ in read_rows(source_list))
    processed = successes = 0
    started = last_progress = time.monotonic()
    seen = set()
    with manifest.open("w", encoding="utf-8") as records, target_list.open("w", encoding="utf-8") as targets:
        for i, (name, label) in enumerate(read_rows(source_list)):
            if max_images is not None and i >= max_images:
                break
            source = Path(resolve_image_path(str(root), name)).resolve()
            row = {"index": i, "source": str(source), "label": label}
            try:
                if str(source) in seen:
                    raise ValueError("Duplicate source image")
                seen.add(str(source))
                row["source_sha256"] = sha256(source)
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Image decode failed")
                crop, details = (aligner.process(image, source) if hasattr(aligner, "process") else aligner(image))
                if crop.shape != (112, 112, 3) or crop.dtype != np.uint8:
                    raise ValueError("Preprocessing output must be uint8 112x112 BGR")
                target = out / "images" / f"{i // 1000:06d}" / f"{i:08d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), crop):
                    raise OSError("PNG write failed")
                row.update(status="prepared", output=str(target), output_sha256=sha256(target), **details)
                targets.write(f"{target.as_posix()} {label}\n")
                successes += 1
            except (ValueError, OSError, RuntimeError, cv2.error, AssertionError) as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            records.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed += 1
            now = time.monotonic()
            if now - last_progress >= 10:
                print(f"Preprocessing {processed}/{total}: prepared={successes}, failed={processed-successes}, "
                      f"images/s={processed/max(now-started, 1e-9):.2f}", flush=True)
                last_progress = now
    if sha256(source_list) != source_hash:
        raise ValueError("Source list changed during alignment")
    report = {"completed": processed == total == successes and total > 0,
              "total": total, "processed": processed, "prepared": successes,
              "failed": processed - successes, "policy": aligner.policy,
              "execution": getattr(aligner, "execution", {}),
              "source_list": str(source_list), "source_list_sha256": source_hash,
              "raw_root": str(root), "manifest": str(manifest), "manifest_sha256": sha256(manifest),
              "train_list": str(target_list), "train_list_sha256": sha256(target_list)}
    from .shared_storage import write_derivation
    report["storage_schema"] = "shared-preprocessing-v1"
    write_derivation(out, "image_preprocessing", {"source_list": {"path":str(source_list), "sha256":source_hash},
        "images": "Per-image source/output SHA-256 in manifest.jsonl"},
        {"raw_root":str(root), "policy":aligner.policy, "max_images":max_images},
        {k:report[k] for k in ("total","processed","prepared","failed")},
        "completed" if report["completed"] else "partial_or_failed")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["completed"]:
        (out / "COMPLETED").touch()
    return report


def verify_alignment(report_path, source_list, expected_policy, verify_images=True, *,
                     integrity_mode="full", cache_dir=None):
    """Check ordering, labels, complete coverage, provenance and image hashes."""
    from .dataset import resolve_image_path
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("storage_schema"):
        directory = Path(report_path).parent
        if directory.with_name(directory.name + ".lock").exists() or not (directory / "COMPLETED").is_file():
            raise ValueError("Alignment is still publishing or partial/failed")
    if not report.get("completed") or report["failed"] != 0 or report.get("prepared") != report["total"]:
        raise ValueError("Alignment is partial/failed; review failures before training")
    if report["policy"] != expected_policy:
        raise ValueError("Alignment policy/model/code differs")
    if Path(report["source_list"]).resolve() != Path(source_list).resolve() or report["source_list_sha256"] != sha256(source_list):
        raise ValueError("Alignment source list differs")
    for key in ("manifest", "train_list"):
        if sha256(report[key]) != report[key + "_sha256"]:
            raise ValueError(f"Alignment {key} hash mismatch")
    from .integrity_cache import MODES, receipt_contract, load_receipt, verify_image, publish_receipt
    if integrity_mode not in MODES:
        raise ValueError("Unknown image integrity mode")
    contract, receipt, reused = None, None, False
    if verify_images and integrity_mode == "content_verified_reuse":
        if cache_dir is None:
            raise ValueError("Conditional image verification requires a cache directory")
        contract = receipt_contract(report_path, report, expected_policy)
        receipt, reused = load_receipt(cache_dir, contract)
    count = 0
    with Path(report["manifest"]).open(encoding="utf-8") as manifest:
        for original, aligned, record in itertools.zip_longest(read_rows(source_list), read_rows(report["train_list"]), manifest):
            if original is None or aligned is None or record is None:
                raise ValueError("Alignment coverage differs")
            row = json.loads(record)
            source = Path(resolve_image_path(report["raw_root"], original[0])).resolve()
            if (row["index"] != count or row["status"] != "prepared" or
                row["label"] != original[1] or original[1] != aligned[1] or
                Path(row["source"]).resolve() != source or Path(row["output"]).resolve() != Path(aligned[0]).resolve()):
                raise ValueError("Alignment image/identity mapping differs")
            if verify_images:
                verify_image(source, row["source_sha256"], decode=not reused)
                verify_image(row["output"], row["output_sha256"], decode=not reused, prepared=True)
            count += 1
    if count != report["total"]:
        raise ValueError("Alignment count differs")
    if receipt is not None and not reused:
        publish_receipt(receipt, contract, count)
    report["verification"] = dict(mode=integrity_mode if verify_images else "mapping_only",
        image_hashes_checked=count * 2 if verify_images else 0,
        images_decoded=count * 2 if verify_images and not reused else 0,
        reused_decode_receipt=reused, receipt_path=str(receipt) if receipt else None,
        receipt_sha256=sha256(receipt) if receipt else None,
        limitation="pre-use byte verification; dataset must not change during the run")
    return report


def bind_aligned_dataset(cfg, verify_images=True):
    """Return a separate effective config; keep original protocol paths intact."""
    import copy
    selected = copy.deepcopy(cfg)
    ds = selected["dataset"]
    policy = alignment_policy(Path(__file__).resolve().parents[2], selected["alignment"], ds["name"])
    integrity = selected.get("data_integrity", {"mode": "content_verified_reuse"})
    cache_dir = integrity.get("cache_dir") or str(Path(ds["alignment_report"]).parent / "verification_cache")
    report = verify_alignment(ds["alignment_report"], ds["train_list"], policy, verify_images,
                              integrity_mode=integrity["mode"], cache_dir=cache_dir)
    if Path(report["raw_root"]).resolve() != Path(ds["raw_root"]).resolve():
        raise ValueError("Alignment raw dataset root differs")
    ds.update(source_train_list=ds["train_list"], train_list=report["train_list"],
              data_root=str(Path(report["train_list"]).parent), alignment=("resize_only" if policy.get("method") == "resize_only" else policy["version"]),
              alignment_policy=policy, alignment_manifest_sha256=report["manifest_sha256"],
              alignment_report_sha256=sha256(ds["alignment_report"]))
    selected["data_integrity_verification"] = report["verification"]
    return selected
