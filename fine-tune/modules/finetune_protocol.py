"""Dataset-specific training provenance. Evaluation images never enter FR training."""
import csv
import json
import os
from pathlib import Path

from .checkpoint_manager import compute_sha256

PAPER_SPLITS = {"survface": (5319, 220888), "tinyface": (2570, 7804), "scface": (50, 800)}


def default_profiles(root):
    root = Path(root)
    profiles = {}
    for name, (classes, images) in PAPER_SPLITS.items():
        folder = root / "fine-tune/data/protocols/local_qmul_v1" / name
        profiles[name] = {"name": name, "raw_root": str(root / ({'tinyface':'tinyface','survface':'QMUL-SurvFace','scface':'scface'}[name])), "num_classes": classes,
                          "expected_images": images, "train_list": str(folder / "train_list.txt"),
                          "val_list": None, "protocol_record": str(folder / "protocol.json"),
                          "training_manifest": str(folder / "training_manifest.csv"),
                          "evaluation_manifest": str(folder / "evaluation_manifest.csv"),
                          "enabled": name != "scface", "local_audit_required": name != "scface",
                          "local_audit": str(folder / "local_audit.json")}
    profiles["survface"]["legacy_fiqa_manifest"] = "C:/ronbun/data/interim/survface/training_manifest.csv"
    return profiles


def canonical_identity(row, name, training):
    if name == "survface":
        path = Path(row["image_path"].replace("\\", "/"))
        if training:
            if "training_set" not in path.parts:
                raise ValueError("SurvFace training image must come from training_set")
            identity = path.parent.name
        else:
            if "training_set" in path.parts or "Face_Identification_Test_Set" not in path.parts:
                raise ValueError("Unexpected SurvFace evaluation image path")
            # Native filenames use both ID_camX_N and ID_N; protocol labels
            # are remapped and must not be compared with training directory IDs.
            identity = path.stem.split("_", 1)[0]
            if identity == "distractors":
                suffix = path.stem.removeprefix("distractors_")
                if path.parent.name != "unmated_probe" or row.get("protocol_role") != "unknown_unknown_probe" or not suffix.isdigit():
                    raise ValueError("Unexpected SurvFace distractor entry")
                return None  # Official distractors have no recoverable native identity.
        if not identity.isdigit():
            raise ValueError("SurvFace native identity must be numeric")
        return identity
    # Do not compare remapped per-split labels: native identity IDs are required.
    identity = row.get("source_identity_id")
    if name == "tinyface" and not identity and row.get("protocol_role") == "gallery_distractor":
        if "Gallery_Distractor" not in Path(row["image_path"]).parts:
            raise ValueError("Unexpected TinyFace distractor path")
        return None
    if not identity:
        raise ValueError("TinyFace/SCFace manifests require canonical source_identity_id")
    return identity


def prepare_protocol(profile):
    name = profile["name"]
    expected_classes, expected_images = PAPER_SPLITS[name]
    if not profile.get("enabled", True):
        raise ValueError(f"{name} training disabled: dataset not acquired/verified")
    local_audit = None
    if profile.get("local_audit_required"):
        local_audit = json.loads(Path(profile["local_audit"]).read_text(encoding="utf-8"))
        if Path(local_audit["raw_root"]).resolve() != Path(profile["raw_root"]).resolve():
            raise ValueError("Local audit dataset root differs")
        if not local_audit.get("completed") or local_audit.get("all_images_decode") is not True:
            raise ValueError("Complete local dataset decode audit required")
        for key in ("training_manifest", "evaluation_manifest"):
            if compute_sha256(profile[key]) != local_audit[key + "_sha256"]:
                raise ValueError("Manifest differs from full local audit")
    train_manifest, eval_manifest = Path(profile["training_manifest"]), Path(profile["evaluation_manifest"])
    if not train_manifest.is_file() or not eval_manifest.is_file():
        raise FileNotFoundError(f"Provide verified official training/evaluation manifests for {name}: {train_manifest}, {eval_manifest}")
    rows, training_ids, evaluation_ids = [], set(), set()
    source_calibration_images = 0
    unknown_identity_images = 0
    with train_manifest.open(encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            allowed = ("training", "train", "development", "calibration") if name == "survface" else ("training", "train", "development")
            if row.get("protocol_role") != "training" or row.get("split") not in allowed:
                raise ValueError("Training manifest contains an evaluation/calibration row")
            source_calibration_images += row.get("split") == "calibration"
            identity = canonical_identity(row, name, True)
            training_ids.add(identity)
            rows.append((row["image_path"].replace("\\", "/"), identity))
    with eval_manifest.open(encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            identity = canonical_identity(row, name, False)
            if identity is None:
                unknown_identity_images += 1
            else:
                evaluation_ids.add(identity)
    if training_ids & evaluation_ids:
        raise ValueError("Training/evaluation identities overlap")
    if len(training_ids) != expected_classes or len(rows) != expected_images:
        raise ValueError(f"{name} counts differ from Supplement Table IX: {len(training_ids)} IDs/{len(rows)} images")
    if profile["num_classes"] != expected_classes or profile.get("val_list"):
        raise ValueError("Use the complete official training set and its class count; test is not validation")
    if len({path for path, _ in rows}) != len(rows):
        raise ValueError("Duplicate training image path")
    legacy_manifest = profile.get("legacy_fiqa_manifest")
    legacy_overlap = "not_checked"
    if legacy_manifest and Path(legacy_manifest).is_file():
        train_names = {Path(path).name for path, _ in rows}
        with Path(legacy_manifest).open(encoding="utf-8-sig", newline="") as src:
            legacy_overlap = sum(row.get("split") == "calibration" and Path(row["image_path"]).name in train_names for row in csv.DictReader(src))
    root = Path(profile["raw_root"]).resolve()
    for path, _ in rows[:20]:
        if not (root / path).is_file():
            raise FileNotFoundError(str(root / path))
    mapping = {identity: label for label, identity in enumerate(sorted(training_ids))}
    content = "".join(f"{path} {mapping[identity]}\n" for path, identity in rows)
    target = Path(profile["train_list"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Existing protocol list differs; select a new protocol directory: {target}")
    if not target.exists():
        temp = target.with_suffix(".tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, target)
    record = {"name": name, "raw_root": str(root), "num_classes": len(mapping), "images": len(rows),
              "train_list": str(target.resolve()), "train_sha256": compute_sha256(str(target)),
              "training_manifest": str(train_manifest.resolve()), "training_manifest_sha256": compute_sha256(str(train_manifest)),
              "evaluation_manifest": str(eval_manifest.resolve()), "evaluation_manifest_sha256": compute_sha256(str(eval_manifest)),
              "identity_disjoint": True, "evaluation_identities": len(evaluation_ids), "labels": mapping,
              "identity_check_scope": "native IDs verified where available; official distractor identity disjointness relies on dataset protocol",
              "evaluation_images_without_native_identity": unknown_identity_images,
              "all_images_decode": "not_checked", "source": "UCFace Supplement II-A/Table IX",
              "existing_ronbun_calibration_overlap_images": legacy_overlap if legacy_manifest else source_calibration_images,
              "ronbun_calibration_reuse": "requires new protocol review; no automatic reuse with this FR model",
              "validation": "none; all official training images used, no test-based checkpoint selection"}
    if local_audit:
        record.update(local_audit=str(Path(profile["local_audit"]).resolve()),
                      local_audit_sha256=compute_sha256(profile["local_audit"]), all_images_decode=True,
                      dataset_release_notes=local_audit["notes"])
    out = Path(profile["protocol_record"])
    serialized = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if out.exists() and out.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Existing protocol provenance differs: {out}")
    if not out.exists():
        out.write_text(serialized, encoding="utf-8")
    return {k: v for k, v in record.items() if k != "labels"}


def verify_protocol(profile):
    if not profile.get("enabled", True):
        raise ValueError(f"{profile['name']} training disabled")
    record = json.loads(Path(profile["protocol_record"]).read_text(encoding="utf-8"))
    expected = PAPER_SPLITS[profile["name"]]
    if not record.get("identity_disjoint") or (record["num_classes"], record["images"]) != expected:
        raise ValueError("Invalid dataset split provenance")
    if Path(profile["raw_root"]).resolve() != Path(record["raw_root"]).resolve() or profile["num_classes"] != record["num_classes"]:
        raise ValueError("Dataset root/class count differs from verified protocol")
    if Path(profile["train_list"]).resolve() != Path(record["train_list"]).resolve() or profile.get("val_list"):
        raise ValueError("Training/validation list differs from verified protocol")
    for key in ("training_manifest", "evaluation_manifest"):
        if Path(profile[key]).resolve() != Path(record[key]).resolve() or compute_sha256(profile[key]) != record[key + "_sha256"]:
            raise ValueError(f"Source manifest changed: {key}")
    if compute_sha256(profile["train_list"]) != record["train_sha256"]:
        raise ValueError("Verified training list changed")
    if profile.get("local_audit_required"):
        if record.get("all_images_decode") is not True or compute_sha256(profile["local_audit"]) != record.get("local_audit_sha256"):
            raise ValueError("Missing/changed local image audit provenance")
    return record


def prepare_selected_protocols(cfg, audit_workers=4):
    """Register both local official releases, never carve FIQA fit out of FR train."""
    from .local_datasets import prepare_local_release
    results = []
    for name in cfg["campaign"]["finetune_datasets"]:
        profile = cfg["finetune_profiles"][name]
        if not profile.get("enabled", True):raise ValueError(f"{name} training disabled")
        if profile.get("local_audit_required"):
            prepare_local_release(profile,workers=audit_workers)
        results.append(prepare_protocol(profile))
    return results
