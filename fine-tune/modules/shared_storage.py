"""External storage settings and local provenance. No ronbun imports."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import socket
import subprocess
import uuid


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def resolve_protocol(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") == "facex-ronbun-roles-v2":
        for key in ("raw_root", "roles_path", "train_list"):
            record[key] = str((path.parent / record[key]).resolve())
        record["sources"] = {k: {**v, "path": str((path.parent / v["path"]).resolve())} for k,v in record["sources"].items()}
    return record


def external_settings(base_dir, dataset_root=None, model_root=None, backbone=None, namespace=None):
    from .notebook_workflow import resolve_settings
    from .backbone_factory import backbone_slug
    data = Path(dataset_root or os.environ.get("FACE_DATASET_ROOT", "F:/face_datasets" if os.name=="nt" else "/data/face_datasets")).expanduser().resolve()
    models = Path(model_root or os.environ.get("FACE_MODEL_ROOT", "F:/models" if os.name=="nt" else "/data/models")).expanduser().resolve()
    cfg = resolve_settings(base_dir)
    if backbone:
        cfg["backbone"].update(backbone)
    cfg["backbone"]["feat_dim"] = 1024
    namespace = namespace or "shared_v1_boundary_" + backbone_slug(cfg["backbone"])
    if (not isinstance(namespace,str) or namespace in (".","..")
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in namespace)):
        raise ValueError("Namespace must be a single explicit alphanumeric/underscore/hyphen name")
    cfg["storage"] = dict(dataset_root=str(data), model_root=str(models), namespace=namespace)
    vgg = data / "VGGFace2"
    extracted = vgg / "extracted/official_train_v1"
    cfg["dataset"].update(raw_root=str(extracted), metadata_root=str(vgg), image_root=str(extracted / "train"),
        train_archive=str(vgg / "data/vggface2_train.tar.gz"), test_archive=str(vgg / "data/vggface2_test.tar.gz"),
        meta_csv=str(vgg / "meta/identity_meta.csv"), train_list=str(vgg / "protocols/official_train_v1/train_list.txt"),
        val_list=None, processed_root=str(vgg / "processed"),
        alignment_report=str(vgg / "processed" / namespace / "report.json"))
    cfg["alignment"]["vggface2"]["archive"] = str(vgg / "meta/bb_landmark.tar.gz")
    for key, folder in (("checkpoint_dir","checkpoints"),("export_dir","exports"),("log_dir","logs"),("report_dir","reports")):
        cfg["paths"][key] = str(models / "facex" / folder / namespace)
    cfg["paths"]["cache_dir"] = str(data / ".cache/facex" / namespace)
    cfg["paths"]["launch_dir"] = str(data / ".cache/facex/launches" / namespace)
    index_path = data / "protocols/shared_v1/handoff.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Prepare the portable dataset handoff first: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != "facex-ronbun-handoff-v2":
        raise ValueError("External settings require a portable v2 handoff")
    profiles = {}
    for name, item in index["protocols"].items():
        if name not in ("survface", "tinyface", "scface"):
            raise ValueError("LFW/RFW cannot be fine-tuning datasets")
        path = (index_path.parent / item["path"]).resolve()
        if sha(path) != item["sha256"]:
            raise ValueError("External handoff hash differs")
        record = resolve_protocol(path)
        if not Path(record["raw_root"]).is_relative_to(data):
            raise ValueError("Portable protocol raw_root escapes the configured dataset root")
        profiles[name] = dict(name=name, enabled=True, split_policy="facex_ronbun_v1", split_seed=record["split_seed"],
            raw_root=record["raw_root"], num_classes=record["num_classes"], expected_images=record["images"],
            train_list=record["train_list"], protocol_record=str(path), training_manifest=record["roles_path"],
            evaluation_manifest=record["roles_path"], research_output=str(path.parent.parent), portable_protocol=True,
            shared_protocol_sha256=item["sha256"], role_manifest_sha256=record["roles_sha256"],
            training_identity_sha256=record["training_identity_sha256"], local_audit_required=False,
            input_size=[112,112], pixel_mean=127.5, pixel_std=128.0,
            alignment_report=str(Path(record["raw_root"]) / "processed" / namespace / "report.json"))
    cfg["finetune_profiles"] = profiles
    cfg["campaign"].update(finetune_datasets=["survface","tinyface","scface"], handoff_path=str(index_path))
    return cfg


@contextmanager
def artifact_lock(directory):
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.with_name(directory.name + ".lock")
    token = uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(), token=token), f)
    try:
        yield
    finally:
        if json.loads(lock.read_text(encoding="utf-8")).get("token") == token:
            lock.unlink()


def write_derivation(directory, operation, sources, parameters, counts, status):
    directory = Path(directory)
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    record = dict(schema="dataset-derivation-v1", repository="FaceX-Zoo", commit=commit, code_sha256=sha(__file__),
                  operation=operation, created_at=datetime.now(timezone.utc).isoformat(), status=status,
                  sources=sources, parameters=parameters, counts=counts,
                  retention="Shared prepared images and source datasets are not removed by KEEP_RAW_RESULTS.")
    (directory / "derivation.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    (directory / "README.md").write_text(
        f"# {operation}\n\nStatus: {status}\n\n"
        "This record describes this execution only. Historical processing is not inferred.\n\n"
        f"```json\n{json.dumps(record,ensure_ascii=False,indent=2)}\n```\n\n"
        "Reproduce with the recorded source and settings in FaceX-Zoo 01/03. "
        "Extraction does not imply image decoding or alignment. Preprocessing failures are in manifest.jsonl. "
        "Partial outputs cannot train a model. Training augmentation remains an online training operation.\n",
        encoding="utf-8")
    return record
