"""Reuse successful decode checks only after current image bytes match hashes.

No stat/mtime trust, no mutable 'latest' receipt, and no skip-hash mode. This
avoids repeated decoding, not all filesystem I/O. Datasets must remain unchanged
while a training/export process is using them.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

MODES = ("content_verified_reuse", "full")
SCHEMA = "image-decode-receipt-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def receipt_contract(report_path, report, policy):
    return dict(schema=SCHEMA, report_path=str(Path(report_path).resolve()),
                report_sha256=hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
                manifest_sha256=report["manifest_sha256"], source_list_sha256=report["source_list_sha256"],
                train_list_sha256=report["train_list_sha256"], policy=policy,
                expected_images=report["total"], opencv=cv2.__version__, numpy=np.__version__,
                validator_sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), Path(__file__).with_name("alignment.py"))},
                checks="sha256_raw_and_prepared; decode_both; prepared_112x112_BGR")


def load_receipt(cache_dir, contract):
    path = Path(cache_dir) / (digest(contract) + ".json")
    if not path.exists():
        return path, False
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("contract") != contract or record.get("completed") is not True
            or record.get("verified_images") != contract["expected_images"]
            or record.get("payload_sha256") != digest({k: v for k, v in record.items() if k != "payload_sha256"})):
        raise ValueError("Invalid image verification receipt; preserve it and use a new cache directory")
    return path, True


def verify_image(path, expected_hash, *, decode, prepared=False):
    # Hash and decode the same bytes; never hash one version and decode another.
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"Aligned/source image hash mismatch: {path}")
    if decode:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or (prepared and image.shape != (112, 112, 3)):
            raise ValueError(f"Invalid {'aligned' if prepared else 'source'} image: {path}")


def publish_receipt(path, contract, count):
    if count != contract["expected_images"]:
        raise ValueError("Incomplete image verification cannot be cached")
    record = dict(contract=contract, completed=True, verified_images=count)
    record["payload_sha256"] = digest(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
