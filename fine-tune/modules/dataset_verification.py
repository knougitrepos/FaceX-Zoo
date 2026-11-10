"""One full check per immutable dataset, with explicit full re-verification.

The user promises not to modify images or their lists after verification. Only
metadata bytes are checked on reuse; this is not current image-byte validation.
Keep alignment.py/integrity_cache.py unchanged so existing transforms and valid
full-check receipts remain identifiable by their original source fingerprints.
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import alignment, integrity_cache
from .shared_storage import artifact_lock

MODE = "immutable_verified_reuse"
MODES = (*integrity_cache.MODES, MODE)
ASSUMPTION = "User keeps verified source/prepared images and lists unchanged; image changes are not detected on reuse."


def validate_options(options):
    if options.get("mode", "content_verified_reuse") not in MODES:
        raise ValueError(f"data_integrity.mode must be one of {MODES}")
    if type(options.get("force_reverify", False)) is not bool:
        raise ValueError("data_integrity.force_reverify must be a bool")


def _write(path, record):
    record = {**record, "updated_at": datetime.now(timezone.utc).isoformat()}
    record["payload_sha256"] = integrity_cache.digest(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(cfg):
    ds = cfg["dataset"]
    path = Path(ds["alignment_report"])
    report = json.loads(path.read_text(encoding="utf-8"))
    directory = path.parent
    if report.get("storage_schema") and (
        directory.with_name(directory.name + ".lock").exists()
        or not (directory / "COMPLETED").is_file()
    ):
        raise ValueError("Alignment is still publishing or partial/failed")
    if (report.get("completed") is not True or report.get("failed") != 0
            or not report.get("total") or report.get("prepared") != report["total"]):
        raise ValueError("Alignment is partial/failed")
    policy = alignment.alignment_policy(Path(__file__).resolve().parents[2], cfg["alignment"], ds["name"])
    if report["policy"] != policy:
        raise ValueError("Alignment policy/model/code differs")
    if Path(report["raw_root"]).resolve() != Path(ds["raw_root"]).resolve():
        raise ValueError("Alignment raw dataset root differs")
    if (Path(report["source_list"]).resolve() != Path(ds["train_list"]).resolve()
            or alignment.sha256(ds["train_list"]) != report["source_list_sha256"]):
        raise ValueError("Alignment source list differs")
    # Stream the few metadata files, never enumerate/open individual images here.
    for key in ("manifest", "train_list"):
        if alignment.sha256(report[key]) != report[key + "_sha256"]:
            raise ValueError(f"Alignment {key} hash mismatch")
    legacy = integrity_cache.receipt_contract(path, report, policy)
    contract = dict(schema="immutable-dataset-verification-v1", full_check_contract=legacy,
                    reuse_validator_sha256=alignment.sha256(__file__), assumption=ASSUMPTION)
    return report, contract


def _load(path, contract):
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        checksum = integrity_cache.digest({k: v for k, v in record.items() if k != "payload_sha256"})
        if (record.get("payload_sha256") != checksum or record.get("contract") != contract
                or record.get("status") != "completed"
                or record.get("verified_images") != contract["full_check_contract"]["expected_images"]):
            raise ValueError("Invalid or unfinished verification record")
        return record
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid/failed verification record; use FORCE_DATA_REVERIFY=True") from exc


def bind_aligned_dataset(cfg, verify_images=True):
    options = cfg.get("data_integrity", {})
    validate_options(options)
    force = options.get("force_reverify", False)
    if options.get("mode", "content_verified_reuse") != MODE or not verify_images:
        selected = copy.deepcopy(cfg)
        if force or options.get("mode") == MODE:
            selected.setdefault("data_integrity", {})["mode"] = "full"
        return alignment.bind_aligned_dataset(selected, verify_images=verify_images)

    print("Data verification: checking report, lists and manifest hashes (no image scan yet).", flush=True)
    report, contract = _metadata(cfg)
    cache = Path(options.get("cache_dir") or Path(cfg["dataset"]["alignment_report"]).parent / "verification_cache")
    receipt = cache / "immutable_v1" / (integrity_cache.digest(contract) + ".json")
    fresh = None
    with artifact_lock(receipt):
        record = None if force else _load(receipt, contract)
        if record is None and not force:
            legacy_path, valid = integrity_cache.load_receipt(cache, contract["full_check_contract"])
            if valid:
                # A v1 receipt was published only after complete mapping/hash/decode
                # validation. Adopt it only with the exact current contract.
                _write(receipt, dict(contract=contract, status="completed", verified_images=report["total"],
                    evidence="existing_full_hash_decode_receipt", source_receipt=str(legacy_path),
                    source_receipt_sha256=alignment.sha256(legacy_path)))
                record = _load(receipt, contract)
        if record is None:
            # Preserve older evidence. A failed/interrupted forced check must not
            # fall back to an older success on the next non-forced invocation.
            if receipt.exists():
                history = receipt.parent / "history"
                history.mkdir(exist_ok=True)
                backup = history / (receipt.stem + "_" + alignment.sha256(receipt) + ".json")
                if not backup.exists():
                    backup.write_bytes(receipt.read_bytes())
            _write(receipt, dict(contract=contract, status="verifying", verified_images=0))
            print("Data verification: full image hash + decode check requested or no valid receipt exists.", flush=True)
            selected = copy.deepcopy(cfg)
            selected["data_integrity"] = {**options, "mode": "full"}
            try:
                fresh = alignment.bind_aligned_dataset(selected)
                # Refuse to publish if metadata changed while images were checked.
                _, after = _metadata(cfg)
                if after != contract:
                    raise ValueError("Dataset metadata changed during full verification")
                _write(receipt, dict(contract=contract, status="completed", verified_images=report["total"],
                    evidence="full_hash_decode_check", source_receipt=None, source_receipt_sha256=None))
            except BaseException as exc:
                _write(receipt, dict(contract=contract, status="failed", verified_images=0,
                                    error=f"{type(exc).__name__}: {exc}"))
                raise
        receipt_hash = alignment.sha256(receipt)

    selected = copy.deepcopy(cfg)
    ds = selected["dataset"]
    policy = report["policy"]
    ds.update(source_train_list=ds["train_list"], train_list=report["train_list"],
              data_root=str(Path(report["train_list"]).parent),
              alignment="resize_only" if policy.get("method") == "resize_only" else policy["version"],
              alignment_policy=policy, alignment_manifest_sha256=report["manifest_sha256"],
              alignment_report_sha256=contract["full_check_contract"]["report_sha256"])
    selected["data_integrity_verification"] = dict(
        mode=MODE, force_reverify=force, reused_full_verification=fresh is None,
        reused_decode_receipt=fresh is None, image_hashes_checked=0 if fresh is None else report["total"] * 2,
        images_decoded=0 if fresh is None else report["total"] * 2,
        verified_images=report["total"], receipt_path=str(receipt), receipt_sha256=receipt_hash,
        contract_sha256=integrity_cache.digest(contract), limitation=ASSUMPTION)
    print(f"Data verification: {'REUSED (0 image files read)' if fresh is None else 'FULL CHECK COMPLETED'}; "
          f"{report['total']} images; receipt: {receipt}", flush=True)
    return selected
