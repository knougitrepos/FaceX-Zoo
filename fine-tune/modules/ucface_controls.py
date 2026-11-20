"""Reproduce the pinned seed-8972 controls independently of the baseline campaign."""
import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from . import notebook_workflow as workflow
from .checkpoint_manager import (CheckpointManager, compute_config_hash, compute_sha256,
                                 load_checkpoint, validate_checkpoint_for_export)
from .control_geometry import SOURCES, write_geometry
from .shared_storage import artifact_lock

PARENT = "3ac633038bed2bf153e2fe58aae900637881aba8c78c979d22712cc995cf879a"
SURV_CONTROL = "survface_continuous_margin_s8972_v1"
ALIGNMENT_CONTROL = "ucface_alignment_arcface_s8972_20261112_v1"
WEIGHTS = {"scface": "a921a13810efe5c646f153ead5ce76972c099c1fe2fc0d62ffd868b28ba1d65e",
           "tinyface": "3af40ef5d90699e47fa208202aa69ca99ce7bbe0900f21cb616747b0e805d7ea"}

# Verified 40-epoch aligned checkpoints (SCFace two_eye_alignment & TinyFace pfld_five_point_alignment).
COMPLETED_ALIGNED_CHECKPOINTS = {
    "scface_adaface_two_eye_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/checkpoint/scface_adaface_two_eye_alignment_s8972/finetune_scface_adaface_seed8972/finetune_mobilefacenet_1024d_adaface_epoch_40.pt",
        "5998e85019adfdebde8c14a0f00652e394174308bbc3b8520a258c1627c4a956",
    ),
    "scface_arcface_two_eye_alignment_s8972": (
        "facex/diagnostics/ucface_alignment_arcface_s8972_20261112_v1/checkpoints/finetune_scface_arcface_seed8972/finetune_mobilefacenet_1024d_arcface_epoch_40.pt",
        "a921a13810efe5c646f153ead5ce76972c099c1fe2fc0d62ffd868b28ba1d65e",
    ),
    "scface_cosface_two_eye_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/checkpoint/scface_cosface_two_eye_alignment_s8972/finetune_scface_cosface_seed8972/finetune_mobilefacenet_1024d_cosface_epoch_40.pt",
        "294f8817ba526e7e20d5d21b01b2d99fbbd6bfb08ab7988918f8b42bd7538461",
    ),
    "scface_magface_two_eye_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/checkpoint/scface_magface_two_eye_alignment_s8972/finetune_scface_magface_seed8972/finetune_mobilefacenet_1024d_magface_epoch_40.pt",
        "b32e4309ca1540c65af1e4c8a703334e1a8981e577e4d37f0184f80633728acb",
    ),
    "tinyface_adaface_pfld_five_point_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/checkpoint/tinyface_adaface_pfld_five_point_alignment_s8972/finetune_tinyface_adaface_seed8972/finetune_mobilefacenet_1024d_adaface_epoch_40.pt",
        "11d0956fc37b4d9f62dc19c4d91dc448dfc443cabf215cf29b231df6c10854a3",
    ),
    "tinyface_arcface_pfld_five_point_alignment_s8972": (
        "facex/diagnostics/ucface_alignment_arcface_s8972_20261112_v1/checkpoints/finetune_tinyface_arcface_seed8972/finetune_mobilefacenet_1024d_arcface_epoch_40.pt",
        "3af40ef5d90699e47fa208202aa69ca99ce7bbe0900f21cb616747b0e805d7ea",
    ),
    "tinyface_cosface_pfld_five_point_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/checkpoint/tinyface_cosface_pfld_five_point_alignment_s8972/finetune_tinyface_cosface_seed8972/finetune_mobilefacenet_1024d_cosface_epoch_40.pt",
        "6a5935ae36d4c7b4c259de92e512e34731bae0c400aa1defb1eed06900f99ffc",
    ),
    "tinyface_magface_pfld_five_point_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/checkpoint/tinyface_magface_pfld_five_point_alignment_s8972/finetune_tinyface_magface_seed8972/finetune_mobilefacenet_1024d_magface_epoch_40.pt",
        "4d762247826a6074d72fa06dbac529ab0b0b2ac9cc10d4b75dcc83c7867ef98f",
    ),
}

COMPLETED_ALIGNED_EXPORTS = {
    "scface_adaface_two_eye_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/exports/scface_adaface_two_eye_alignment_s8972/scface_adaface_two_eye_alignment_s8972.pt",
        "8e44ba03d0434ff51187e9b5dbf9521d5dbf7c45350f43775c5a10483de1a339",
    ),
    "scface_cosface_two_eye_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/exports/scface_cosface_two_eye_alignment_s8972/scface_cosface_two_eye_alignment_s8972.pt",
        "955fabe4c96668ea5f11efd5818357a831f03e82a72b0981d68d2c1bcef85204",
    ),
    "scface_magface_two_eye_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/exports/scface_magface_two_eye_alignment_s8972/scface_magface_two_eye_alignment_s8972.pt",
        "234b8a9e78bc76238c563981a4d023a88614720c27eb10bcc787897ec80995dd",
    ),
    "tinyface_adaface_pfld_five_point_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/exports/tinyface_adaface_pfld_five_point_alignment_s8972/tinyface_adaface_pfld_five_point_alignment_s8972.pt",
        "fddf2ac59ca774b5e01dce61b709a2df562b45d59aa9be04f14eb557cb2b1532",
    ),
    "tinyface_cosface_pfld_five_point_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/exports/tinyface_cosface_pfld_five_point_alignment_s8972/tinyface_cosface_pfld_five_point_alignment_s8972.pt",
        "f7b787fd6adb01184e33602995191a69849bfb881279b98d636f8ae74e5400f5",
    ),
    "tinyface_magface_pfld_five_point_alignment_s8972": (
        "facex/selected/ucface_recipes_v1/exports/tinyface_magface_pfld_five_point_alignment_s8972/tinyface_magface_pfld_five_point_alignment_s8972.pt",
        "34e55b431f6a73e632286c9cee155eae8ebf92eea5bf2cc0c2a9dfa29131fba9",
    ),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_file(path, digest):
    if compute_sha256(str(path)) != digest:
        raise ValueError(f"Control source checksum mismatch: {path}")


def survface_settings(baseline, *, purpose=None):
    if purpose != "diagnostic_control":
        raise ValueError("continuous_margin is an optional diagnostic_control, not a UCFace baseline")
    cfg = copy.deepcopy(baseline)
    parent = workflow.checkpoint_path(cfg, "pretrain", "arcface", 8972)
    check_file(parent, PARENT)
    profile = cfg["finetune_profiles"]["survface"]
    protocol = workflow.verify_protocol(profile)
    if protocol["roles_sha256"] != "be171ed212db0c8115230a91d80df9db3a93cd5ea017218626cbc37dcd8dce4c":
        raise ValueError("Select the reference SurvFace 4255-identity split")
    previous = parent.parent.parent / "finetune_survface_arcface_seed8972/notebook_settings.json"
    check_file(previous, "1c31a76acd95cf29aef8ead2575c075564bee06e6a1956764ffa053b193cd74c")
    ref = read_json(previous)
    for key in ("augmentation", "backbone", "heads", "parent_head", "training", "alignment"):
        if cfg[key] != ref[key]:
            raise ValueError(f"Reference training settings differ: {key}")
    if (cfg["campaign"]["parent_seed"] != 8972 or profile != ref["finetune_profiles"]["survface"]
            or cfg["dataset"]["pixel_mean"] != 127.5 or cfg["dataset"]["pixel_std"] != 128):
        raise ValueError("Baseline parent, FR split or normalization differs")
    cfg["project"].update(seed=8972, seed_list=[8972])
    cfg["campaign"].update(heads=["arcface"], finetune_datasets=["survface"],
                           purpose=purpose,
                           parent_checkpoint=str(parent), parent_sha256=PARENT)
    cfg["heads"]["arcface"]["margin_am"] = 1 - math.cos(cfg["heads"]["arcface"]["margin_arc"])
    cfg["paths"] = {k: str(Path(v).with_name(SURV_CONTROL)) for k, v in cfg["paths"].items()}
    workflow.validate_settings(cfg)
    workflow.validate_paper_parameters(cfg)
    return cfg


def export_alignment_control(cfg, dataset, diagnostic_root, export_root):
    root = Path(diagnostic_root)
    if dataset not in WEIGHTS or not (root / "COMPLETED").is_file():
        raise ValueError("Select a completed SCFace or TinyFace alignment control")
    profile = cfg["finetune_profiles"][dataset]
    protocol = workflow.verify_protocol(profile)
    summary = read_json(root / "summary.json")
    if summary["status"] != "completed" or summary["run_id"] != ALIGNMENT_CONTROL:
        raise ValueError("Unexpected diagnostic campaign")
    record = summary["results"][dataset]
    source = root / record["new_model"]["file"]
    check_file(source, WEIGHTS[dataset])
    check_file(workflow.checkpoint_path(cfg, "pretrain", "arcface", 8972), PARENT)
    payload = load_checkpoint(str(source))
    validate_checkpoint_for_export(payload, target_epoch=40, expected_seed=8972, expected_head="arcface",
                                   expected_stage="finetune", expected_parent_hash=PARENT)
    stored = payload["config"]
    data = stored["dataset"]
    prep = root / "prepared" / dataset
    check_file(prep / "report.json", data["alignment_report_sha256"])
    check_file(prep / "train_list.txt", data["train_list_hash"])
    check_file(prep / "images.csv", data["alignment_manifest_sha256"])
    policy = stored["preprocessing"]["alignment_policy"]
    if (stored["dataset"]["name"] != dataset or compute_config_hash(stored) != payload["config_hash"]
            or data["role_manifest_sha256"] != protocol["roles_sha256"]
            or data["training_identity_sha256"] != protocol["training_identity_sha256"]
            or policy["version"] != SOURCES[dataset][0]):
        raise ValueError("Alignment control config, roles or preprocessing mismatch")
    alias = f"{dataset}_arcface_aligned_s8972"
    destination = Path(export_root) / alias
    filename = alias + ".pt"
    with artifact_lock(destination):
        if not destination.exists():
            with TemporaryDirectory(dir=destination.parent) as temp:
                staging = Path(temp) / alias
                staging.mkdir()
                geometry = staging / "affine.csv"
                count = write_geometry(geometry, dataset, profile, policy, prep)
                metadata = dict(run_id=alias, dataset=dataset, stage="export", head="arcface", seed=8972,
                                completed=True, is_smoke=False, feat_dim=1024, epochs=40, common_parent_seed=8972,
                                parent_hash=PARENT, source_checkpoint=str(source), source_sha256=WEIGHTS[dataset],
                                source_config=stored, source_config_hash=payload["config_hash"],
                                preprocessing={**stored["preprocessing"], "color": "BGR"},
                                protocol_record=read_json(profile["protocol_record"]),
                                input_geometry=dict(file=geometry.name, sha256=compute_sha256(str(geometry)),
                                                    rows=count, schema="raw_to_arc112_affine_v1"),
                                reproduction_scope="Single-seed alignment control with dataset-specific geometry")
                CheckpointManager(str(staging)).export_backbone_only(payload, filename, metadata)
                staging.rename(destination)
        out = destination / filename
        meta = read_json(out.with_name(out.stem + "_meta.json"))
        if (meta["source_sha256"] != WEIGHTS[dataset] or meta["source_config_hash"] != payload["config_hash"]
                or meta["input_geometry"]["file"] != "affine.csv"):
            raise ValueError("Published control export differs")
        check_file(out, meta["sha256"])
        check_file(destination / "affine.csv", meta["input_geometry"]["sha256"])
        published = torch.load(out, map_location="cpu", weights_only=True)
        embedded = {k: v for k, v in meta.items() if k not in ("sha256", "exported_file")}
        if published["metadata"] != embedded or meta["source_config"] != stored or meta["parent_hash"] != PARENT:
            raise ValueError("Published control metadata differs from the source checkpoint")
        original = payload["backbone_state_dict"]
        weights = published["state_dict"]
        if original.keys() != weights.keys() or any(not torch.equal(v.cpu(), weights[k]) for k, v in original.items()):
            raise ValueError("Published control backbone differs from the source checkpoint")
    return dict(completed=True, export=str(out), sha256=meta["sha256"], rows=meta["input_geometry"]["rows"])

