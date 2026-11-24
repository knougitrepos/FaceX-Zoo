"""Fixed, read-only inputs and paper references for the clean campaign."""

import json
import math
from pathlib import Path

from .clean_benchmarks import DATA, checked

HEADS = ("cosface", "adaface", "magface", "arcface")
DATASETS = ("tinyface", "scface", "survface")
SCOPES = ("paper_table1", "calibration_study")
PARENT_SOURCES = ("clean_parent", "legacy_parent_wd0")
PAPER_ROLES_SHA = "edc09ab3acfeb0a0be451b8fd619ed26794c1d882874f0e4fba2f67ffe6cf9bd"
LEGACY_SHA = "640ff9bf64d22462464d65cb42acb5cc86f3549c88ec568158169015d54a7213"
LEGACY_PARENT = "facex/diagnostics/parent_weight_decay_zero_v1/checkpoint/pretrain_arcface_seed8972/pretrain_mobilefacenet_1024d.pt"
ALIGNMENT_RUN = "ucface_alignment_arcface_s8972_20261112_v1"

HEAD_OPTIONS = {
    "cosface": dict(margin=0.3, scale=64.0),
    "adaface": dict(m=0.4, h=0.333, scale=64.0, t_alpha=0.01),
    "magface": dict(
        l_a=10.0,
        u_a=110.0,
        l_margin=0.45,
        u_margin=0.8,
        lamda=20.0,
        scale=64.0,
        detach_margin_norm=True,
        official_monotonic_fallback=True,
    ),
    "arcface": dict(margin_arc=0.6, scale=64.0, margin_am=0.6 * math.sin(0.6)),
}
LOCAL_VARIANTS = {
    "magface": "Detached adaptive-margin norm; differentiable norm regularizer; monotonic m*sin(m) fallback.",
    "arcface": "Monotonic m*sin(m) fallback selected for the local training recipe.",
}
ALIGNED = {
    "scface": (
        50,
        800,
        "two_eye_alignment",
        "32a29b3156f4fdae9398172298ad06d56bbb706cd224519657974bdce3b2c3a0",
        "14850c37a4b9159cad464d9e9bbfe3658cbc14c4ddfb715328472a8e71e89716",
        2080,
    ),
    "tinyface": (
        2570,
        7804,
        "pfld_five_point_alignment",
        "def359ecf5babd120768dfa62e391217b2f2dac26cfaecb64ae3067c5521e2a0",
        "3c30b76b3193a325c2f55404215e70b8bdfedcfa8670217a053ecafedf80c5c8",
        169403,
    ),
}

# Percentage units. Table I VGGFace2, MobileFaceNet, fine-tuned rows without UCFace.
TABLE_I = {
    "cosface": [[73.16, 67.05], [83.30, 97.30, 96.90, 92.50], [28.51, 23.53, 19.33, 13.15]],
    "adaface": [[73.16, 67.32], [85.80, 97.35, 97.50, 93.55], [29.10, 24.15, 19.44, 13.42]],
    "magface": [[73.33, 67.52], [86.55, 97.35, 97.50, 93.80], [29.85, 24.80, 19.01, 13.87]],
    "arcface": [[73.39, 67.36], [83.60, 97.15, 96.80, 92.52], [26.73, 21.80, 18.84, 12.21]],
}
METRICS = {
    "tinyface": ("rank1_without_distractors", "rank1_with_distractors"),
    "scface": ("d1", "d2", "d3", "average"),
    "survface": ("0.20", "0.10", "0.05", "0.01"),
}
TABLE_XIII = dict(
    tinyface={"rank1_without_distractors": 55.58},
    scface=dict(d1=41.75, d2=91.75, d3=97.20, average=77.08),
    survface={"0.20": 5.01, "0.10": 1.12},
)


def protocol(root, dataset, scope):
    directory, version, role_hash = DATA[dataset]
    base = Path(root) / directory
    path = base / "protocols/shared_v1" / version
    if dataset == "survface" and scope == "paper_table1":
        path = base / "protocols/paper_baseline_v1/5319_full_fr_v1"
    record = json.loads((path / "protocol.json").read_bytes())
    expected_hash = PAPER_ROLES_SHA if dataset == "survface" and scope == "paper_table1" else role_hash
    if record["roles_sha256"] != expected_hash:
        raise ValueError("Protocol and pinned roles differ")
    checked(path / "roles.csv", expected_hash)
    if dataset == "survface" and record["num_classes"] != (5319 if scope == "paper_table1" else 4255):
        raise ValueError("SurvFace training split differs from its scope")
    return record


def geometry_source(model_root, dataset):
    path = Path(model_root) / "facex/exports" / ALIGNMENT_RUN / f"{dataset}_arcface_aligned_s8972/affine.csv"
    _, _, _, _, digest, count = ALIGNED[dataset]
    return checked(path, digest), dict(
        file="affine.csv", sha256=digest, rows=count, schema="raw_to_arc112_affine_v1"
    )


def downstream(data_root, model_root, dataset, scope):
    root = Path(data_root)
    record = protocol(root, dataset, scope)
    if dataset in ALIGNED:
        classes, images, name, digest, _, _ = ALIGNED[dataset]
        folder = Path(model_root) / "facex/diagnostics" / ALIGNMENT_RUN / "prepared" / dataset
        report_path = checked(folder / "report.json", digest)
        report = json.loads(report_path.read_bytes())
        if (
            report["status"],
            report["image_count"],
            report["class_count"],
            report["original_role_manifest_sha256"],
        ) != ("completed", images, classes, record["roles_sha256"]):
            raise ValueError("Aligned training manifest does not match the official role selection")
        manifest = "images.csv"
        policy = report["alignment_policy"]
    else:
        directory = (
            "paper_baseline_v1_resize_only" if scope == "paper_table1" else "shared_v1_boundary_mobilefacenet"
        )
        folder = root / "QMUL-SurvFace/processed" / directory
        classes, images, name = (
            (5319, 220888, "full_fr_resize_only")
            if scope == "paper_table1"
            else (4255, 176205, "resize_only")
        )
        report = json.loads((folder / "report.json").read_bytes())
        if not (
            (folder / "COMPLETED").is_file()
            and report["completed"]
            and report["failed"] == 0
            and report["prepared"] == images
        ):
            raise ValueError("Incomplete SurvFace preparation")
        hashes = (
            (
                "5bb88f70d3981c2d4c8e4d6ba90c75ddc8d5fc775e0c0b189b397f12a863ece5",
                "af7a7954c5cd7474ee55d998c8d55fb3412a4d5a915c994bf35230fea11e4ab5",
            )
            if scope == "paper_table1"
            else (
                "7944ae7116fcab841b1ad246c71239715c15175f3a60f066810b7bc179f4d99d",
                "54538cc3d0ef6208f8dbcd3e13e1ef638b5f41ec3c697291540daa9ad4aee522",
            )
        )
        if (report["train_list_sha256"], report["manifest_sha256"]) != hashes:
            raise ValueError("SurvFace prepared files differ from the frozen scope")
        manifest = "manifest.jsonl"
        policy = report["policy"]
    checked(folder / manifest, report["manifest_sha256"])
    checked(folder / "train_list.txt", report["train_list_sha256"])
    return dict(
        dataset=dataset,
        prepared=str(folder),
        train_hash=report["train_list_sha256"],
        manifest_hash=report["manifest_sha256"],
        manifest_file=manifest,
        images=images,
        classes=classes,
        preprocessing=dict(
            input_size=[112, 112],
            pixel_mean=127.5,
            pixel_std=128.0,
            color="BGR",
            alignment=policy["version"] if dataset in ALIGNED else "resize_only",
            alignment_policy=policy,
        ),
        data_recipe=name,
        protocol_record=record,
    )
