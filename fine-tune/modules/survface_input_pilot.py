"""Paired SurvFace development experiment on native training crops."""
import csv
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline_policy import code_fingerprints, DEFAULT_AUGMENTATION
from .checkpoint_manager import compute_sha256, verify_parent_checkpoint
from .dataset import apply_face_augmentation
from .model_builder import FaceRecognitionModel
from .backbone_factory import build_backbone
from .target_margin import MagFace, target_sine
from .trainer import set_seed


SEED = 8972
PARENT_HASH = "640ff9bf64d22462464d65cb42acb5cc86f3549c88ec568158169015d54a7213"
ROLES_HASH = "be171ed212db0c8115230a91d80df9db3a93cd5ea017218626cbc37dcd8dce4c"
ARMS = ("stretch", "aspect_pad")
SIZES = {"train": (256, 16), "registered": (128, 8), "fit": (128, 8), "unknown": (128, 8)}
GATE = "epoch 10: paired registered-ID bootstrap 95% lower bound of pad minus stretch TPIR > 0 and held-out pad FPIR <= 0.10"


def ordered(values):
    return sorted(values, key=lambda v: hashlib.sha256(f"{SEED}/{v}".encode()).digest())


def cohort(roles):
    if compute_sha256(roles) != ROLES_HASH:
        raise ValueError("SurvFace shared_v1 roles hash mismatch")
    groups = defaultdict(list)
    with Path(roles).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["fr_train"] == "1":
                identity = row["source_identity_id"]
                rel = Path(row["image_path"])
                if (rel.is_absolute() or len(rel.parts) != 3 or rel.parts[:2] != ("training_set", identity)
                        or ".." in rel.parts or row["role"].startswith("test_")):
                    raise ValueError("Unexpected native training source")
                groups[identity].append(rel.as_posix())
    pool = ordered(k for k, paths in groups.items() if len(set(paths)) >= 16)
    records = []
    for role, (people, images) in SIZES.items():
        if len(pool) < people:
            raise ValueError("Insufficient eligible identities")
        for label, identity in enumerate(pool[:people]):
            for i, path in enumerate(ordered(set(groups[identity]))[:images]):
                actual = ("gallery" if i < 2 else "probe") if role == "registered" else role
                records.append((actual, identity, label, path))
        pool = pool[people:]
    return records


def transform(image, arm):
    if arm == "stretch":
        return cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
    if arm != "aspect_pad":
        raise ValueError(arm)
    h, w = image.shape[:2]
    scale = 112 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    canvas = np.full((112, 112, 3), 128, dtype=np.uint8)
    y, x = (112 - nh) // 2, (112 - nw) // 2
    canvas[y:y+nh, x:x+nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return canvas


class PilotMagFace(MagFace):
    """Stage 7e detached margin and monotonic fallback, confined to this pilot."""
    def _forward_fp32(self, feats, labels):
        norm = feats.norm(dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m = self.calc_margin(norm.detach()).flatten()
        cosine = (F.normalize(feats) @ F.normalize(self.weight, dim=0)).clamp(-1, 1)
        rows = torch.arange(len(feats), device=feats.device)
        target = cosine[rows, labels]
        angular = target * m.cos() - target_sine(target) * m.sin()
        logits = cosine.clone()
        logits[rows, labels] = torch.where(target > (math.pi-m).cos(), angular, target-m*m.sin())
        return logits * self.scale, self.lamda * (norm / self.u_a**2 + 1 / norm)


def batch_tensor(images):
    return (torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().cuda().float() - 127.5) / 128


def train_arm(parent, records, images):
    set_seed(SEED)
    backbone = build_backbone({"type": "MobileFaceNet", "feat_dim": 1024, "dropout_rate": .6})
    backbone.load_state_dict(parent, strict=True)
    model = FaceRecognitionModel(backbone, PilotMagFace(1024, SIZES["train"][0], scale=64), "magface").cuda()
    backbone.bn.bias.requires_grad_(False).zero_()
    bn2ds = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    bn = [p for m in bn2ds for p in m.parameters()] + [backbone.bn.weight]
    excluded = {id(p) for p in bn + [backbone.bn.bias]}
    conv = [p for p in backbone.parameters() if id(p) not in excluded]
    for p in conv:
        p.requires_grad_(False)
    opt = torch.optim.Adam([dict(params=model.head.parameters(), lr=.0002), dict(params=bn, lr=.0001)])
    indices = [i for i, r in enumerate(records) if r[0] == "train"]
    losses = []
    for epoch in range(1, 11):
        model.train()
        if epoch == 4:
            for p in conv:
                p.requires_grad_(True)
            opt.add_param_group(dict(params=conv, lr=.0001))
        if epoch >= 4:
            for m in bn2ds:
                m.requires_grad_(False).eval()
        order = np.random.default_rng(SEED + epoch).permutation(indices)
        step_losses = []
        for chunk in np.split(order, len(order) // 64):
            xs = [apply_face_augmentation(images[i], True, random.Random((SEED << 40) ^ (epoch << 32) ^ int(i)))
                  for i in chunk]
            ys = torch.tensor([records[i][2] for i in chunk], device="cuda")
            opt.zero_grad(set_to_none=True)
            logits, reg = model(batch_tensor(xs), ys)
            loss = F.cross_entropy(logits, ys) + reg.mean()
            if not loss.isfinite():
                raise FloatingPointError("Nonfinite pilot loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not torch.nn.utils.get_total_norm(gradients, float("inf")).isfinite():
                raise FloatingPointError("Nonfinite pilot gradient")
            opt.step()
            step_losses.append(loss.item())
        losses.append(float(np.mean(step_losses)))
    return backbone.eval(), losses


def evaluate(vectors, records):
    roles = np.array([r[0] for r in records])
    labels = np.array([r[2] for r in records])
    gallery = vectors[roles == "gallery"]
    gallery_labels = labels[roles == "gallery"]
    templates = np.stack([gallery[gallery_labels == i].mean(0) for i in sorted(set(gallery_labels))])
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("Invalid gallery or embedding")
    scores = vectors @ (templates / norms).T
    fit = scores[roles == "fit"].max(1).astype(np.float64)
    budget = len(fit) // 10
    cutoff = np.nextafter(np.sort(fit)[-(budget+1)], np.inf)
    unknown = scores[roles == "unknown"].max(1).astype(np.float64)
    reg = scores[roles == "probe"]
    ids = labels[roles == "probe"]
    genuine = reg[np.arange(len(reg)), ids]
    rank = np.sum(reg >= genuine[:, None], axis=1)  # conservative ranking of ties
    hits = (rank <= 20) & (genuine.astype(np.float64) >= cutoff)
    return dict(threshold=float(cutoff), fit_fp=int((fit >= cutoff).sum()), fit_n=len(fit),
                fp=int((unknown >= cutoff).sum()), unknown_n=len(unknown), registered_n=len(reg),
                tp=int(hits.sum()), tpir=float(hits.mean()), realized_fpir=float((unknown >= cutoff).mean()),
                rank1=float((rank == 1).mean()), top20=float((rank <= 20).mean()),
                rank_failure=int((rank > 20).sum()), threshold_failure=int(((rank <= 20) & ~hits).sum()),
                unknown_ge_095=int((unknown >= .95).sum())), hits, ids


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_pilot(dataset_root, parent_checkpoint, output, *, keep_raw_results=False):
    root, output = Path(dataset_root) / "QMUL-SurvFace", Path(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    roles = root / "protocols/shared_v1/9aeb13b96df3ae0e1912/roles.csv"
    records = cohort(roles)
    payload, parent_hash = verify_parent_checkpoint(str(parent_checkpoint), expected_hash=PARENT_HASH,
                                                    require_completed=True, expected_seed=SEED, target_epoch=40)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 100 * 2**20:
        raise OSError("Pilot requires 100 MiB of free space")
    output.mkdir(exist_ok=False)
    plan = dict(schema="survface-input-pilot-v1", seed=SEED, parent_path=str(parent_checkpoint),
                parent_sha256=parent_hash, roles_sha256=ROLES_HASH, arms=ARMS, sizes=SIZES,
                epochs=10, warmup_epochs=3, batch=64, head_lr=.0002, backbone_lr=.0001,
                augmentation=DEFAULT_AUGMENTATION, gate=GATE, fit_target_fpir=.10,
                keep_raw_results=keep_raw_results, source_sha256=compute_sha256(__file__),
                facex_sources=code_fingerprints(), device=torch.cuda.get_device_name(),
                torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__, opencv=cv2.__version__)
    raw = []
    fingerprint = hashlib.sha256()
    for row in records:
        content = (root / row[3]).read_bytes()
        fingerprint.update(json.dumps(row).encode() + hashlib.sha256(content).digest())
        img = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode {row[3]}")
        raw.append(img)
    plan["cohort_and_image_bytes_sha256"] = fingerprint.hexdigest()
    write_json(output / "plan.json", plan)
    metrics, hit_vectors = {}, {}
    dev = [i for i, r in enumerate(records) if r[0] != "train"]
    dev_records = [records[i] for i in dev]
    for arm in ARMS:
        images = [transform(img, arm) for img in raw]
        backbone, losses = train_arm(payload["backbone_state_dict"], records, images)
        parts = []
        with torch.inference_mode():
            for j in range(0, len(dev), 64):
                feats = backbone(batch_tensor([images[i] for i in dev[j:j+64]]))
                if not feats.isfinite().all() or (feats.norm(dim=1) == 0).any():
                    raise ValueError("Invalid embedding")
                parts.append(F.normalize(feats).cpu().numpy())
        vectors = np.concatenate(parts)
        metrics[arm], hit_vectors[arm], ids = evaluate(vectors, dev_records)
        metrics[arm]["train_epoch_loss"] = losses
        if keep_raw_results:
            np.savez(output / f"{arm}_dev.npz", unit=vectors, rows=np.array(dev_records, dtype=str))
            torch.save(backbone.state_dict(), output / f"{arm}_backbone.pt")
        write_json(output / f"{arm}.json", metrics[arm])
        del backbone, images
    diff = hit_vectors["aspect_pad"].astype(float) - hit_vectors["stretch"]
    by_id = [diff[ids == i].mean() for i in sorted(set(ids))]
    samples = np.random.default_rng(SEED).choice(by_id, (2000, len(by_id))).mean(1)
    ci = np.quantile(samples, [.025, .975])
    passed = ci[0] > 0 and metrics["aspect_pad"]["realized_fpir"] <= .1
    summary = dict(status="completed", planned=2, completed=2, arms=metrics, delta_tpir_pp=float(100*diff.mean()),
                   paired_registered_id_ci95_pp=(100*ci).tolist(), gate=GATE, pass_gate=bool(passed),
                   next_step="confirm_on_larger_development_cohort" if passed else "hold_expansion",
                   plan_sha256=compute_sha256(output / "plan.json"), guide="HOW_TO_READ.md")
    shutil.copyfile(Path(__file__).resolve().parents[1] / "notebooks/pilots/HOW_TO_READ.md", output / "HOW_TO_READ.md")
    write_json(output / "summary.json", summary)
    files = [output / name for name in ("summary.json", "plan.json", "HOW_TO_READ.md")]
    write_json(output / "bundle.json", {"recommended_order": [p.name for p in files],
        "bytes": sum(p.stat().st_size for p in files),
        "estimated_tokens": sum(len(p.read_text(encoding="utf-8")) for p in files),
        "estimate_basis": "conservative planning estimate, one token per character",
        "sha256": {p.name: compute_sha256(p) for p in files}})
    (output / "COMPLETED").write_text(compute_sha256(output / "summary.json"), encoding="ascii")
    return summary


def read_summary(output):
    folder = Path(output)
    if (folder / "COMPLETED").read_text().strip() != compute_sha256(folder / "summary.json"):
        raise ValueError("Invalid pilot completion marker")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if (summary["status"] != "completed" or summary["completed"] != len(ARMS)
            or set(summary["arms"]) != set(ARMS) or summary["plan_sha256"] != compute_sha256(folder / "plan.json")):
        raise ValueError("Invalid pilot summary or plan")
    return summary
