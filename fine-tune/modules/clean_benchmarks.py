"""Frozen input panels for the clean VGGFace2 parent and final SurvFace ROC."""

import csv
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint_manager import compute_sha256
from .control_geometry import SOURCES, read_bgr, similarity, tiny_landmarks

DATA = {
    "scface": (
        "SCface_database",
        "dca1e3f6d9df4daf750a",
        "95a8c996353f34acb1de029b6f188b7963773216db275d170643020f006a1101",
    ),
    "tinyface": (
        "tinyface",
        "394221dea5af128174d4",
        "4b92e525a7c4171529577f452cbc8bc67992dc2c620997b78ac3335912d5d777",
    ),
    "survface": (
        "QMUL-SurvFace",
        "9aeb13b96df3ae0e1912",
        "be171ed212db0c8115230a91d80df9db3a93cd5ea017218626cbc37dcd8dce4c",
    ),
}
MILESTONES = (2, 5, 10, 18, 24)
LFW_ALIGNMENT = "pfld_center144_five_point_arc112_v1"


def checked(path, expected):
    if compute_sha256(str(path)) != expected:
        raise ValueError(f"Protocol/input checksum mismatch: {path}")
    return path


def evaluation_rows(root, dataset, *, distractors=False):
    directory, protocol, digest = DATA[dataset]
    base = root / directory
    path = checked(base / "protocols/shared_v1" / protocol / "roles.csv", digest)
    roles = {"test_gallery", "test_registered_probe"}
    if dataset == "survface":
        roles.add("test_unknown_probe")
    if dataset == "tinyface" and distractors:
        roles.add("test_gallery_distractor")
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["role"] in roles]
    if any(r["fr_train"] != "0" for r in rows):
        raise ValueError(f"Training image in evaluation panel: {dataset}")
    return base, rows


def lfw_pairs(root):
    path = checked(
        root / "LFW/protocols/shared_v1/pairs.txt",
        "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[0].split() != ["10", "300"] or len(lines) != 6001:
        raise ValueError("Expected LFW View 2, ten ordered folds of 600")
    images, pairs = {}, []
    for i, line in enumerate(lines[1:]):
        fields = line.split()
        same = len(fields) == 3
        if same != (i % 600 < 300) or len(fields) not in (3, 4):
            raise ValueError(f"Invalid LFW pair at {i}")
        a, j = fields[:2]
        b, k = (a, fields[2]) if same else fields[2:]
        indices = []
        for name, number in ((a, j), (b, k)):
            file = root / "LFW/lfw-deepfunneled/lfw-deepfunneled" / name / f"{name}_{int(number):04d}.jpg"
            indices.append(images.setdefault(file, len(images)))
        pairs.append((*indices, int(same)))
    return list(images), np.asarray(pairs)


@torch.inference_mode()
def lfw_landmarks(model, images):
    if any(img.shape != (250, 250, 3) for img in images):
        raise ValueError("LFW deepfunneled alignment requires 250x250 BGR images")
    patches = [cv2.resize(img[53:197, 53:197], (112, 112)) for img in images]
    batch = torch.from_numpy(np.stack(patches)).to(next(model.parameters()).device)
    points = model(batch.permute(0, 3, 1, 2).float().div_(255))[1].reshape(len(images), 106, 2)
    return points[:, [104, 105, 54, 84, 90]].cpu().numpy() * 144.0 + 53.0


def prepare_panels(root):
    """Prepare raw pixels once per process. No recognition model or old export is read."""
    start = time.perf_counter()
    cv2.setNumThreads(1)
    root = Path(root)
    sdk = Path(__file__).resolve().parents[2] / "face_sdk"
    pfld = checked(
        sdk / "models/face_alignment/face_alignment_1.0/face_landmark_pfld.pkl", SOURCES["tinyface"][1]
    )
    sys.path[:0] = [str(sdk), str(sdk / "models/network_def")]
    try:
        landmark_model = torch.load(pfld, map_location="cpu", weights_only=False).eval().cuda()
    finally:
        del sys.path[:2]
    eye_file = checked(root / "SCface_database/all.txt", SOURCES["scface"][1])
    eyes = {}
    for line in eye_file.read_text().splitlines():
        fields = line.split()
        if fields:
            eyes[fields[0]] = np.array(fields[1:5], dtype=float).reshape(2, 2)
    panels = {}
    digest = hashlib.sha256()
    for ds in ("lfw", "scface", "tinyface"):
        if ds == "lfw":
            paths, labels = lfw_pairs(root)
        else:
            base, labels = evaluation_rows(root, ds)
            paths = [base / r["image_path"] for r in labels]
        pixels = np.empty((len(paths), 112, 112, 3), dtype=np.uint8)
        with ThreadPoolExecutor(6) as pool:
            for offset in range(0, len(paths), 64):
                chunk = paths[offset : offset + 64]
                decoded = list(pool.map(read_bgr, chunk))
                imgs = [img for img, _ in decoded]
                if ds == "scface":
                    landmarks = [eyes[p.stem] for p in chunk]
                elif ds == "tinyface":
                    landmarks = tiny_landmarks(landmark_model, imgs)
                else:
                    landmarks = lfw_landmarks(landmark_model, imgs)
                for j, (img, sha) in enumerate(decoded):
                    digest.update(bytes.fromhex(sha))
                    pixels[offset + j] = cv2.warpAffine(
                        img, similarity(landmarks[j]), (112, 112), flags=cv2.INTER_CUBIC
                    )
        panels[ds] = (pixels, labels)
    return panels, dict(
        raw_images_sha256=digest.hexdigest(),
        lfw_alignment=LFW_ALIGNMENT,
        prepare_seconds=time.perf_counter() - start,
        unique_images={ds: len(item[0]) for ds, item in panels.items()},
    )


@torch.inference_mode()
def embed(backbone, pixels):
    batch = torch.from_numpy(np.asarray(pixels)).to(next(backbone.parameters()).device)
    x = batch.permute(0, 3, 1, 2).contiguous().float().sub_(127.5).div_(128.0)
    feats = backbone(x)
    if not feats.isfinite().all() or (feats.norm(dim=1) == 0).any():
        raise ValueError("Invalid recognition embeddings")
    return F.normalize(feats).cpu().numpy()


def lfw_accuracy(vectors, pairs):
    # Pairwise products in bounded chunks avoid two extra full embedding arrays.
    scores = np.concatenate(
        [np.einsum("ij,ij->i", vectors[c[:, 0]], vectors[c[:, 1]]) for c in np.array_split(pairs, 10)]
    )
    same = pairs[:, 2].astype(bool)
    fold = np.arange(len(pairs)) // 600
    results = []
    for f in range(10):
        train = fold != f
        values, labels = scores[train], same[train]
        order = np.argsort(values)
        # Accept score >= threshold. Candidate cuts occur only after complete tie groups.
        v, y = values[order], labels[order]
        cuts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1, len(v)]
        correct = y.sum() + np.r_[0, np.cumsum(np.where(y, -1, 1))]
        cut = cuts[np.argmax(correct[cuts])]
        threshold = float(v[cut]) if cut < len(v) else float(np.nextafter(v[-1], np.float32(np.inf)))
        hits = int(np.count_nonzero((scores[~train] >= threshold) == same[~train]))
        results.append(dict(correct=hits, pairs=600, threshold=threshold))
    hits = sum(r["correct"] for r in results)
    return dict(correct=hits, pairs=6000, accuracy=hits / 6000, folds=results)


def rank_one(vectors, rows):
    ids = np.asarray([r["identity_id"] for r in rows])
    gallery = np.array([r["role"] == "test_gallery" for r in rows])
    g_ids, q_ids = ids[gallery], ids[~gallery]
    if not set(q_ids) <= set(g_ids):
        raise ValueError("Probe lacks a matching gallery identity")
    hits = []
    gallery_vectors = vectors[gallery]
    for qs in np.array_split(np.flatnonzero(~gallery), max(1, (len(q_ids) + 255) // 256)):
        scores = vectors[qs] @ gallery_vectors.T
        match = ids[qs, None] == g_ids[None, :]
        hits.extend(np.where(match, scores, -np.inf).max(1) > np.where(match, -np.inf, scores).max(1))
    hits = np.array(hits)
    distances = np.array([r["distance"] for i, r in enumerate(rows) if not gallery[i]])

    def count(mask):
        return dict(correct=int(hits[mask].sum()), probes=int(mask.sum()), rank1=float(hits[mask].mean()))

    return dict(
        **count(np.ones(len(hits), bool)),
        gallery=len(g_ids),
        distances={d: count(distances == d) for d in sorted(set(distances)) if d},
    )


def monitor(backbone, panels, *, raw_file=None):
    start = time.perf_counter()
    was_training = backbone.training
    backbone.eval()
    result, raw = {}, {}
    try:
        for ds, (pixels, labels) in panels.items():
            vectors = np.concatenate([embed(backbone, pixels[i : i + 64]) for i in range(0, len(pixels), 64)])
            result[ds] = lfw_accuracy(vectors, labels) if ds == "lfw" else rank_one(vectors, labels)
            if raw_file is not None:
                raw[ds] = vectors
        if raw_file is not None:
            np.savez(raw_file, **raw)
    finally:
        backbone.train(was_training)
    return dict(metrics=result, seconds=time.perf_counter() - start)


def reference_gaps(metrics, epoch):
    # Signed percentage points, for display only. No checkpoint is selected here.
    gaps = {
        "scface_vs_77.08_pp": 100 * metrics["scface"]["rank1"] - 77.08,
        "tinyface_vs_55.58_pp": 100 * metrics["tinyface"]["rank1"] - 55.58,
    }
    if epoch >= 10:
        ref = 99.4 if epoch == 24 else 99.0
        gaps["lfw_reference_pct"] = ref
        gaps["lfw_gap_pp"] = 100 * metrics["lfw"]["accuracy"] - ref
    return gaps


def survface_roc(best, genuine, ranks, registered):
    unknown = np.asarray(best, dtype=np.float64)[~registered]
    genuine = np.asarray(genuine, dtype=np.float64)[registered]
    rank_ok = ranks[registered] <= 20
    if (
        not len(unknown)
        or not len(genuine)
        or not np.isfinite(unknown).all()
        or not np.isfinite(genuine).all()
    ):
        raise ValueError("Invalid SurvFace score population")
    points = {}
    descending = np.sort(unknown)[::-1]
    for target in (0.20, 0.10, 0.05, 0.01):
        budget = int(np.floor(target * len(unknown)))
        threshold = float(np.nextafter(descending[budget], np.inf))
        fp = int((unknown >= threshold).sum())
        tp = int((rank_ok & (genuine >= threshold)).sum())
        points[f"{target:.2f}"] = dict(
            target_fpir=target,
            threshold=threshold,
            fp=fp,
            unknown=len(unknown),
            realized_fpir=fp / len(unknown),
            tp=tp,
            registered=len(genuine),
            tpir_at_20=tp / len(genuine),
            rank_failures=int((~rank_ok).sum()),
            threshold_failures=int(rank_ok.sum()) - tp,
        )
    return points


@torch.inference_mode()
def evaluate_survface(backbone, data_root, raw_file=None):
    """Final Origin ROC. Mean unit vectors per identity, unit-normalized before search."""
    base, rows = evaluation_rows(Path(data_root), "survface")
    gallery = [r for r in rows if r["role"] == "test_gallery"]
    probes = [r for r in rows if r["role"] != "test_gallery"]
    keys = sorted({r["identity_id"] for r in gallery})
    lookup = {k: i for i, k in enumerate(keys)}
    registered = np.array([r["role"] == "test_registered_probe" for r in probes])
    if (len(gallery), len(keys), int(registered.sum()), int((~registered).sum())) != (
        60294,
        3000,
        60423,
        121736,
    ):
        raise ValueError("Unexpected official SurvFace population")
    if any((r["identity_id"] in lookup) != bool(m) for r, m in zip(probes, registered)):
        raise ValueError("Invalid probe identity roles")

    def decode(row):
        image, _ = read_bgr(base / row["image_path"])
        return cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)

    backbone.eval()
    best = np.empty(len(probes), dtype=np.float32)
    genuine = np.full(len(probes), np.nan, dtype=np.float32)
    ranks = np.zeros(len(probes), dtype=np.int32)
    with ThreadPoolExecutor(8) as pool:

        def batches(group):
            for i in range(0, len(group), 128):
                part = group[i : i + 128]
                yield i, part, embed(backbone, np.stack(list(pool.map(decode, part))))

        templates = np.zeros((len(keys), 1024), dtype=np.float32)
        for _, part, vectors in batches(gallery):
            np.add.at(templates, [lookup[r["identity_id"]] for r in part], vectors)
        norms = np.linalg.norm(templates, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Zero gallery template")
        templates /= norms
        for start, part, vectors in batches(probes):
            scores = vectors @ templates.T
            best[start : start + len(part)] = scores.max(1)
            for i, row in enumerate(part):
                if registered[start + i]:
                    label = lookup[row["identity_id"]]
                    g = scores[i, label]
                    genuine[start + i] = g
                    # All nonmatching ties count ahead of the genuine.
                    ranks[start + i] = np.count_nonzero(scores[i] >= g)
    if raw_file is not None:
        np.savez(
            raw_file,
            best=best,
            genuine=genuine,
            ranks=ranks,
            registered=registered,
            image_ids=np.array([r["image_id"] for r in probes]),
        )
    return {
        **survface_roc(best, genuine, ranks, registered),
        "unknown_cos_ge_095": int(np.count_nonzero(best[~registered] >= np.float32(0.95))),
    }


def rank_one_distractors(vectors, rows, distractors, device):
    """Compute conservative Rank-1 with the full distractor set in score blocks."""
    gallery = np.array([r["role"] == "test_gallery" for r in rows])
    ids = np.asarray([r["identity_id"] for r in rows])
    hits = []
    g_vectors = torch.as_tensor(vectors[gallery], device=device)
    for probe in np.array_split(np.flatnonzero(~gallery), max(1, (sum(~gallery) + 127) // 128)):
        queries = torch.as_tensor(vectors[probe], device=device)
        scores = queries @ g_vectors.T
        match = torch.as_tensor(ids[probe, None] == ids[gallery][None, :], device=device)
        genuine = scores.masked_fill(~match, -torch.inf).amax(1)
        impostor = scores.masked_fill(match, -torch.inf).amax(1)
        for start in range(0, len(distractors), 16384):
            block = torch.as_tensor(distractors[start : start + 16384], device=device)
            impostor = torch.maximum(impostor, (queries @ block.T).amax(1))
        hits.extend((genuine > impostor).cpu().tolist())
    return dict(
        correct=sum(hits),
        probes=len(hits),
        rank1=sum(hits) / len(hits),
        gallery=int(gallery.sum()),
        distractors=len(distractors),
    ), np.asarray(hits)


def evaluate_aligned(backbone, root, dataset, affine_path, raw_file=None, *, include_distractors=True):
    base, rows = evaluation_rows(Path(root), dataset, distractors=include_distractors)
    with Path(affine_path).open(encoding="utf-8-sig", newline="") as f:
        matrices = {
            r["image_id"]: (
                r["source_sha256"],
                np.array([float(r[k]) for k in ("a", "b", "tx", "c", "d", "ty")]).reshape(2, 3),
            )
            for r in csv.DictReader(f)
        }
    regular = [r for r in rows if r["role"] != "test_gallery_distractor"]
    distractors = [r for r in rows if r["role"] == "test_gallery_distractor"]
    counts = (len(regular), len(distractors), sum(r["role"] == "test_gallery" for r in regular))
    if (
        counts
        != {"tinyface": (8171, 153428 if include_distractors else 0, 4443), "scface": (1280, 0, 80)}[dataset]
    ):
        raise ValueError("Unexpected aligned official cohort")

    def decode(row):
        img, digest = read_bgr(base / row["image_path"])
        expected, matrix = matrices[row["image_id"]]
        if digest != expected:
            raise ValueError(f"Image differs from frozen alignment source: {row['image_id']}")
        return cv2.warpAffine(img, matrix, (112, 112), flags=cv2.INTER_CUBIC)

    backbone.eval()
    with ThreadPoolExecutor(8) as pool:

        def extract(batch_rows):
            array = np.empty((len(batch_rows), 1024), dtype=np.float32)
            for start in range(0, len(batch_rows), 128):
                batch = list(pool.map(decode, batch_rows[start : start + 128]))
                array[start : start + len(batch)] = embed(backbone, np.stack(batch))
            return array

        regular_vectors = extract(regular)
        result = dict(without_distractors=rank_one(regular_vectors, regular))
        if distractors:
            distractor_vectors = extract(distractors)
            result["with_distractors"], hits = rank_one_distractors(
                regular_vectors, regular, distractor_vectors, next(backbone.parameters()).device
            )
        if raw_file is not None:
            arrays = dict(vectors=regular_vectors, image_ids=[r["image_id"] for r in regular])
            if distractors:
                arrays.update(
                    distractor_vectors=distractor_vectors,
                    distractor_ids=[r["image_id"] for r in distractors],
                    distractor_rank1_hits=hits,
                )
            np.savez(raw_file, **arrays)
    return result
