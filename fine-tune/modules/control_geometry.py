"""Create portable raw-image-to-112 affine matrices for the two fixed ArcFace controls."""
import csv
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from .checkpoint_manager import compute_sha256

TARGET = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
SOURCES = {
    "scface": ("scface_two_eye_arc112_v1", "fd3602ee6abd4205ba2f931ebfa3597e00faa5a6ead3ccf6063d2fe8209fe303"),
    "tinyface": ("tinyface_pfld_five_point_arc112_v1", "5cb8410672e89ab87047114ba2dd4491d0ef82c412d4952d66d7f98e23c578f7"),
}


def similarity(points):
    target = TARGET[:len(points)]
    x = points - points.mean(0)
    y = target - target.mean(0)
    denom = (x * x).sum()
    if denom <= 0 or not np.isfinite(points).all():
        raise ValueError("Degenerate landmarks")
    a = (x * y).sum() / denom
    b = (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]).sum() / denom
    rotation = np.array([[a, -b], [b, a]])
    return np.column_stack((rotation, target.mean(0) - rotation @ points.mean(0)))


def read_bgr(path):
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image, hashlib.sha256(data).hexdigest()


def tiny_landmarks(model, images):
    patches, boxes, offsets = [], [], []
    for im in images:
        h, w = im.shape[:2]
        box = int(max(h, w) * 1.2)
        x, y = w // 2 - box // 2, h // 2 - box // 2
        patch = im[max(y, 0):min(y + box, h), max(x, 0):min(x + box, w)]
        patch = cv2.copyMakeBorder(patch, max(-y, 0), max(y + box - h, 0),
                                   max(-x, 0), max(x + box - w, 0), cv2.BORDER_CONSTANT, value=0)
        patches.append(cv2.resize(patch, (112, 112)))
        boxes.append(box)
        offsets.append((x, y))
    batch = torch.as_tensor(np.stack(patches), device="cuda").permute(0, 3, 1, 2).float() / 255
    with torch.inference_mode():
        points = model(batch)[1].reshape(-1, 106, 2)[:, [104, 105, 54, 84, 90]].cpu().numpy()
    return [lm * size + offset for lm, size, offset in zip(points, boxes, offsets)]


def write_geometry(output, dataset, profile, policy, prepared):
    source = Path(policy["landmark_source"])
    version, digest = SOURCES[dataset]
    if (policy["version"] != version or policy["landmark_sha256"] != digest
            or compute_sha256(str(source)) != digest or policy["opencv"] != cv2.__version__):
        raise ValueError("Landmark source or OpenCV version differs from the training control")
    if dataset == "tinyface":
        sdk = Path(__file__).resolve().parents[2] / "face_sdk"
        sys.path[:0] = [str(sdk), str(sdk / "models/network_def")]
        try:
            model = torch.load(source, map_location="cpu", weights_only=False).eval().cuda()
        finally:
            del sys.path[:2]
    else:
        eyes = {}
        for line in source.read_text().splitlines():
            fields = line.split()
            if fields:
                eyes[fields[0]] = np.array(fields[1:5], dtype=np.float64).reshape(2, 2)
    with Path(profile["training_manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (prepared / "images.csv").open(encoding="utf-8-sig", newline="") as stream:
        train = list(csv.DictReader(stream))
    by_id = {r["image_id"]: r for r in rows}
    train_rows = [by_id[r["image_id"]] for r in train]
    if (len(by_id) != len(rows) or len(train_rows) != len({r["image_id"] for r in train})
            or {r["image_id"] for r in train_rows} != {r["image_id"] for r in rows if r["fr_train"] == "1"}):
        raise ValueError("Control training cohort and role manifest differ")
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
                points = ([eyes[p.stem] for p in paths] if dataset == "scface"
                          else tiny_landmarks(model, [im for im, _ in decoded]))
                for i, (row, (image, sha), lm) in enumerate(zip(part, decoded, points)):
                    matrix = similarity(lm)
                    if group is train_rows:
                        ref = train[start + i]
                        actual = cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_CUBIC)
                        expected, png_hash = read_bgr(prepared / ref["png"])
                        if sha != ref["source_sha256"] or png_hash != ref["png_sha256"] or not np.array_equal(actual, expected):
                            raise ValueError(f"Control training pixels differ: {row['image_id']}")
                    writer.writerow([row["image_id"], sha, *matrix.ravel().tolist()])
    return len(rows)

