"""Fixed identity-disjoint VGGFace2 development diagnostics for parent controls.

This is a local diagnostic protocol, not an official VGGFace2 benchmark and not
the final SurvFace test. No training parameters or checkpoint selection are read
when selecting images.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import cv2
import numpy as np
import torch

from .checkpoint_manager import compute_sha256, load_checkpoint
from .notebook_workflow import write_json
from .shared_storage import artifact_lock

SCHEMA = "facex-parent-vggface2-fixed-diagnostic-v1"


def _order(value, seed):
    return hashlib.sha256(f"parent-wd-diagnostic/{seed}/{value}".encode()).hexdigest()


def select_cohort(metadata, test_list, *, seed=8972, identity_count=200, images_per_identity=10, gallery_per_identity=2):
    if not (identity_count >= 2 and 1 <= gallery_per_identity < images_per_identity):
        raise ValueError("Require multiple identities and separate gallery/probe images")
    with Path(metadata).open(encoding="utf-8-sig", newline="") as stream:
        rows = [{k.strip(): v.strip() for k,v in row.items()} for row in csv.DictReader(stream, skipinitialspace=True)]
    train = {row["Class_ID"] for row in rows if row["Flag"] == "1"}
    test = {row["Class_ID"] for row in rows if row["Flag"] == "0"}
    if train & test:
        raise ValueError("Training/test identities overlap")
    by_identity = {}
    for name in Path(test_list).read_text(encoding="utf-8").splitlines():
        name = name.strip().replace("\\", "/").removeprefix("test/")
        parts = name.split("/")
        if len(parts) != 2 or any(x in ("", ".", "..") for x in parts) or parts[0] not in test or parts[0] in train:
            raise ValueError(f"Invalid or training identity in test list: {name}")
        by_identity.setdefault(parts[0], []).append(name)
    eligible = [identity for identity,names in by_identity.items() if len(names) >= images_per_identity]
    if len(eligible) < identity_count:
        raise ValueError("Not enough identities for the fixed diagnostic")
    chosen = sorted(eligible, key=lambda name: _order(name, seed))[:identity_count]
    selected = []
    for identity in chosen:
        names = by_identity[identity]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate test image")
        for i,name in enumerate(sorted(names, key=lambda name: _order(name, seed))[:images_per_identity]):
            selected.append(dict(source=name, identity=identity, role="gallery" if i < gallery_per_identity else "probe"))
    return selected


def prepare_validation(base_dir, dataset_root, output_root, *, seed=8972, identity_count=200,
                       images_per_identity=10, gallery_per_identity=2):
    base, data, out = Path(base_dir), Path(dataset_root), Path(output_root)
    metadata, image_list = data / "meta/identity_meta.csv", data / "meta/test_list.txt"
    images_tar, landmarks_tar = data / "data/vggface2_test.tar.gz", data / "meta/bb_landmark.tar.gz"
    selection = select_cohort(metadata, image_list, seed=seed, identity_count=identity_count,
                              images_per_identity=images_per_identity, gallery_per_identity=gallery_per_identity)
    plan = dict(schema=SCHEMA, seed=seed, identity_count=identity_count, images_per_identity=images_per_identity,
        gallery_per_identity=gallery_per_identity, split="official VGGFace2 test identities; local diagnostic only",
        selection_method="SHA-256 order independent of all model outputs", selection=selection,
        sources={str(p):compute_sha256(str(p)) for p in (metadata, image_list, images_tar, landmarks_tar)},
        landmarks_member="bb_landmark/loose_landmark_test.csv", parent_training_identity_overlap=0)
    with artifact_lock(out):
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "plan.json", plan, immutable=True)
        if (out / "COMPLETED").is_file():
            verify_validation(out)
            return out / "protocol.json"
        sys.path.insert(0, str(base / "face_sdk"))
        from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
        cropper = FaceRecImageCropper()
        wanted = {row["source"]: i for i,row in enumerate(selection)}
        points = {}
        with tarfile.open(landmarks_tar, "r|gz") as archive:
            for member in archive:
                if member.name != plan["landmarks_member"]:
                    continue
                reader = csv.reader(line.decode("utf-8-sig") for line in archive.extractfile(member))
                if next(reader) != ["NAME_ID"] + [f"P{i}{axis}" for i in range(1,6) for axis in "XY"]:
                    raise ValueError("Unexpected test landmarks header")
                for row in reader:
                    name = row[0].removeprefix("test/")
                    if not name.endswith(".jpg"):
                        name += ".jpg"
                    if name in wanted:
                        coords = np.asarray(row[1:], dtype=np.float64)
                        if len(coords) != 10 or not np.isfinite(coords).all() or name in points:
                            raise ValueError("Invalid/duplicate test landmarks")
                        points[name] = coords.tolist()
        if set(points) != set(wanted):
            raise ValueError("Selected test images lack landmarks")
        journal = out / "prepared.jsonl"
        records = {}
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["source"] in records or row["source"] not in wanted:
                    raise ValueError("Invalid partial diagnostic manifest")
                if compute_sha256(row["output"]) != row["output_sha256"]:
                    raise ValueError("Partial diagnostic image changed")
                records[row["source"]] = row
        with journal.open("a", encoding="utf-8", buffering=1) as record_file, tarfile.open(images_tar, "r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix("./").removeprefix("test/")
                if not member.isfile() or name not in wanted or name in records:
                    continue
                raw = archive.extractfile(member).read()
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Diagnostic image decode failed: {name}")
                crop = cropper.crop_image_by_mat(image, points[name])
                if crop.shape != (112,112,3) or crop.dtype != np.uint8:
                    raise ValueError("Invalid cropper output")
                target = out / "images" / f"{wanted[name]:06d}.png"
                target.parent.mkdir(exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"Unjournaled output requires inspection: {target}")
                success, encoded = cv2.imencode(".png", crop)
                if not success:
                    raise ValueError("PNG encoding failed")
                target.write_bytes(encoded.tobytes())
                row = {**selection[wanted[name]], "output":str(target.resolve()),
                       "output_sha256":compute_sha256(str(target)), "source_sha256":hashlib.sha256(raw).hexdigest(),
                       "landmarks":points[name]}
                record_file.write(json.dumps(row)+"\n")
                records[name] = row
                if len(records) % 500 == 0:
                    print(f"Fixed VGGFace2 diagnostic: {len(records)}/{len(selection)} prepared", flush=True)
        if set(records) != set(wanted):
            raise ValueError("Selected diagnostic images are missing from the test archive")
        manifest = [records[row["source"]] for row in selection]
        write_json(out / "images.json", manifest, immutable=True)
        protocol = {k:v for k,v in plan.items() if k != "selection"}
        protocol.update(status="completed", image_count=len(manifest), gallery_count=identity_count*gallery_per_identity,
            probe_count=identity_count*(images_per_identity-gallery_per_identity),
            plan_sha256=compute_sha256(str(out / "plan.json")), manifest_sha256=compute_sha256(str(out / "images.json")),
            preprocessing=dict(method="provided_test_five_point_arcface112", color="BGR", pixel_mean=127.5, pixel_std=128.0,
                cropper_sha256=compute_sha256(str(base / "face_sdk/core/image_cropper/arcface_cropper/FaceRecImageCropper.py"))),
            source_module_sha256=compute_sha256(__file__))
        write_json(out / "protocol.json", protocol, immutable=True)
        (out / "COMPLETED").write_text(compute_sha256(str(out / "protocol.json")), encoding="utf-8")
        return out / "protocol.json"


def verify_validation(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if ((root / "COMPLETED").read_text(encoding="utf-8") != compute_sha256(str(root / "protocol.json"))
            or protocol["status"] != "completed"
            or protocol["manifest_sha256"] != compute_sha256(str(root / "images.json"))
            or protocol["plan_sha256"] != compute_sha256(str(root / "plan.json"))):
        raise ValueError("Incomplete or modified diagnostic protocol")
    images = json.loads((root / "images.json").read_text(encoding="utf-8"))
    if len(images) != protocol["image_count"]:
        raise ValueError("Diagnostic image count differs")
    for row in images:
        if compute_sha256(row["output"]) != row["output_sha256"]:
            raise ValueError("Diagnostic image changed")
    return protocol, images


def recognition_metrics(embeddings, images):
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    ids = np.asarray([row["identity"] for row in images])
    is_gallery = np.asarray([row["role"] == "gallery" for row in images])
    identities = list(dict.fromkeys(ids[is_gallery]))
    gallery, probes = unit[is_gallery], unit[~is_gallery]
    scores = probes @ gallery.T
    id_scores = np.stack([scores[:,ids[is_gallery] == identity].max(axis=1) for identity in identities], axis=1)
    truth = np.asarray([identities.index(identity) for identity in ids[~is_gallery]])
    genuine = id_scores[np.arange(len(probes)), truth]
    non_genuine = np.arange(len(identities))[None,:] != truth[:,None]
    ranks = 1 + ((id_scores >= genuine[:,None]) & non_genuine).sum(axis=1)
    identity_results = [dict(identity=identity, probes=int((ids[~is_gallery] == identity).sum()),
        rank1_correct=int((ranks[ids[~is_gallery] == identity] <= 1).sum())) for identity in identities]
    mass = (unit.astype(np.float64)**2).mean(axis=0)
    top = np.argsort(-mass)[:5]
    total_pair_sum = float(np.square(unit.astype(np.float64).sum(axis=0)).sum() - np.square(unit.astype(np.float64)).sum())
    within_sum, within_n = 0.0, 0
    for identity in identities:
        group = unit[ids == identity].astype(np.float64)
        within_sum += float(np.square(group.sum(axis=0)).sum() - np.square(group).sum())
        within_n += len(group)*(len(group)-1)
    different_n = len(unit)*(len(unit)-1)-within_n
    return dict(probes=len(probes), gallery_images=len(gallery), identities=len(identities),
        rank1_correct=int((ranks <= 1).sum()), rank1=float((ranks <= 1).mean()),
        rank5_correct=int((ranks <= 5).sum()), rank5=float((ranks <= 5).mean()),
        rank20_correct=int((ranks <= 20).sum()), rank20=float((ranks <= 20).mean()),
        tie_policy="Other identity ties rank ahead of genuine identity; gallery scores max-pooled by identity",
        different_identity_pair_mean_cosine=(total_pair_sum-within_sum)/different_n,
        top_component_energy=[dict(index=int(i), mean_squared_unit_component=float(mass[i])) for i in top],
        raw_norm_median=float(np.median(np.linalg.norm(embeddings,axis=1))),
        identity_results=identity_results), ranks


def evaluate_parent(checkpoint, validation_root, output_root, *, keep_raw_results=True, batch_size=32):
    if not torch.cuda.is_available():
        raise RuntimeError("Fixed parent diagnostics require CUDA; no silent CPU fallback")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    protocol, images = verify_validation(validation_root)
    checkpoint_hash = compute_sha256(str(checkpoint))
    protocol_hash = compute_sha256(str(Path(validation_root) / "protocol.json"))
    module_hash = compute_sha256(__file__)
    key = hashlib.sha256((checkpoint_hash+protocol_hash+module_hash+str(keep_raw_results)).encode()).hexdigest()[:24]
    out = Path(output_root) / key
    with artifact_lock(out):
        if (out / "COMPLETED").is_file():
            if (out / "COMPLETED").read_text() != compute_sha256(str(out / "summary.json")):
                raise ValueError("Diagnostic summary changed")
            return out / "summary.json"
        out.mkdir(parents=True, exist_ok=True)
        payload = load_checkpoint(str(checkpoint))
        cfg = payload["config"]
        if cfg["stage"] != "pretrain" or cfg["dataset"]["name"] != "VGGFace2":
            raise ValueError("Expected a VGGFace2 Parent checkpoint")
        from backbone.MobileFaceNets import MobileFaceNet
        model = MobileFaceNet(cfg["backbone"]["feat_dim"],7,7,dropout_rate=cfg["backbone"]["dropout_rate"])
        model.load_state_dict(payload["backbone_state_dict"],strict=True)
        model = model.cuda().eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0,len(images),batch_size):
                arrays = [cv2.imdecode(np.fromfile(row["output"],np.uint8),cv2.IMREAD_COLOR) for row in images[start:start+batch_size]]
                batch = (np.stack(arrays).astype(np.float32)-127.5)/128.0
                inputs = torch.from_numpy(batch.transpose(0,3,1,2).copy()).cuda()
                outputs.append(model(inputs).cpu().numpy())
        embeddings = np.concatenate(outputs)
        if not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings,axis=1) <= 0).any():
            raise ValueError("Invalid diagnostic embeddings")
        metrics, ranks = recognition_metrics(embeddings, images)
        identity_results = metrics.pop("identity_results")
        write_json(out / "identity_counts.json", identity_results, immutable=True)
        if keep_raw_results:
            np.savez_compressed(out / "raw_outputs.npz", embeddings=embeddings, ranks=ranks)
        summary = dict(schema=SCHEMA, status="completed", scope="Local fixed VGGFace2 test diagnostic; not official benchmark",
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, epoch=payload["epoch"],
            weight_decay=cfg["training"]["weight_decay"], protocol_sha256=protocol_hash,
            implementation_sha256=module_hash, device=torch.cuda.get_device_name(0), batch_size=batch_size,
            metrics=metrics, keep_raw_results=keep_raw_results, raw_retained=keep_raw_results,
            identity_counts_sha256=compute_sha256(str(out / "identity_counts.json")),
            limitations="Same cohort reused across models; not independent repeats. Parent selection must not use final SurvFace test.")
        write_json(out / "summary.json", summary, immutable=True)
        (out / "COMPLETED").write_text(compute_sha256(str(out / "summary.json")),encoding="utf-8")
        del model, payload
        torch.cuda.empty_cache()
        return out / "summary.json"


def compare_parents(reference_summary, control_summary, output, *, seed=8972):
    a, b = [json.loads(Path(path).read_text(encoding="utf-8")) for path in (reference_summary, control_summary)]
    if a["protocol_sha256"] != b["protocol_sha256"] or a["epoch"] != b["epoch"]:
        raise ValueError("Paired comparison requires same cohort and epoch")
    groups = []
    for path, summary in ((reference_summary,a),(control_summary,b)):
        counts = Path(path).with_name("identity_counts.json")
        if compute_sha256(str(counts)) != summary["identity_counts_sha256"]:
            raise ValueError("Identity aggregates changed")
        groups.append(json.loads(counts.read_text(encoding="utf-8")))
    if [(x['identity'],x['probes']) for x in groups[0]] != [(x['identity'],x['probes']) for x in groups[1]]:
        raise ValueError("Identity/probe cohorts differ")
    delta = np.asarray([y['rank1_correct']/y['probes']-x['rank1_correct']/x['probes'] for x,y in zip(*groups)])
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
    result = dict(reference=str(reference_summary), control=str(control_summary), epoch=a["epoch"],
        protocol_sha256=a["protocol_sha256"], rank1_difference=float(delta.mean()),
        paired_identity_bootstrap_95ci=np.quantile(bootstrap,[.025,.975]).tolist(),
        confidence_scope="Paired identity bootstrap on this diagnostic cohort; not independent model seeds or a performance guarantee",
        reference_metrics=a["metrics"], control_metrics=b["metrics"])
    write_json(output,result,immutable=True)
    return output
