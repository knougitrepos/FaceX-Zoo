"""Immutable training/calibration handoff consumed by ronbun.

Only derived files are written. The default SurvFace split is a fixed research
candidate, not a claim of optimality or a reproduction of published scores.
"""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "facex-ronbun-roles-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key(value, seed, namespace):
    return hashlib.sha256(f"{namespace}:{seed}:{value}".encode()).hexdigest()


def _publish(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Immutable protocol differs: {path}")
    else:
        with path.open("x", encoding="utf-8", newline="") as f:
            f.write(content)


def _csv(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def publish_handoff(profiles, previous_path=None):
    """Publish a new immutable index when more datasets become available.

    handoff.json remains the first snapshot for compatibility. Consumers pin
    the returned content-addressed path; no mutable 'latest' pointer is used.
    """
    roots = {Path(p["research_output"]).resolve().parent for p in profiles}
    if len(roots) != 1:
        raise ValueError("A handoff must use one explicit protocol namespace")
    root = roots.pop()
    first = root / "handoff.json"
    previous = Path(previous_path) if previous_path else first
    records = {}
    if previous_path or previous.exists():
        prior = json.loads(previous.read_text(encoding="utf-8"))
        if prior.get("schema") != "facex-ronbun-handoff-v1":
            raise ValueError("Unsupported prior handoff")
        records.update(prior["protocols"])
        for item in records.values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Prior handoff protocol changed")
    for profile in profiles:
        item = dict(path=str(Path(profile["protocol_record"]).resolve()), sha256=sha(profile["protocol_record"]))
        name = profile["name"]
        if name in records and records[name] != item:
            raise FileExistsError("Changed dataset protocol requires a new namespace and no previous handoff")
        records[name] = item
    content = json.dumps(dict(schema="facex-ronbun-handoff-v1", protocols=records), sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    path = root / "handoffs" / f"{digest}.json"
    _publish(path, content)
    if not first.exists():
        _publish(first, content)
    return path


def _scface(profile, seed):
    root = Path(profile["raw_root"])
    frontal = sorted((root / "mugshot_frontal_cropped_all").glob("*.jpg"))
    ids = [p.stem.split("_")[0] for p in frontal]
    if len(ids) != 130 or len(set(ids)) != 130:
        raise ValueError("SCFace requires 130 unique frontal identities")
    selected = profile.get("train_identity_ids")
    if selected is None:
        selected = sorted(ids, key=lambda i: key(i, seed, "scface:research-train"))[:50]
    if len(set(selected)) != 50 or not set(selected) <= set(ids):
        raise ValueError("SCFace requires 50 explicit unique training identities")
    train_ids = set(selected)
    train, test = [], []
    for identity, p in zip(ids, frontal):
        dest = train if identity in train_ids else test
        dest.append(dict(image_path=p.relative_to(root).as_posix(), source_identity_id=identity,
                         protocol_role="training" if identity in train_ids else "gallery",
                         split="training" if identity in train_ids else "test"))
        for camera in range(1, 6):
            for distance in range(1, 4):
                p = root / "surveillance_cameras_all" / f"{identity}_cam{camera}_{distance}.jpg"
                if not p.is_file():
                    raise FileNotFoundError(p)
                dest.append(dict(image_path=p.relative_to(root).as_posix(), source_identity_id=identity,
                                 protocol_role="training" if identity in train_ids else "registered_probe",
                                 split="training" if identity in train_ids else "test"))
    return train, test


def prepare_research_protocol(profile):
    if profile.get("portable_protocol"):
        return verify_research_protocol(profile)
    if not profile.get("enabled", True):
        raise ValueError("Selected research training dataset is disabled")
    name, root = profile["name"], Path(profile["raw_root"]).resolve()
    if name not in ("survface", "tinyface", "scface"):
        raise ValueError("LFW/RFW are evaluation-only, never FR training")
    seed = int(profile.get("split_seed", 8972))
    sources = {}
    if name == "scface":
        train, test = _scface(profile, seed)
    else:
        train, test = _read(profile["source_training_manifest"]), _read(profile["source_evaluation_manifest"])
        sources = {k: {"path": str(Path(profile[k]).resolve()), "sha256": sha(profile[k])}
                   for k in ("source_training_manifest", "source_evaluation_manifest")}
    groups = defaultdict(list)
    for r in train:
        if r.get("protocol_role") != "training":
            raise ValueError("FR source contains non-training rows")
        groups[str(r["source_identity_id"])].append(r)
    expected = {"survface": (5319, 220888), "tinyface": (2570, 7804), "scface": (50, 800)}[name]
    if not profile.get("allow_test_fixture") and (len(groups), len(train)) != expected:
        raise ValueError(f"Source counts differ for {name}: {(len(groups), len(train))}")
    fr_ids = set(groups)
    if name == "survface":
        reference = Path(profile["split_reference"])
        assignments = {}
        image_counts = Counter()
        for r in _read(reference):
            identity = r["identity_id"].split(":")[-1]
            if identity in assignments and assignments[identity] != r["split"]:
                raise ValueError("Reference identity spans roles")
            assignments[identity] = r["split"]
            image_counts[identity] += 1
        if (set(assignments) != set(groups) or set(assignments.values()) != {"development", "calibration"}
                or any(image_counts[i] != len(groups[i]) for i in groups)):
            raise ValueError("Split reference does not match native training identities/images")
        fr_ids = {i for i in groups if assignments[i] == "development"}
        sources["split_reference"] = {"path": str(reference.resolve()), "sha256": sha(reference)}
    gallery_ids = set()
    if name == "survface":
        count = int(profile.get("gallery_identity_count", 3000))
        if len(fr_ids) <= count:
            raise ValueError("Reserve PQ-development identities outside gallery")
        gallery_ids = set(sorted(fr_ids, key=lambda i: key(f"survface:train:{i}", seed,
                          "survface:training-matched-calibration:gallery-identity"))[:count])
    rows = []
    for identity in sorted(groups):
        canonical = f"{name}:train:{identity}"
        group = groups[identity]
        if identity in gallery_ids:
            group = sorted(group, key=lambda r: key(f"survface:train:{identity}:{Path(r['image_path']).stem}", seed,
                           f"survface:training-matched-calibration:{canonical}:half-gallery"))
        for j, r in enumerate(group):
            role = "pq_development"
            if identity in gallery_ids:
                role = "calibration_gallery" if j < len(group) // 2 else "calibration_registered_probe"
            elif identity not in fr_ids:
                u = int.from_bytes(hashlib.sha256(f"{seed}:{canonical}".encode()).digest()[:8], "big") / 2**64
                role = "calibration_safety" if u < .3 else "calibration_fit"
            rows.append(dict(image_id=f"{canonical}:{Path(r['image_path']).stem}", identity_id=canonical,
                             source_identity_id=identity, image_path=r["image_path"], role=role,
                             fr_train=int(identity in fr_ids), group="", distance=""))
    for index, r in enumerate(test):
        native = r.get("source_identity_id", "")
        # Opaque distractors are never assigned one fake shared identity.
        if native in fr_ids or (native and native in groups):
            raise ValueError("Native training/test identity overlap")
        role = {"gallery": "test_gallery", "registered_probe": "test_registered_probe",
                "unknown_unknown_probe": "test_unknown_probe", "unknown_known_probe": "test_unknown_probe",
                "gallery_distractor": "test_gallery_distractor"}.get(r["protocol_role"])
        if role is None:
            raise ValueError(f"Unknown evaluation role: {r['protocol_role']}")
        identity = f"{name}:test:{native}" if native else ""
        stem = Path(r["image_path"]).stem
        rows.append(dict(image_id=f"{name}:test:{index}:{stem}", identity_id=identity,
                         source_identity_id=native, image_path=r["image_path"], role=role,
                         fr_train=0, group="", distance=(stem.rsplit("_", 1)[-1] if name == "scface" and "_cam" in stem else "")))
    if len({r["image_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image path across roles")
    # Check every experimental input; raw duplicate folders are intentionally ignored.
    for r in rows:
        if not (root / r["image_path"]).is_file():
            raise FileNotFoundError(root / r["image_path"])
    columns = list(rows[0])
    content = _csv(rows, columns)
    digest = hashlib.sha256(content.encode()).hexdigest()
    out = Path(profile["research_output"]) / digest[:20]
    role_path = out / "roles.csv"
    _publish(role_path, content)
    train_rows = [r for r in rows if r["fr_train"]]
    labels = {i: n for n, i in enumerate(sorted({r["identity_id"] for r in train_rows}))}
    train_list = out / "train_list.txt"
    _publish(train_list, "".join(f"{r['image_path']} {labels[r['identity_id']]}\n" for r in train_rows))
    by_role = {role: dict(images=sum(r["role"] == role for r in rows),
                         identities=len({r["identity_id"] for r in rows if r["role"] == role and r["identity_id"]}))
               for role in sorted({r["role"] for r in rows})}
    manifest = dict(schema=SCHEMA, name=name, policy="fixed_research_split_not_optimal", split_seed=seed,
                    raw_root=str(root), roles_path=str(role_path.resolve()), roles_sha256=digest,
                    train_list=str(train_list.resolve()), train_sha256=sha(train_list), num_classes=len(labels),
                    images=len(train_rows), identity_disjoint=True, role_counts=by_role, sources=sources,
                    calibration_gallery_fr_seen=name == "survface", parent_identity_overlap="not_audited",
                    source="UCFace quantity reference; researcher-defined identity roles",
                    all_images_decode="not_checked", training_identity_ids=sorted(fr_ids),
                    training_identity_sha256=hashlib.sha256("\n".join(sorted(fr_ids)).encode()).hexdigest())
    path = out / "protocol.json"
    _publish(path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    profile.update(num_classes=len(labels), expected_images=len(train_rows), train_list=str(train_list),
                   protocol_record=str(path), training_manifest=str(role_path), evaluation_manifest=str(role_path),
                   role_manifest_sha256=digest, training_identity_sha256=manifest["training_identity_sha256"],
                   shared_protocol_sha256=sha(path))
    return manifest


def verify_research_protocol(profile):
    if not profile.get("enabled", True):
        raise ValueError("Selected research training dataset is disabled")
    from .shared_storage import resolve_protocol
    record = resolve_protocol(profile["protocol_record"])
    if record["schema"] not in (SCHEMA, "facex-ronbun-roles-v2") or record["name"] != profile["name"]:
        raise ValueError("Research protocol identity mismatch")
    for key, checksum in (("roles_path", "roles_sha256"), ("train_list", "train_sha256")):
        if sha(record[key]) != record[checksum]:
            raise ValueError(f"Research protocol changed: {key}")
    for source in record["sources"].values():
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Research protocol source changed")
    if record["num_classes"] != profile["num_classes"] or Path(record["raw_root"]) != Path(profile["raw_root"]).resolve():
        raise ValueError("Research training class count/root changed")
    if Path(profile["train_list"]).resolve() != Path(record["train_list"]).resolve():
        raise ValueError("Research training list changed")
    return record
