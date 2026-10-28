from pathlib import Path
import hashlib
import json

import pytest


@pytest.fixture
def external_tree(tmp_path):
    """Portable index for plan-only tests; no dependence on a developer's F drive."""
    data, models = tmp_path / "datasets", tmp_path / "models"
    index = data / "protocols/shared_v1/handoff.json"
    index.parent.mkdir(parents=True)
    entries = {}
    for name, folder in (("survface","QMUL-SurvFace"),("tinyface","tinyface"),("scface","SCface_database")):
        path = data / folder / "protocols/shared_v1/fixture/protocol.json"
        path.parent.mkdir(parents=True)
        record = dict(schema="facex-ronbun-roles-v2", name=name, raw_root="../../..", roles_path="roles.csv",
                      train_list="train_list.txt", sources={}, split_seed=8972, num_classes=1, images=1,
                      roles_sha256="fixture", training_identity_sha256="fixture")
        path.write_text(json.dumps(record), encoding="utf-8")
        entries[name] = dict(path=f"../../{folder}/protocols/shared_v1/fixture/protocol.json",
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    index.write_text(json.dumps(dict(schema="facex-ronbun-handoff-v2",protocols=entries)),encoding="utf-8")
    return data, models
