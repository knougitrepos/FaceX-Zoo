"""Portable storage and provenance without importing the consumer repository."""
from pathlib import Path
import io
import json
import sys
import tarfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.shared_storage import artifact_lock, external_settings, sha, resolve_protocol
from modules.archive_preparation import extract_train_archive
from modules.alignment import prepare_alignment, verify_alignment, ResizePreprocessor

ROOT = Path(__file__).resolve().parents[2]


def make_archive(path, content=b"abc"):
    with tarfile.open(path,"w:gz") as t:
        m=tarfile.TarInfo("train/a/a.jpg");m.size=len(content);t.addfile(m,io.BytesIO(content))


def test_extraction_receipt_is_immutable_and_blocks_changed_archive(tmp_path):
    archive=tmp_path / "vgg.tar.gz";make_archive(archive)
    out=tmp_path / "data/VGGFace2/extracted/v1"
    extract_train_archive(archive,out)
    before=sha(out / "derivation.json")
    extract_train_archive(archive,out)
    assert sha(out / "derivation.json")==before
    assert (out / "README.md").exists() and (out / "COMPLETED").exists()
    make_archive(archive,b"changed")
    with pytest.raises(FileExistsError,match="different archive"):extract_train_archive(archive,out)
    assert (out / "train/a/a.jpg").read_bytes()==b"abc"


def test_extraction_exclusive_lock(tmp_path):
    archive=tmp_path / "a.tar.gz";make_archive(archive)
    out=tmp_path / "extracted"
    with artifact_lock(out):
        with pytest.raises(FileExistsError):extract_train_archive(archive,out)
    assert not out.exists()


def test_prepared_pixels_provenance_and_partial_gate(tmp_path):
    image=np.arange(11*13*3,dtype=np.uint8).reshape(11,13,3)
    cv2.imwrite(str(tmp_path / "a.png"),image)
    index=tmp_path / "index.txt";index.write_text("a.png 0\na.png 1\n")
    provider=ResizePreprocessor(dict(interpolation="bicubic"))
    out=tmp_path / "processed/v1"
    report=prepare_alignment(index,tmp_path,out,provider,max_images=1)
    assert not report["completed"] and not (out / "COMPLETED").exists()
    receipt=json.loads((out / "derivation.json").read_text())
    assert receipt["status"]=="partial_or_failed" and receipt["counts"]["prepared"]==1
    with pytest.raises(ValueError,match="partial/failed"):verify_alignment(out / "report.json",index,provider.policy)
    actual=cv2.imread(str(out / "images/000000/00000000.png"))
    np.testing.assert_array_equal(actual,cv2.resize(image,(112,112),interpolation=cv2.INTER_CUBIC))
    assert (out / "README.md").exists()


def test_external_settings_have_no_consumer_or_legacy_runtime_paths(tmp_path):
    data=tmp_path / "datasets";models=tmp_path / "models"
    idx=data / "protocols/shared_v1/handoff.json";idx.parent.mkdir(parents=True)
    entries={}
    for name in ("tinyface","survface","scface"):
        folder={"tinyface":"tinyface","survface":"QMUL-SurvFace","scface":"SCface_database"}[name]
        p=data / folder / "protocols/shared_v1/fixture/protocol.json";p.parent.mkdir(parents=True)
        record=dict(schema="facex-ronbun-roles-v2",name=name,raw_root="../../..",roles_path="roles.csv",train_list="train_list.txt",
                    sources={},split_seed=8972,num_classes=1,images=1,roles_sha256="fixture",training_identity_sha256="fixture")
        p.write_text(json.dumps(record))
        entries[name]=dict(path=f"../../{folder}/protocols/shared_v1/fixture/protocol.json",sha256=sha(p))
    idx.write_text(json.dumps(dict(schema="facex-ronbun-handoff-v2",protocols=entries)))
    cfg=external_settings(ROOT,data,models)
    assert cfg["backbone"]["feat_dim"]==1024
    assert Path(cfg["dataset"]["metadata_root"])==data / "VGGFace2"
    assert Path(cfg["dataset"]["raw_root"])==data / "VGGFace2/extracted/official_train_v1"
    assert all(Path(cfg["paths"][k]).is_relative_to(models) for k in ("checkpoint_dir","export_dir","log_dir","report_dir"))
    assert cfg["campaign"]["finetune_datasets"]==["survface","tinyface","scface"]
    assert all(Path(p["raw_root"]).is_relative_to(data) for p in cfg["finetune_profiles"].values())
    from modules.notebook_workflow import resolve_settings
    effective = resolve_settings(ROOT, overrides=cfg)
    for profile in effective["finetune_profiles"].values():
        assert "local_audit" not in profile and "legacy_fiqa_manifest" not in profile
    resnet=external_settings(ROOT,data,models,backbone=dict(type="Resnet",num_layers=50,mode="ir"))
    assert resnet["paths"]["checkpoint_dir"]!=cfg["paths"]["checkpoint_dir"]


def test_parent_indices_use_metadata_root_not_extraction_root(tmp_path):
    from modules.notebook_workflow import prepare_indices
    metadata=tmp_path / "VGGFace2";images=metadata / "extracted/v1"
    (metadata / "meta").mkdir(parents=True)
    (metadata / "meta/identity_meta.csv").write_text("id,name,count,flag\na,A,1,1\nb,B,1,1\n")
    (metadata / "meta/train_list.txt").write_text("a/a_1.jpg\nb/b_1.jpg\n")
    index=metadata / "protocols/v1/train_list.txt"
    cfg=dict(dataset=dict(raw_root=str(images),metadata_root=str(metadata),num_classes=2,train_list=str(index),val_list=None))
    result=prepare_indices(cfg)
    assert result["train_count"]==2
    assert index.read_text().splitlines()==["a/a_1.jpg 0","b/b_1.jpg 1"]
    assert not images.exists()  # index preparation does not extract or mutate source images
    record=json.loads((index.parent / "derivation.json").read_text())
    assert record["operation"]=="official_training_index" and record["counts"]["train_count"]==2
    assert (index.parent / "README.md").exists()
