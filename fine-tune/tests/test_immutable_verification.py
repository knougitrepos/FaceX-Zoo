"""Receipt reuse is an explicit immutability promise, not a byte-integrity claim."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import alignment, integrity_cache, dataset_verification as verify
from modules import notebook_workflow as flow
from modules.baseline_policy import code_fingerprints
from modules.checkpoint_compatibility import MANIFEST, source_transition, verification_only_transition
from modules.checkpoint_manager import compare_configs
from test_notebook_workflow import cfg, aligned_cfg


def immutable(cfg, force=False):
    result = copy.deepcopy(cfg)
    result["data_integrity"] = dict(mode=verify.MODE, force_reverify=force)
    return result


def forbidden(*args, **kwargs):
    raise AssertionError("Reuse must not walk mappings or open/hash/decode individual images")


def test_initial_full_then_reuse_without_image_or_mapping_scan(aligned_cfg, monkeypatch):
    cfg = immutable(aligned_cfg)
    first = verify.bind_aligned_dataset(cfg)["data_integrity_verification"]
    assert first["image_hashes_checked"] == first["images_decoded"] == 16
    assert not first["reused_full_verification"]
    record = json.loads(Path(first["receipt_path"]).read_text())
    assert record["status"] == "completed" and record["verified_images"] == 8
    monkeypatch.setattr(integrity_cache, "verify_image", forbidden)
    monkeypatch.setattr(alignment, "read_rows", forbidden)
    second = verify.bind_aligned_dataset(cfg)["data_integrity_verification"]
    assert second["reused_full_verification"]
    assert second["image_hashes_checked"] == second["images_decoded"] == 0
    assert second["receipt_sha256"] == first["receipt_sha256"]


def test_adopts_exact_existing_full_receipt(aligned_cfg, monkeypatch):
    original = alignment.bind_aligned_dataset(aligned_cfg)
    previous = Path(original["data_integrity_verification"]["receipt_path"])
    original_bytes = previous.read_bytes()
    monkeypatch.setattr(integrity_cache, "verify_image", forbidden)
    monkeypatch.setattr(alignment, "read_rows", forbidden)
    selected = verify.bind_aligned_dataset(immutable(aligned_cfg))
    assert selected["dataset"] == original["dataset"]
    result = selected["data_integrity_verification"]
    assert result["image_hashes_checked"] == 0
    record = json.loads(Path(result["receipt_path"]).read_text())
    assert record["evidence"] == "existing_full_hash_decode_receipt"
    assert record["source_receipt"] == str(previous)
    assert previous.read_bytes() == original_bytes


def test_force_rechecks_and_failure_cannot_reuse_old_success(aligned_cfg):
    cfg = immutable(aligned_cfg)
    verify.bind_aligned_dataset(cfg)
    full = verify.bind_aligned_dataset(immutable(cfg, True))["data_integrity_verification"]
    assert full["force_reverify"] and full["images_decoded"] == 16
    report = json.loads(Path(cfg["dataset"]["alignment_report"]).read_text())
    row = json.loads(Path(report["manifest"]).read_text().splitlines()[0])
    image = Path(row["output"])
    original = image.read_bytes()
    image.write_bytes(b"changed")
    # Deliberately demonstrate the documented limitation, even for changed size.
    assert verify.bind_aligned_dataset(cfg)["data_integrity_verification"]["image_hashes_checked"] == 0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify.bind_aligned_dataset(immutable(cfg, True))
    assert json.loads(Path(full["receipt_path"]).read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="FORCE_DATA_REVERIFY"):
        verify.bind_aligned_dataset(cfg)
    image.write_bytes(original)
    assert verify.bind_aligned_dataset(immutable(cfg, True))["data_integrity_verification"]["images_decoded"] == 16
    assert list(Path(full["receipt_path"]).parent.joinpath("history").glob("*.json"))


@pytest.mark.parametrize("key", ["manifest", "train_list", "source_list"])
def test_metadata_modification_is_rejected_even_with_success(aligned_cfg, key):
    cfg = immutable(aligned_cfg)
    verify.bind_aligned_dataset(cfg)
    report = json.loads(Path(cfg["dataset"]["alignment_report"]).read_text())
    with Path(report[key]).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch|source list differs"):
        verify.bind_aligned_dataset(cfg)


def test_partial_or_locked_data_cannot_use_receipt(aligned_cfg):
    cfg = immutable(aligned_cfg)
    verify.bind_aligned_dataset(cfg)
    root = Path(cfg["dataset"]["alignment_report"]).parent
    (root / "COMPLETED").unlink()
    with pytest.raises(ValueError, match="partial/failed"):
        verify.bind_aligned_dataset(cfg)


def test_bad_receipt_requires_explicit_reverification(aligned_cfg):
    cfg = immutable(aligned_cfg)
    v = verify.bind_aligned_dataset(cfg)["data_integrity_verification"]
    receipt = Path(v["receipt_path"])
    receipt.write_text('{"status": "completed"}')
    with pytest.raises(ValueError, match="FORCE_DATA_REVERIFY"):
        verify.bind_aligned_dataset(cfg)
    assert verify.bind_aligned_dataset(immutable(cfg, True))["data_integrity_verification"]["images_decoded"] == 16


def test_interrupted_first_check_cannot_publish_success(aligned_cfg, monkeypatch):
    cfg = immutable(aligned_cfg)
    def interrupt(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(integrity_cache, "verify_image", interrupt)
    with pytest.raises(KeyboardInterrupt):
        verify.bind_aligned_dataset(cfg)
    with pytest.raises(ValueError, match="FORCE_DATA_REVERIFY"):
        verify.bind_aligned_dataset(cfg)


def test_no_receipt_never_trusts_a_completed_preprocessing_report(aligned_cfg, monkeypatch):
    monkeypatch.setattr(integrity_cache.cv2, "imdecode", lambda *a, **k: None)
    with pytest.raises(ValueError, match="Invalid source"):
        verify.bind_aligned_dataset(immutable(aligned_cfg))


def test_mode_and_force_reach_common_workflow(cfg):
    selected = flow.resolve_settings(Path(__file__).resolve().parents[2], overrides=immutable(cfg))
    assert selected["data_integrity"] == dict(mode=verify.MODE, force_reverify=False)
    assert flow.bind_aligned_dataset is verify.bind_aligned_dataset
    cfg["data_integrity"] = dict(mode=verify.MODE, force_reverify="False")
    with pytest.raises(ValueError, match="bool"):
        flow.validate_settings(cfg)


def test_only_exact_audited_source_transition_is_allowed():
    record = json.loads(MANIFEST.read_text())
    old, new = record["previous_sources"], code_fingerprints()
    assert new == record["current_sources"]
    assert source_transition(old, new) == record["id"]
    before = dict(code_fingerprints=old, training={"batch_size": 64}, seed=8972)
    after = {**before, "code_fingerprints": new}
    assert verification_only_transition(before, after)
    assert not compare_configs(before, after, full_strict=True)
    after["training"] = {"batch_size": 32}
    assert not verification_only_transition(before, after)
    assert compare_configs(before, after, full_strict=True)
    for key in new:
        changed = {**new, key: "0" * 64}
        assert source_transition(old, changed) is None
    assert source_transition(new, old) is None


def test_notebooks_expose_force_and_validated_resume():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for path in root.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][2]["source"])
        assert 'DATA_INTEGRITY_MODE = "immutable_verified_reuse"' in source
        assert "FORCE_DATA_REVERIFY = False" in source
        assert "force_reverify=FORCE_DATA_REVERIFY" in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")


def test_preprocessing_resume_of_completed_dataset_uses_same_policy(aligned_cfg, tmp_path, monkeypatch):
    import runpy
    cfg = immutable(aligned_cfg)
    verify.bind_aligned_dataset(cfg)
    settings = tmp_path / "settings.json"
    flow.save_settings(settings, cfg)
    monkeypatch.setattr(alignment, "create_preprocessor", forbidden)
    monkeypatch.setattr(integrity_cache, "verify_image", forbidden)
    monkeypatch.setattr(alignment, "read_rows", forbidden)
    monkeypatch.setattr(sys, "argv", ["prepare_alignment", "--settings", str(settings),
                                      "--dataset", "vggface2", "--resume"])
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_alignment.py"
    assert runpy.run_path(str(script))["main"]() == 0
