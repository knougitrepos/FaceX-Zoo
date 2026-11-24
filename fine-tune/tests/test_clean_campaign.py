import csv
import json
import os
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import clean_campaign as campaign
from modules import clean_training as training
from modules.clean_protocols import DATASETS, HEADS, LEGACY_PARENT, LEGACY_SHA, SCOPES, downstream
from modules.checkpoint_manager import load_checkpoint


def test_expansion_needs_human_approval_and_frozen_probe(tmp_path):
    with pytest.raises(ValueError, match="APPROVE_EXPANSION"):
        campaign.run_full(
            tmp_path,
            tmp_path,
            "paper_table1",
            ["tinyface"],
            ["adaface"],
            approved=False,
            reviewed_datasets=["tinyface"],
        )
    with pytest.raises(FileNotFoundError):
        campaign.run_full(
            tmp_path,
            tmp_path,
            "paper_table1",
            ["tinyface"],
            ["adaface"],
            approved=True,
            reviewed_datasets=["tinyface"],
        )
    assert campaign.model_dir(
        tmp_path, "paper_table1", "clean_parent", "survface", "cosface"
    ) != campaign.model_dir(tmp_path, "calibration_study", "clean_parent", "survface", "cosface")


@pytest.mark.parametrize(
    "dataset, source_scope, target_scope",
    [("tinyface", "paper_table1", "calibration_study"), ("scface", "calibration_study", "paper_table1")],
)
def test_shared_scope_reuses_verified_checkpoint_and_exports_for_target(
    tmp_path, monkeypatch, dataset, source_scope, target_scope
):
    from modules.backbone_factory import build_backbone
    from modules.checkpoint_manager import compute_config_hash, compute_sha256

    torch.set_num_threads(2)
    run = tmp_path / "facex/clean_ucface/reuse_fixture"
    parent_sha = "9" * 64
    model = build_backbone(dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=0.6))

    def recipe_for(run, root, ds, head, scope):
        return dict(
            dataset=ds,
            head=head,
            epochs=40,
            dropout=0.6,
            classes=50 if ds == "scface" else 2570,
            margin=0.3,
            scale=64.0,
            protocol_scope=scope,
            origin_only=scope == "paper_table1",
            preprocessing=dict(input_size=[112, 112], pixel_mean=127.5, pixel_std=128.0, color="BGR"),
            protocol_record={"name": ds},
            data_recipe="alignment",
            train_hash="b" * 64,
        )

    monkeypatch.setattr(campaign, "matrix_recipe", recipe_for)
    monkeypatch.setattr(campaign, "load_parent", lambda *a: (None, parent_sha))
    monkeypatch.setattr(
        campaign, "run_stage", lambda *a, **kw: pytest.fail("A completed model was retrained")
    )
    source = campaign.model_dir(run, source_scope, "clean_parent", dataset, "cosface")
    source.mkdir(parents=True)
    original = recipe_for(run, tmp_path, dataset, "cosface", source_scope)
    train_list = str(tmp_path / "train_list.txt")
    # Exercise historical compatibility with the exact known previous implementation.
    plan = dict(
        stage="finetune",
        run_dir=str(run),
        seed=8972,
        recipe=original,
        train_list=train_list,
        parent_sha256=parent_sha,
        initial_weights="clean_parent",
        keep_raw_results=True,
        implementation={**training.files_for_protocol(), **training._PRE_SCOPE_REUSE},
    )
    training.write_json(source / "plan.json", plan)
    plan_sha = compute_sha256(source / "plan.json")
    checkpoint = source / "epoch_40_abcdefabcdef.pt"
    config = training.checkpoint_config("finetune", original, train_list, matrix=True)
    torch.save(
        dict(
            epoch=40,
            config=config,
            config_hash=compute_config_hash(config),
            seed=8972,
            parent_hash=parent_sha,
            stage="finetune",
            completed=True,
            extra_meta=dict(plan_sha256=plan_sha),
            backbone_state_dict=model.state_dict(),
        ),
        checkpoint,
    )
    source_sha = compute_sha256(checkpoint)
    training.publish_summary(
        source,
        dict(
            plan_sha256=plan_sha,
            status="completed",
            guide="HOW_TO_READ.md",
            parent_sha256=parent_sha,
            epoch=40,
            epochs=[dict(epoch=e, loss=0.7, head_direction_norm=0.4) for e in range(1, 41)],
            checkpoint=checkpoint.name,
            sha256=source_sha,
            protocol_scope=source_scope,
            origin_only=source_scope == "paper_table1",
            keep_raw_results=True,
        ),
    )
    target = campaign.model_dir(run, target_scope, "clean_parent", dataset, "cosface")
    args = (tmp_path, run, target_scope, "clean_parent", [dataset], ["cosface"])

    with pytest.raises(ValueError, match="Parent, seed or settings"):
        campaign.train_models(*args, seed=8973)
    assert not target.exists()
    original_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        campaign.train_models(*args)
    assert not target.exists()
    checkpoint.write_bytes(original_bytes)
    result = campaign.train_models(*args)
    assert result[0]["status"] == "verified_cross_scope_reuse"
    assert campaign.train_models(*args)[0]["status"] == "verified_reuse"

    cloned, receipt, _, clone_plan = campaign.verified_model(
        tmp_path, run, target_scope, "clean_parent", dataset, "cosface", 8972
    )
    assert cloned.parent == target and compute_sha256(cloned) == source_sha == compute_sha256(checkpoint)
    assert (source / "COMPLETED").read_text() == receipt["reused_from"]["summary_sha256"]
    assert receipt["reused_from"] == clone_plan["reused_from"]
    assert receipt["protocol_scope"] == target_scope and receipt["origin_only"] is (
        target_scope == "paper_table1"
    )
    assert receipt["epochs"][-1] == dict(epoch=40, loss=0.7, head_direction_norm=0.4)
    assert receipt["keep_raw_results"] is False and receipt["reused_from"]["keep_raw_results"] is True

    affine = tmp_path / "frozen_affine.csv"
    affine.write_text("geometry", encoding="utf8")
    geometry = dict(
        file="affine.csv", sha256=compute_sha256(affine), rows=1, schema="raw_to_arc112_affine_v1"
    )
    monkeypatch.setattr(campaign, "geometry_source", lambda *a: (affine, geometry))
    export = campaign.export_model(tmp_path, run, target_scope, "clean_parent", dataset, "cosface")
    metadata = torch.load(export["path"], weights_only=True)["metadata"]
    assert metadata["source_config"] == config and metadata["source_config_hash"] == compute_config_hash(
        config
    )
    assert metadata["protocol_scope"] == target_scope and metadata["origin_only"] is receipt["origin_only"]
    assert metadata["source_sha256"] == source_sha and metadata["reused_from"] == receipt["reused_from"]


@pytest.mark.parametrize("dataset, source", [("survface", "clean_parent"), ("tinyface", "legacy_parent_wd0")])
def test_other_scope_never_supplies_survface_or_legacy_parent_model(tmp_path, monkeypatch, dataset, source):
    run = tmp_path / "facex/clean_ucface/unshared_fixture"
    peer = campaign.model_dir(run, "paper_table1", source, dataset, "cosface")
    peer.mkdir(parents=True)
    (peer / "COMPLETED").touch()
    monkeypatch.setattr(campaign, "load_parent", lambda *a: (None, "9" * 64))
    monkeypatch.setattr(
        campaign, "verified_model", lambda *a, **kw: pytest.fail("An unrelated model was selected")
    )
    monkeypatch.setattr(campaign, "matrix_recipe", lambda *a: {})
    started = []
    monkeypatch.setattr(campaign, "run_stage", lambda *a, **kw: started.append(kw["matrix"][0]))
    result = campaign.train_models(tmp_path, run, "calibration_study", source, [dataset], ["cosface"])
    assert started == [campaign.model_dir(run, "calibration_study", source, dataset, "cosface")]
    assert result[0]["status"] == "completed"


@pytest.mark.skipif(os.getenv("FACE_CLEAN_GPU_SMOKE") != "1", reason="opt-in real local datasets and GPU")
def test_real_inputs_four_heads_three_datasets_resume_and_parent_sources(tmp_path, monkeypatch):
    data = Path(os.getenv("FACE_DATASET_ROOT", "F:/face_datasets"))
    model_root = Path(os.getenv("FACE_MODEL_ROOT", "F:/models"))
    assert torch.cuda.is_available()
    run = tmp_path / "facex/clean_ucface/verification"
    legacy = model_root / LEGACY_PARENT
    monkeypatch.setattr(training, "LEGACY_PARENT", str(legacy.resolve()))
    old, old_sha = training.load_parent(run, "legacy_parent_wd0", 8972)
    assert old_sha == LEGACY_SHA and old["epoch"] == 40
    del old

    paths = {(ds, scope): downstream(data, model_root, ds, scope) for ds in DATASETS for scope in SCOPES}
    assert (paths["tinyface", "paper_table1"]["classes"], paths["tinyface", "paper_table1"]["images"]) == (
        2570,
        7804,
    )
    assert (paths["scface", "paper_table1"]["classes"], paths["scface", "paper_table1"]["images"]) == (
        50,
        800,
    )
    assert (
        paths["survface", "calibration_study"]["classes"],
        paths["survface", "calibration_study"]["images"],
    ) == (4255, 176205)
    assert (paths["survface", "paper_table1"]["classes"], paths["survface", "paper_table1"]["images"]) == (
        5319,
        220888,
    )
    for rel, expect_overlap in (
        ("shared_v1/9aeb13b96df3ae0e1912", False),
        ("paper_baseline_v1/5319_full_fr_v1", True),
    ):
        with (data / "QMUL-SurvFace/protocols" / rel / "roles.csv").open(
            encoding="utf-8-sig", newline=""
        ) as source:
            train, calibration = set(), set()
            for row in csv.DictReader(source):
                if row["fr_train"] == "1":
                    train.add(row["identity_id"])
                if row["role"] in {"calibration_fit", "calibration_safety"}:
                    calibration.add(row["identity_id"])
            assert bool(train & calibration) is expect_overlap

    original_recipe = training.recipe
    monkeypatch.setattr(training, "downstream", lambda root, model, ds, scope: paths[ds, scope])
    parent_src = training.input_list(data, original_recipe("parent"))

    def sample(path, target, count):
        wanted = set(random.Random(8972).sample(range(count), 64))
        with path.open(encoding="utf-8") as stream:
            lines = [line for i, line in enumerate(stream) if i in wanted]
        target.mkdir(parents=True, exist_ok=True)
        (target / "train_list.txt").write_text("".join(lines), encoding="utf-8")
        assert len(lines) == 64

    sample(parent_src, tmp_path / "parent_input", 3141890)
    for (ds, scope), record in paths.items():
        sample(Path(record["prepared"]) / "train_list.txt", tmp_path / f"{ds}_{scope}", record["images"])
    # Use the production checkpointing, input loader and models with a one-batch epoch.
    # Temporary runs stay below formal epoch budgets and outside the real model root.
    monkeypatch.setattr(
        training,
        "recipe",
        lambda stage: dict(original_recipe(stage), epochs=1 if stage == "parent" else 4, batch=64, images=64),
    )
    monkeypatch.setattr(training, "MILESTONES", ())
    monkeypatch.setattr(training, "prepare_panels", lambda root: ({}, {}))
    monkeypatch.setattr(training, "input_list", lambda root, r: tmp_path / "parent_input/train_list.txt")
    parent = training.run_stage(data, run, "parent")
    clean, clean_sha = training.load_parent(run, "clean_parent", 8972)
    assert clean_sha == parent["sha256"] and clean_sha != old_sha
    del clean

    model_rows = [(ds, head, "paper_table1", "clean_parent") for ds in DATASETS for head in HEADS]
    model_rows.extend(
        [
            ("tinyface", "cosface", "paper_table1", "legacy_parent_wd0"),
            ("survface", "cosface", "calibration_study", "clean_parent"),
        ]
    )
    train_epoch = training.train_epoch

    def interrupt(model, opt, stage, r, loader, epoch):
        if epoch == 4:
            raise InterruptedError("controlled stop before full backbone phase")
        return train_epoch(model, opt, stage, r, loader, epoch)

    for ds, head, scope, source in model_rows:
        r = training.matrix_recipe(run, data, ds, head, scope)
        assert r["origin_only"] is (scope == "paper_table1")
        r = dict(r, prepared=str(tmp_path / f"{ds}_{scope}"), images=64)
        folder = campaign.model_dir(run, scope, source, ds, head)
        options = dict(matrix=(folder, r), parent_source=source)
        monkeypatch.setattr(training, "train_epoch", interrupt)
        with pytest.raises(InterruptedError):
            training.run_stage(data, run, "finetune", **options)
        assert not (folder / "COMPLETED").exists()
        saved = json.loads((folder / "progress.json").read_bytes())
        assert saved["epoch"] == 3
        monkeypatch.setattr(training, "train_epoch", train_epoch)
        report = training.run_stage(data, run, "finetune", resume=True, **options)
        assert report["epochs"][:3] == saved["epochs"]
        assert report["parent_sha256"] == (old_sha if source == "legacy_parent_wd0" else clean_sha)
        assert report["origin_only"] is (scope == "paper_table1")
        checkpoint, _ = training.final_artifact(folder, expected_recipe=r)
        state = load_checkpoint(str(checkpoint))
        assert state["optimizer_state_dict"]["state"]
        assert state["rng_state"]["cuda_rng"]
        assert all(
            row["output_bn_weight_l2"] > 0 and 0 <= row["head_direction_norm"] <= 1
            for row in report["epochs"]
        )
        assert (state["backbone_state_dict"]["bn.bias"] == 0).all()
    args = (run, "paper_table1", "clean_parent", "tinyface", "cosface")
    folder = campaign.model_dir(*args)
    recorded = json.loads((folder / "plan.json").read_bytes())["recipe"]
    _, trained = training.final_artifact(folder, expected_recipe=recorded)
    monkeypatch.setattr(campaign, "matrix_recipe", lambda *a: recorded)
    monkeypatch.setattr(
        campaign, "run_stage", lambda *a, **kw: pytest.fail("completed CosFace was retrained")
    )
    probe_folder = campaign.evaluation_dir(folder, "tinyface", probe=True)
    campaign._record(
        probe_folder,
        dict(
            source_sha256=trained["sha256"],
            parent_sha256=trained["parent_sha256"],
            protocol_scope="paper_table1",
            dataset="tinyface",
            head="cosface",
            implementation=training.files_for_protocol(),
        ),
        lambda: {"comparison": []},
    )
    reused = campaign.run_full(
        data, run, "paper_table1", ["tinyface"], ["cosface"], approved=True, reviewed_datasets=["tinyface"]
    )
    assert reused[0]["status"] == "verified_reuse"


def test_backbone_only_publish_and_reuse_for_each_scope(tmp_path, monkeypatch):
    from modules.backbone_factory import build_backbone
    from modules.checkpoint_manager import compute_config_hash, compute_sha256

    torch.set_num_threads(2)
    model = build_backbone(dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=0.6))
    run = tmp_path / "facex/clean_ucface/export_fixture"
    for scope, classes in (("paper_table1", 5319), ("calibration_study", 4255)):
        folder = campaign.model_dir(run, scope, "clean_parent", "survface", "cosface")
        folder.mkdir(parents=True)
        checkpoint = folder / "fixture.pt"
        prep = dict(input_size=[112, 112], pixel_mean=127.5, pixel_std=128.0, color="BGR")
        config = dict(
            backbone=dict(type="MobileFaceNet", feat_dim=1024, dropout_rate=0.6),
            head=dict(type="cosface"),
            dataset=dict(name="survface", num_classes=classes),
            preprocessing=prep,
        )
        torch.save(
            dict(
                epoch=40,
                config=config,
                config_hash=compute_config_hash(config),
                backbone_state_dict=model.state_dict(),
            ),
            checkpoint,
        )
        receipt = dict(sha256=compute_sha256(checkpoint), parent_sha256="9" * 64)
        r = dict(
            protocol_record={"num_classes": classes},
            origin_only=scope == "paper_table1",
            data_recipe="resize_only",
            preprocessing=prep,
        )
        monkeypatch.setattr(campaign, "verified_model", lambda *a: (checkpoint, receipt, r, {}))
        args = (tmp_path, run, scope, "clean_parent", "survface", "cosface")
        result = campaign.export_model(*args)
        assert campaign.export_model(*args) == result
        payload = torch.load(result["path"], weights_only=True)
        assert set(payload) == {"state_dict", "metadata"}
        assert payload["metadata"]["origin_only"] is (scope == "paper_table1")
        assert payload["metadata"]["parent_sha256"] == receipt["parent_sha256"]
