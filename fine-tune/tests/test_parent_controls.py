import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_notebook_workflow import cfg, aligned_cfg
from modules import notebook_workflow as w
from modules import parent_controls as controls
from modules.trainer import Trainer, EpochLimitReached
from modules.checkpoint_manager import CheckpointManager, load_checkpoint


def test_epoch_pause_and_resume_matches_uninterrupted_training(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg['training'].update(epoches=3, save_freq=3, torch_threads=1)
    cfg['dataset']['val_list'] = None
    cfg['project']['keep_raw_results'] = True
    torch.set_num_threads(1)

    def trainer(path, cap):
        current = copy.deepcopy(cfg)
        current['runtime'] = {'stop_after_epoch': cap}
        w.set_seed(8972)
        model = w.build_model(current['backbone'], dict(current['heads']['cosface'], num_class=2))
        train, val = w.make_loaders(current)
        return Trainer(model, train, val, current, CheckpointManager(str(path)), torch.device('cpu'),
                       head_type='cosface', stage='pretrain')

    reference = trainer(tmp_path / 'reference', None)
    reference.train(model_prefix='control')
    expected = load_checkpoint(reference.last_saved_ckpt)
    paused = trainer(tmp_path / 'paused', 1)
    with pytest.raises(EpochLimitReached):
        paused.train(model_prefix='control')
    checkpoint = load_checkpoint(paused.last_saved_ckpt)
    assert checkpoint['completed'] is False and checkpoint['total_epochs'] == 3
    assert checkpoint['epoch'] == 1
    assert checkpoint['scheduler_state_dict']['last_epoch'] == 1
    resumed = trainer(tmp_path / 'paused', None)
    start = resumed.resume_from_checkpoint(paused.last_saved_ckpt)
    assert start == 2
    resumed.train(start_epoch=start, model_prefix='control')
    actual = load_checkpoint(resumed.last_saved_ckpt)
    assert actual['completed'] is True
    for group in ('backbone_state_dict', 'head_state_dict'):
        assert all(torch.equal(v, actual[group][k]) for k,v in expected[group].items())
    assert actual['scheduler_state_dict'] == expected['scheduler_state_dict']
    for key, state in expected['optimizer_state_dict']['state'].items():
        for name, value in state.items():
            assert torch.equal(value, actual['optimizer_state_dict']['state'][key][name])


def test_planned_pause_never_publishes_completed_parent(aligned_cfg, monkeypatch):
    current = copy.deepcopy(aligned_cfg)
    current['campaign']['pretrain_epochs'] = 3
    current['runtime'] = {'stop_after_epoch': 1}
    monkeypatch.setattr(w, '_formal_device', lambda cfg: torch.device('cpu'))
    result = w.run_training_matrix(current, 'pretrain')[0]
    assert result['status'] == 'paused_epoch_limit' and result['completed'] is False
    assert result['total_epochs'] == 3 and result['next_epoch'] == 2
    final = w.checkpoint_path(current, 'pretrain', 'arcface', 8972)
    assert not final.exists()
    assert not final.with_name(final.stem + '_hash.json').exists()


def test_control_plan_does_not_mutate_shared_settings(cfg, tmp_path):
    before = copy.deepcopy(cfg)
    result = controls.make_parent_control(cfg, tmp_path/'reference/model.pt', 'a'*64, tmp_path/'new')
    assert cfg == before
    assert result['training']['weight_decay'] == 0
    assert result['campaign']['pretrain_epochs'] == 40
    assert result['runtime']['stop_after_epoch'] == 2
    with pytest.raises(ValueError, match='separate'):
        controls.make_parent_control(cfg, tmp_path/'reference/model.pt', 'a'*64, tmp_path/'reference')
    with pytest.raises(ValueError, match='NEW parent'):
        controls.validate_control_resume(result, tmp_path/'reference/model.pt')


def test_existing_control_keeps_original_declaration_across_execution_limits(cfg, tmp_path):
    root=tmp_path/'control'
    before=controls.make_parent_control(cfg,tmp_path/'ref/model.pt','a'*64,root)
    original=copy.deepcopy(before['parent_control'])
    original['module_sha256']='b'*64
    root.mkdir()
    (root/'control.json').write_text(json.dumps(original))
    after=controls.make_parent_control(cfg,tmp_path/'ref/model.pt','a'*64,root,stop_after_epoch=None,keep_raw_results=False)
    assert after['parent_control']==original
    assert after['runtime']['control_driver_sha256']!='b'*64
    assert after['runtime']['stop_after_epoch'] is None
    assert after['project']['keep_raw_results'] is False
    with pytest.raises(ValueError,match='incompatible'):
        controls.make_parent_control(cfg,tmp_path/'ref/model.pt','c'*64,root)


def test_replacement_parent_is_not_baseline_or_old_model_reuse(cfg, tmp_path):
    from modules import selected_campaign as s
    cfg = copy.deepcopy(cfg)
    cfg['campaign'].update(parent_binding={'path':'new.pt', 'sha256':'a'*64, 'reference':{}},
                           parent_checkpoint='new.pt', parent_sha256='a'*64, purpose='diagnostic_control')
    plan = [dict(dataset='survface', recipe='full_fr_resize_only', heads=['magface'], seeds=[8972])]
    with pytest.raises(ValueError, match='diagnostic_control'):
        s.make_selection(cfg, plan, tmp_path/'run', tmp_path/'diagnostic', purpose='paper_baseline')
    selection = s.make_selection(cfg, plan, tmp_path/'run', tmp_path/'diagnostic', purpose='diagnostic_control')
    assert selection['reproduction_scope']['calibration_unseen_in_fr'] is False
    assert len(selection['rows']) == 1
    assert s.default_references(selection) == {}
    assert s.parent_digest(cfg) == 'a'*64
    cfg['campaign']['parent_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='binding'):
        s.parent_digest(cfg)


@pytest.mark.parametrize('name', ['03_finetune.ipynb', '04_export_and_validate.ipynb'])
@pytest.mark.parametrize('campaign', ['parent_control', 'finetune_wd_zero_control'])
def test_followup_notebook_blocks_without_completed_parent_selection(cfg, tmp_path, monkeypatch, name, campaign):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    monkeypatch.setenv('FACE_MODEL_ROOT',str(tmp_path/'models'))
    monkeypatch.setattr(w,'resolve_settings',lambda *a,**kw: copy.deepcopy(cfg))
    notebook = json.loads((root/'fine-tune/notebooks'/name).read_text(encoding='utf-8'))
    source = ''.join(next(c for c in notebook['cells'] if c['cell_type']=='code')['source'])
    source = source.replace('CAMPAIGN = "finetune_wd_zero_control"',f'CAMPAIGN = "{campaign}"')
    source = source.replace('CAMPAIGN = "paper_baseline"',f'CAMPAIGN = "{campaign}"')
    with pytest.raises(FileNotFoundError):
        exec(compile(source,name,'exec'),{})


def test_finetune_wd_zero_selection_changes_only_finetune_weight_decay(tmp_path):
    settings_path = Path("F:/face_datasets/.state/facex/shared_v1_boundary/notebook_settings_mobilefacenet.json")
    ref_sel = Path("F:/models/facex/diagnostics/parent_weight_decay_zero_v1/followup/selection.json")
    if not (settings_path.is_file() and ref_sel.is_file()):
        pytest.skip("Pinned WD=0 Parent followup artifacts unavailable")
    from modules import selected_campaign as s
    root = Path(__file__).resolve().parents[2]
    base = w.resolve_settings(root, settings_path, require_saved=True)
    selection = controls.build_finetune_wd_zero_selection(base)
    s.validate_selection(selection)
    assert selection["purpose"] == "diagnostic_control"
    assert selection["settings"]["training"]["weight_decay"] == 0.0
    assert selection["settings"]["training"]["save_freq"] == 1
    assert selection["reproduction_scope"]["finetune_control"]["schema"] == controls.FINETUNE_WD_ZERO_SCHEMA
    assert selection["reproduction_scope"]["finetune_control"]["reference_selection_sha256"] == "472022d0d95064fff9346e62b79062bcc0be5c408afa5b7618504c79258cc366"
    plan = s.inspect_selection(selection)
    assert len(plan) == 1 and plan[0]["status"] == "train_new"
    diag = controls.diagnose_finetune_wd_zero_epochs(selection)
    assert [r["epoch"] for r in diag["epochs"]] == [1, 2, 5, 10, 40]
    assert diag["parent_before_finetune"]["norm_lt_10"] == 0
    assert diag["epochs"][0]["reference_wd_0001"]["norm_lt_10"] == 241
    assert diag["epochs"][0]["reference_wd_0001"]["sha256"] == "f60a48662a9abc6afb3bb58d7231dc90529cf7ebb199d30eb8bb888857595699"

    # Zero-tolerance check: 0.0001 margin_am or 0.00002 scale drift must be rejected.
    for mut in (
        lambda c: c["heads"]["magface"].update(margin_am=0.0001),
        lambda c: c["heads"]["arcface"].update(margin_am=0.0001),
        lambda c: c["heads"]["magface"].update(scale=64.00002),
    ):
        drifted = copy.deepcopy(base)
        mut(drifted)
        with pytest.raises(ValueError):
            controls.build_finetune_wd_zero_selection(drifted)
        tampered_sel = copy.deepcopy(selection)
        mut(tampered_sel["settings"])
        with pytest.raises(ValueError):
            s.settings_for(tampered_sel, tampered_sel["rows"][0])

    # Schema tampering must fail common selection validation and settings_for.
    bad_schema = copy.deepcopy(selection)
    bad_schema["settings"]["campaign"]["finetune_control"]["schema"] = "tampered-schema"
    with pytest.raises(ValueError, match="schema"):
        s.validate_selection(bad_schema)
    with pytest.raises(ValueError, match="schema"):
        s.settings_for(bad_schema, bad_schema["rows"][0])

    # Existing ft_wd=0.0001 checkpoints (epoch 1 and epoch 40) must be blocked in resume and _validate.
    row = selection["rows"][0]
    cfg = s.settings_for(selection, row)
    ref_ckpt_40 = Path(selection["settings"]["campaign"]["finetune_control"]["reference_checkpoint"])
    ref_ckpt_1 = ref_ckpt_40.with_name("finetune_mobilefacenet_1024d_magface_epoch_1.pt")
    for old_ckpt in (ref_ckpt_1, ref_ckpt_40):
        resumed_plan = s.inspect_selection(selection, {row["run_id"]: str(old_ckpt)})
        assert resumed_plan[0]["status"] == "blocked"
        assert "weight_decay" in resumed_plan[0]["reason"]
    with pytest.raises(ValueError, match="weight_decay"):
        s._validate(ref_ckpt_40, cfg, row)

    # A checkpoint whose config claims weight_decay=0.0 but optimizer param_groups has 0.0001 must be rejected.
    fake_ckpt = tmp_path / "fake_epoch_40.pt"
    payload = s.load_checkpoint(str(ref_ckpt_40))
    payload["config"] = s._expected(cfg, row)
    payload["config_hash"] = s.compute_config_hash(payload["config"])
    payload.setdefault("extra_meta", {})["execution_settings"] = cfg
    torch.save(payload, fake_ckpt)
    with pytest.raises(ValueError, match="param_groups weight_decay"):
        s._validate(fake_ckpt, cfg, row)

    # Out-of-scope target rows under finetune_control must be rejected in common selection and inspect_selection.
    out_of_scope_plans = [
        [dict(dataset="tinyface", recipe="resize_only", heads=["magface"], seeds=[8972])],
        [dict(dataset="scface", recipe="two_eye_alignment", heads=["magface"], seeds=[8972])],
        [dict(dataset="survface", recipe="resize_only", heads=["magface"], seeds=[8972])],
        [dict(dataset="survface", recipe="full_fr_resize_only", heads=["arcface"], seeds=[8972])],
        [dict(dataset="survface", recipe="full_fr_resize_only", heads=["magface"], seeds=[1234])],
    ]
    for bad_plan in out_of_scope_plans:
        with pytest.raises(ValueError, match="exactly one target row"):
            s.make_selection(selection["settings"], bad_plan, tmp_path / "bad_out",
                             selection["diagnostic_root"], purpose="diagnostic_control")
        hand_sel = copy.deepcopy(selection)
        hand_sel["rows"] = s.expand_plan(bad_plan)
        blocked_plan = s.inspect_selection(hand_sel)
        assert blocked_plan[0]["status"] == "blocked"
        assert "exactly one target row" in blocked_plan[0]["reason"]


def test_canonical_inventory_file_only_written_for_finetune_wd_zero_control(tmp_path, monkeypatch):
    from modules import selected_campaign as s
    shared_root = tmp_path / "shared_export_root"
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(s, "validate_selection_execution", lambda *a: None)
    monkeypatch.setattr(s, "settings_for", lambda *a, **kw: {})
    monkeypatch.setattr(s, "_estimated_checkpoint_bytes", lambda *a: 0)

    def fake_export(selection, row, cfg, path):
        return {**row, "completed": True, "export": str(tmp_path / f"{row['run_id']}.pt"),
                "sha256": "a" * 64, "source_checkpoint": str(path), "source_sha256": "b" * 64,
                "evaluation_datasets": [row["dataset"]], "training_dataset": row["dataset"]}

    monkeypatch.setattr(s, "export_selected", fake_export)

    sel1 = s.make_selection(base, [dict(dataset="scface", recipe="two_eye_alignment", heads=["magface"], seeds=[8972])],
                            shared_root, tmp_path / "diag", purpose="local_protocol_baseline")
    sel2 = s.make_selection(base, [dict(dataset="tinyface", recipe="pfld_five_point_alignment", heads=["magface"], seeds=[8972])],
                            shared_root, tmp_path / "diag", purpose="local_protocol_baseline")
    for sel in (sel1, sel2):
        row = sel["rows"][0]
        monkeypatch.setattr(s, "inspect_selection", lambda *_a, _r=row, **_kw: [{**_r, "status": "reuse_completed",
            "checkpoint": str(tmp_path / "c.pt"), "target": str(tmp_path / "c.pt")}])
        s.execute_selection(sel, "export")
    assert len(list((shared_root / "inventories").glob("*.json"))) == 2
    assert not (shared_root / "inventory.json").exists()

    wd0_root = tmp_path / "wd0_root"
    wd0_sel = copy.deepcopy(sel1)
    wd0_sel["root"] = str(wd0_root)
    wd0_sel["settings"]["campaign"]["finetune_control"] = {"schema": controls.FINETUNE_WD_ZERO_SCHEMA}
    row = wd0_sel["rows"][0]
    monkeypatch.setattr(s, "inspect_selection", lambda *_a, _r=row, **_kw: [{**_r, "status": "reuse_completed",
        "checkpoint": str(tmp_path / "c.pt"), "target": str(tmp_path / "c.pt")}])
    s.execute_selection(wd0_sel, "export")
    hashed = list((wd0_root / "inventories").glob("*.json"))
    assert len(hashed) == 1
    assert (wd0_root / "inventory.json").is_file()
    assert (wd0_root / "inventory.json").read_text(encoding="utf-8") == hashed[0].read_text(encoding="utf-8")
