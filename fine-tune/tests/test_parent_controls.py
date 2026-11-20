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
def test_followup_notebook_blocks_without_completed_parent_selection(cfg, tmp_path, monkeypatch, name):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    monkeypatch.setenv('FACE_MODEL_ROOT',str(tmp_path/'models'))
    monkeypatch.setattr(w,'resolve_settings',lambda *a,**kw: copy.deepcopy(cfg))
    notebook = json.loads((root/'fine-tune/notebooks'/name).read_text(encoding='utf-8'))
    source = ''.join(next(c for c in notebook['cells'] if c['cell_type']=='code')['source'])
    source = source.replace('CAMPAIGN = "paper_baseline"','CAMPAIGN = "parent_control"')
    with pytest.raises(FileNotFoundError):
        exec(compile(source,name,'exec'),{})
