import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import selected_campaign as s
from test_notebook_workflow import cfg


def test_selection_expansion_is_incremental_and_rejects_ambiguous_entries():
    plan = s.default_plan()
    rows = s.expand_plan(plan)
    assert len(rows) == 12
    old = {r['run_id'] for r in rows}
    plan[0]['seeds'].append(12)
    assert len(s.expand_plan(plan)) == 16
    assert old <= {r['run_id'] for r in s.expand_plan(plan)}
    for bad in ([plan[0], plan[0]], [dict(plan[0], dataset='lfw')],
                [dict(plan[0], seeds=[True])], [dict(plan[0], heads=[])],
                [dict(plan[-1], recipe='arcface_continuous_margin', heads=['magface'])]):
        with pytest.raises(ValueError):
            s.expand_plan(bad)


def test_immutable_selection_preserves_old_plan(tmp_path):
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2])
    selection = s.make_selection(base, s.default_plan(), tmp_path, tmp_path / 'diagnostic')
    path = s.save_selection(tmp_path / 'plan.json', selection)
    assert s.load_selection(path) == selection
    force = copy.deepcopy(base)
    force.setdefault('data_integrity', {})['force_reverify'] = True
    assert s.make_selection(force, s.default_plan(), tmp_path, tmp_path / 'diagnostic') == selection
    s.save_selection(path, selection)
    changed = copy.deepcopy(selection)
    changed['reuse_completed'] = False
    with pytest.raises(FileExistsError):
        s.save_selection(path, changed)
    assert s.load_selection(path) == selection


def test_resume_is_explicit_and_no_missing_reference_fallback(tmp_path, monkeypatch):
    row = s.expand_plan(s.default_plan())[0]
    target = tmp_path / 'job/final.pt'
    selection = dict(rows=[row], references={}, reuse_completed=True)
    monkeypatch.setattr(s, 'settings_for', lambda *a, **kw: {})
    monkeypatch.setattr(s, 'final_path', lambda *a: target)
    monkeypatch.setattr(s, '_validate', lambda *a, **kw: None)
    assert s.inspect_selection(selection)[0]['status'] == 'train_new'
    target.parent.mkdir()
    recovery = target.with_name('recovery.pt')
    recovery.touch()
    assert s.inspect_selection(selection)[0]['status'] == 'blocked'
    assert s.inspect_selection(selection, {row['run_id']: str(recovery)})[0]['status'] == 'resume'
    selection['references'][row['run_id']] = dict(path=str(tmp_path / 'missing.pt'), sha256='0'*64)
    assert s.inspect_selection(selection)[0]['status'] == 'blocked'
    with pytest.raises(ValueError, match='Resume keys'):
        s.inspect_selection(selection, {'old_run_id': str(recovery)})


def test_failed_job_stops_campaign_and_records_unstarted(tmp_path, monkeypatch):
    rows = s.expand_plan(s.default_plan())[:2]
    selection = dict(root=str(tmp_path), rows=rows)
    monkeypatch.setattr(s, 'validate_selection_execution', lambda *a: None)
    plan = [{**r, 'status': 'train_new', 'checkpoint': str(tmp_path / r['run_id'] / 'final.pt'),
             'target': str(tmp_path / r['run_id'] / 'final.pt')} for r in rows]
    monkeypatch.setattr(s, 'inspect_selection', lambda *a, **kw: plan)
    monkeypatch.setattr(s, 'settings_for', lambda *a, **kw: {})
    monkeypatch.setattr(s, '_estimated_checkpoint_bytes', lambda *a: 0)
    called = []
    def fail(cfg, row, resume):
        called.append(row['run_id'])
        raise RuntimeError('retained traceback')
    monkeypatch.setattr(s, '_train', fail)
    with pytest.raises(RuntimeError, match='retained traceback'):
        s.execute_selection(selection, 'finetune')
    assert called == [rows[0]['run_id']]
    result = s.read_json(next((tmp_path / 'attempts').glob('*/result.json')))
    assert result['status'] == 'failed' and len(result['not_started']) == 1
    assert not list((tmp_path / 'inventories').glob('*'))


def test_export_blocks_partial_selection_before_writing(tmp_path, monkeypatch):
    row = s.expand_plan(s.default_plan())[0]
    monkeypatch.setattr(s, 'validate_selection_execution', lambda *a: None)
    monkeypatch.setattr(s, 'inspect_selection', lambda *a, **kw: [{**row, 'status': 'train_new'}])
    with pytest.raises(ValueError, match='every selected model'):
        s.execute_selection(dict(root=str(tmp_path)), 'export')
    assert not (tmp_path / 'exports').exists()


def test_actual_pinned_selection_and_incompatible_checkpoint(tmp_path):
    path = Path('F:/face_datasets/.state/facex/shared_v1_boundary/notebook_settings_mobilefacenet.json')
    if not path.exists():
        pytest.skip('Pinned local artifacts unavailable')
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2], path)
    selection = s.make_selection(base, s.default_plan(), tmp_path,
        'F:/models/facex/diagnostics/' + s.ALIGNMENT_CONTROL)
    selection['references'] = s.default_references(selection)
    plan = s.inspect_selection(selection)
    assert [p for p in plan if p['status'] == 'blocked'] == []
    assert sum(p['status'] == 'reuse_completed' for p in plan) == 6
    assert sum(p['status'] == 'train_new' for p in plan) == 6
    diagnostic = s.make_selection(base, [dict(dataset='survface', recipe='arcface_continuous_margin',
        heads=['arcface'], seeds=[8972])], tmp_path / 'diagnostic_run', selection['diagnostic_root'], purpose='diagnostic_control')
    new = diagnostic['rows'][0]
    cfg = s.settings_for(diagnostic, new)
    source = s._recipe_source(diagnostic, new)
    with pytest.raises(ValueError, match='Selected checkpoint differs'):
        s._validate(source['path'], cfg, new)
    # A different seed never reuses seed-8972 just because its head matches.
    changed = dict(new, seed=123)
    with pytest.raises(ValueError, match='Seed mismatch'):
        s._validate(source['path'], s.settings_for(diagnostic, changed), changed)


def test_default_is_original_losses_and_continuous_requires_diagnostic_purpose(tmp_path):
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2])
    assert all(r['recipe'] != 'arcface_continuous_margin' for r in s.expand_plan(s.default_plan()))
    control = [dict(dataset='survface', recipe='arcface_continuous_margin', heads=['arcface'], seeds=[8972])]
    with pytest.raises(ValueError, match='diagnostic_control'):
        s.make_selection(base, control, tmp_path, tmp_path)
    diagnostic = s.make_selection(base, control, tmp_path, tmp_path, purpose='diagnostic_control')
    s.validate_selection_execution(diagnostic)
    diagnostic['purpose'] = 'local_protocol_baseline'
    with pytest.raises(ValueError, match='diagnostic_control'):
        s.validate_selection_execution(diagnostic)


def test_old_in_memory_selection_cannot_launch_or_export(tmp_path, monkeypatch):
    from modules import process_runner
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2])
    selection = s.make_selection(base, s.default_plan(), tmp_path / 'outputs', tmp_path)
    selection.pop('purpose')
    selection.pop('reproduction_scope')
    selection['schema'] = 'facex-selection-v1'
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(selection), encoding='utf-8')
    assert s.load_selection(path) == selection
    monkeypatch.setattr(process_runner.subprocess, 'Popen', lambda *a, **kw: pytest.fail('spawned stale campaign'))
    with pytest.raises(ValueError, match='read-only'):
        process_runner.launch_campaign(tmp_path, base, 'finetune', selection=selection)
    with pytest.raises(ValueError, match='read-only'):
        s.execute_selection(selection, 'export')
    assert not (tmp_path / 'outputs').exists()


def test_reproduction_review_exposes_split_and_dimension_without_training(tmp_path):
    from modules.ucface_reference import reproduction_review, require_reproduction_ready
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2])
    base['finetune_profiles']['survface'].update(num_classes=4255, expected_images=176205)
    selection = s.make_selection(base, s.default_plan(), tmp_path / 'outputs', tmp_path)
    report = reproduction_review(selection)
    assert report['training_ready'] is False and report['exact_ucface_reproduction'] is False
    assert report['populations']['survface']['counts_match'] is False
    assert report['embedding_dimension']['main_paper'] == 1024
    assert report['embedding_dimension']['author_recollection'] == 512
    with pytest.raises(ValueError, match='not verified'):
        require_reproduction_ready(selection)
    assert not (tmp_path / 'outputs').exists()


def test_existing_export_keeps_uid_bytes(tmp_path):
    path = Path('F:/face_datasets/.state/facex/shared_v1_boundary/notebook_settings_mobilefacenet.json')
    if not path.exists():
        pytest.skip('Pinned local artifacts unavailable')
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2], path)
    selection = s.make_selection(base, [dict(dataset='scface', recipe='two_eye_alignment', heads=['arcface'], seeds=[8972])],
        tmp_path, 'F:/models/facex/diagnostics/' + s.ALIGNMENT_CONTROL)
    selection['references'] = s.default_references(selection)
    row = selection['rows'][0]
    cfg = s.settings_for(selection, row)
    ref = selection['references'][row['run_id']]
    result = s.export_selected(selection, row, cfg, ref['path'])
    assert result['sha256'] == 'cbb3f65d8969c7c2ec4c822ea3cfcc15085c55b410a19f09c214111edec811fe'
    assert not (tmp_path / 'exports').exists()


def test_new_export_publishes_verified_geometry_and_rejects_tampering(tmp_path, monkeypatch):
    path = Path('F:/face_datasets/.state/facex/shared_v1_boundary/notebook_settings_mobilefacenet.json')
    if not path.exists():
        pytest.skip('Pinned local artifacts unavailable')
    base = s.w.resolve_settings(Path(__file__).resolve().parents[2], path)
    selection = s.make_selection(base, [dict(dataset='scface', recipe='two_eye_alignment', heads=['arcface'], seeds=[8972])],
        tmp_path, 'F:/models/facex/diagnostics/' + s.ALIGNMENT_CONTROL)
    row = selection['rows'][0]
    cfg = s.settings_for(selection, row)
    source = s._recipe_source(selection, row)['path']
    # Exercise the new publication branch with a real completed source in an isolated test directory.
    monkeypatch.setattr(s, 'default_references', lambda *a: {})
    result = s.export_selected(selection, row, cfg, source)
    assert Path(result['export']).is_relative_to(tmp_path / 'exports')
    assert s.export_selected(selection, row, cfg, source) == result
    geometry = Path(result['export']).parent / 'affine.csv'
    geometry.write_bytes(geometry.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        s.export_selected(selection, row, cfg, source)


def test_selected_training_resume_matches_uninterrupted(cfg, tmp_path, monkeypatch):
    """Exercise the new orchestration with the real Trainer on eight synthetic images."""
    import torch
    from modules.trainer import Trainer, TrainingStopped
    cfg = copy.deepcopy(cfg)
    cfg['dataset']['val_list'] = None
    cfg['training'].update(epoches=2, torch_threads=1, batch_size=2, save_freq=1)
    cfg['campaign'].update(parent_checkpoint='synthetic_parent.pt')
    row = dict(run_id='synthetic', dataset='VGGFace2', head='cosface', seed=8972, recipe='resize_only')
    model = s.w.build_model(cfg['backbone'], dict(cfg['heads']['cosface'], num_class=2))
    parent = dict(backbone_state_dict=model.backbone.state_dict(), config={'code_fingerprints':s.w.code_fingerprints()})
    monkeypatch.setattr(s, 'verify_parent_checkpoint', lambda *a, **kw: (parent, s.PARENT))
    monkeypatch.setattr(s.w, 'validate_parent_preprocessing', lambda *a: None)
    monkeypatch.setattr(s.w, '_formal_device', lambda *a: torch.device('cpu'))
    cfg['paths']['checkpoint_dir'] = str(tmp_path / 'reference')
    reference = s._train(cfg, row)
    expected = s.load_checkpoint(str(reference))
    cfg['paths']['checkpoint_dir'] = str(tmp_path / 'interrupted')
    stop = tmp_path / 'stop.request'
    cfg['runtime'] = dict(stop_file=str(stop))
    original = Trainer._progress
    def request_stop(self, phase, **values):
        original(self, phase, **values)
        if phase == 'training' and values.get('batches_completed') == 0:
            stop.touch()
    monkeypatch.setattr(Trainer, '_progress', request_stop)
    with pytest.raises(TrainingStopped):
        s._train(cfg, row)
    final = s.final_path(cfg, row)
    assert not final.exists()
    progress = s.read_json(final.parent / 'progress.json')
    recovery = progress['last_checkpoint']
    stop.unlink()
    monkeypatch.setattr(Trainer, '_progress', original)
    resumed = s._train(cfg, row, recovery)
    actual = s.load_checkpoint(str(resumed))
    for key in ('backbone_state_dict', 'head_state_dict'):
        assert expected[key].keys() == actual[key].keys()
        for name, tensor in expected[key].items():
            assert torch.equal(tensor, actual[key][name]), name
    assert actual['completed'] is True and actual['epoch'] == 2
