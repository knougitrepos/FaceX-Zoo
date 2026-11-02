import json
import os
from pathlib import Path
import socket
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.alignment import ResizePreprocessor, verify_alignment
from modules.alignment_resume import resume_alignment, rows, os_guard


def fixture(tmp_path):
    index = tmp_path / 'source.txt'
    index.write_text('a.png 0\nb.png 1\nc.png 2\n')
    for i, name in enumerate(['a', 'b', 'c']):
        cv2.imwrite(str(tmp_path / (name + '.png')), np.full((15, 13, 3), 40+i, np.uint8))
    provider = ResizePreprocessor(dict(method='resize_only', interpolation='bicubic'))
    return index, tmp_path, tmp_path / 'out', provider


def test_new_resume_and_completed_immutable(tmp_path):
    args = fixture(tmp_path)
    report = resume_alignment(*args)
    assert report['completed'] and report['regenerated_images'] == 3
    verify_alignment(args[2]/'report.json', args[0], args[3].policy)
    before = (args[2]/'report.json').read_bytes()
    resume_alignment(*args)
    assert (args[2]/'report.json').read_bytes() == before


def test_repeated_interrupt_reuses_verified_journal(tmp_path):
    args = fixture(tmp_path)
    original = args[3]
    class Interrupt:
        policy = original.policy
        def __init__(self): self.calls = 0
        def __call__(self, image):
            self.calls += 1
            if self.calls == 2: raise KeyboardInterrupt()
            return original(image)
    with pytest.raises(KeyboardInterrupt): resume_alignment(*args[:3], Interrupt())
    target = args[2]/'images/000000/00000000.png'
    stamp = target.stat().st_mtime_ns
    with pytest.raises(KeyboardInterrupt): resume_alignment(*args[:3], Interrupt())
    report = resume_alignment(*args)
    assert report['reused_images'] == 2 and report['regenerated_images'] == 1
    assert target.stat().st_mtime_ns == stamp
    verify_alignment(args[2]/'report.json', args[0], original.policy)


def test_legacy_truncated_manifest_orphan_and_corrupt_image(tmp_path):
    args = fixture(tmp_path)
    index, root, out, provider = args
    images = out/'images/000000'; images.mkdir(parents=True)
    # Legacy row has no immutable start contract; current pixels must match.
    from modules.alignment import sha256
    crop, _ = provider(cv2.imread(str(root/'a.png')))
    cv2.imwrite(str(images/'00000000.png'), crop)
    stamp = (images/'00000000.png').stat().st_mtime_ns
    (images/'00000001.png').write_bytes(b'broken PNG')
    crop, _ = provider(cv2.imread(str(root/'c.png')))
    cv2.imwrite(str(images/'00000002.png'), crop)
    row = dict(index=0,source=str(root/'a.png'),label=0,source_sha256=sha256(root/'a.png'))
    original = json.dumps(row).encode()+b'\n{"index":1,"x":"\xe3'
    (out/'manifest.jsonl').write_bytes(original)
    (out/'train_list.txt').write_text('incomplete')
    report = resume_alignment(*args)
    assert report['completed'] and report['reused_images'] == 2
    assert (images/'00000000.png').stat().st_mtime_ns == stamp
    attempt = Path(report['recovery_attempt'])
    assert (attempt/'previous-manifest.jsonl').read_bytes() == original
    assert (attempt/'replaced/000000/00000001.png').read_bytes() == b'broken PNG'
    verify_alignment(out/'report.json', index, provider.policy)


def test_changed_source_and_contract_rejected(tmp_path):
    args = fixture(tmp_path)
    class Stop:
        policy = args[3].policy
        def __call__(self, image): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): resume_alignment(*args[:3], Stop())
    args[0].write_text('a.png 1\n')
    with pytest.raises(ValueError, match='contract differs'): resume_alignment(*args)


def test_live_foreign_and_dead_lock(tmp_path):
    args = fixture(tmp_path)
    lock = args[2].with_name('out.lock')
    lock.write_text(json.dumps(dict(pid=os.getpid(),host=socket.gethostname())))
    with pytest.raises(RuntimeError, match='alive'): resume_alignment(*args)
    lock.write_text(json.dumps(dict(pid=99999999,host='other-host')))
    with pytest.raises(RuntimeError, match='foreign'): resume_alignment(*args)
    lock.write_text(json.dumps(dict(pid=99999999,host=socket.gethostname())))
    assert resume_alignment(*args)['completed']
    assert list(tmp_path.glob('out.lock.stale-*'))


def test_interior_corruption_and_bounded_rejected(tmp_path):
    import io
    with pytest.raises(json.JSONDecodeError): list(rows(io.BytesIO(b'bad\n')))
    with pytest.raises(ValueError, match='max_images'): resume_alignment(*fixture(tmp_path), max_images=2)


def test_os_guard_blocks_concurrent_owner(tmp_path):
    with os_guard(tmp_path/'guard'):
        with pytest.raises(OSError):
            with os_guard(tmp_path/'guard'): pass


def test_real_process_death_then_recovery(tmp_path):
    import subprocess
    args = fixture(tmp_path)
    code = '''
import os,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from modules.alignment import ResizePreprocessor
from modules.alignment_resume import resume_alignment
p=ResizePreprocessor(dict(method='resize_only',interpolation='bicubic'))
class Crash:
    policy=p.policy
    calls=0
    def __call__(self,image):
        self.calls+=1
        if self.calls==2: os._exit(77)
        return p(image)
root=Path(sys.argv[2])
resume_alignment(root/'source.txt',root,root/'out',Crash())
'''
    run = subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve().parents[1]), str(tmp_path)])
    assert run.returncode == 77
    assert (tmp_path/'out.lock').exists()
    result = resume_alignment(*args)
    assert result['completed'] and result['reused_images'] == 1
    verify_alignment(args[2]/'report.json', args[0], args[3].policy)


def test_changed_bytes_block_and_missing_marker_recovers(tmp_path):
    args = fixture(tmp_path)
    report = resume_alignment(*args)
    (args[2]/'COMPLETED').unlink()
    resume_alignment(*args)
    assert (args[2]/'COMPLETED').exists()
    (tmp_path/'a.png').write_bytes(b'changed source')
    with pytest.raises(ValueError): resume_alignment(*args)
    assert json.loads((args[2]/'report.json').read_text()) == report


def test_launcher_passes_resume_and_rejects_bounded(tmp_path, monkeypatch):
    from modules import process_runner, notebook_workflow
    monkeypatch.setattr(notebook_workflow, 'save_settings', lambda p,c: None)
    commands=[]
    monkeypatch.setattr(process_runner, '_stream_alignment', lambda cmd,*args: commands.append(cmd))
    cfg={'paths':{'launch_dir':str(tmp_path/'launch')}}
    process_runner.launch_alignment(tmp_path,cfg,'vggface2',max_images=None,resume=True)
    assert '--resume' in commands[0]
    with pytest.raises(ValueError):
        process_runner.launch_alignment(tmp_path,cfg,'vggface2',max_images=32,resume=True)


def test_child_traceback_is_saved_and_shown(tmp_path, capsys):
    from modules.process_runner import _stream_alignment
    log = tmp_path/'console.log'
    with pytest.raises(RuntimeError, match='console.log'):
        _stream_alignment([sys.executable, '-u', '-c',
                           "print('progress', flush=True); raise ValueError('child failure')"], tmp_path, log)
    assert 'progress' in log.read_text()
    assert 'ValueError: child failure' in log.read_text()
    assert 'ValueError: child failure' in capsys.readouterr().out


def test_child_stdout_success(tmp_path):
    from modules.process_runner import _stream_alignment
    log = tmp_path/'console.log'
    _stream_alignment([sys.executable, '-u', '-c', "print('done')"], tmp_path, log)
    assert log.read_text().strip() == 'done'
