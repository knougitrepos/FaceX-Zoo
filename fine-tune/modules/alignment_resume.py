"""Explicit, journaled recovery of incomplete preprocessing; never edit raw inputs."""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import uuid

import cv2
import numpy as np

from .alignment import read_rows, sha256, verify_alignment
from .dataset import resolve_image_path
from .shared_storage import artifact_lock, write_derivation


def atomic_json(path, value):
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextmanager
def recovery_lock(out):
    # OS locks release on process death, including reboot.
    out.parent.mkdir(parents=True, exist_ok=True)
    with os_guard(out.with_name(out.name + '.recovery.guard')):
        lock = out.with_name(out.name + '.lock')
        if lock.exists():
            import psutil
            record = json.loads(lock.read_text(encoding='utf-8'))
            if record.get('host') != socket.gethostname() or type(record.get('pid')) is not int:
                raise RuntimeError('Cannot reclaim foreign/invalid preprocessing lock')
            if psutil.pid_exists(record['pid']):
                raise RuntimeError('Preprocessing lock owner is still alive; do not resume')
            os.replace(lock, lock.with_name(lock.name + '.stale-' + uuid.uuid4().hex))
        with artifact_lock(out):
            yield


@contextmanager
def os_guard(path):
    with path.open('a+b') as f:
        f.seek(0, 2)
        if not f.tell():
            f.write(b'0')
            f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def rows(stream):
    """Only an unterminated final record may be discarded after a crash."""
    expected = 0
    for line in stream:
        if not line.endswith(b'\n'):
            if stream.read(1):
                raise ValueError('Malformed interior manifest record')
            return
        row = json.loads(line)
        if row.get('index') != expected:
            raise ValueError('Recovery manifest index/order differs')
        expected += 1
        yield row


def resume_alignment(source_list, raw_root, output_dir, aligner, max_images=None):
    """Validate from row zero, reuse verified PNGs, journal before final publication."""
    if max_images is not None:
        raise ValueError('Resume requires max_images=None (full preparation)')
    out = Path(output_dir).resolve()
    source_list, root = Path(source_list).resolve(), Path(raw_root).resolve()
    with recovery_lock(out):
        # Completed artifacts are immutable, including policy mismatches.
        if (out / 'COMPLETED').exists() or (out / 'report.json').exists():
            if (out / 'report.json').exists():
                report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
                if report.get('completed'):
                    # verify_alignment observes the held lock; only this reader bypasses that guard.
                    return _verify_completed(out, source_list, root, aligner)
            if (out / 'COMPLETED').exists():
                raise ValueError('Inconsistent completion markers; preserve for review')
        out.mkdir(parents=True, exist_ok=True)
        contract = dict(schema='alignment-resume-v1', source_list=str(source_list),
                        source_list_sha256=sha256(source_list), raw_root=str(root),
                        policy=aligner.policy, recovery_code_sha256=sha256(__file__))
        state = out / 'resume_contract.json'
        if state.exists():
            if json.loads(state.read_text(encoding='utf-8')) != contract:
                raise ValueError('Resume source/configuration/code contract differs')
        else:
            atomic_json(state, contract)
        history = out / 'recovery'
        history.mkdir(exist_ok=True)
        previous = sorted(history.glob('*/manifest.jsonl'))
        base = out / 'manifest.jsonl'
        candidates = ([(base, False)] if base.exists() else []) + [(p, True) for p in previous]
        attempt = history / (str(time.time_ns()) + '-' + uuid.uuid4().hex)
        attempt.mkdir()
        total = sum(1 for _ in read_rows(source_list))
        success = reused = processed = 0
        seen = set()
        start = last = time.monotonic()
        with ExitStack() as stack:
            readers = [(iter(rows(stack.enter_context(p.open('rb')))), trusted) for p, trusted in candidates]
            records = stack.enter_context((attempt / 'manifest.jsonl').open('w', encoding='utf-8'))
            targets = stack.enter_context((attempt / 'train_list.txt').open('w', encoding='utf-8'))
            for i, (name, label) in enumerate(read_rows(source_list)):
                source = Path(resolve_image_path(str(root), name)).resolve()
                target = out / 'images' / f'{i // 1000:06d}' / f'{i:08d}.png'
                prior, trusted = None, False
                for reader, trust in readers:
                    entry = next(reader, None)
                    if entry is not None:
                        if entry.get('source') != str(source) or entry.get('label') != label:
                            raise ValueError('Resume source mapping differs')
                        prior, trusted = entry, trust
                if str(source) in seen:
                    raise ValueError('Duplicate source image')
                seen.add(str(source))
                row = dict(index=i, source=str(source), label=label)
                try:
                    data = source.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if prior and prior.get('source_sha256') and prior['source_sha256'] != digest:
                        raise RuntimeError('Resume source bytes changed')
                    row['source_sha256'] = digest
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError('Image decode failed')
                    old = target.read_bytes() if target.exists() else None
                    decoded = cv2.imdecode(np.frombuffer(old, np.uint8), cv2.IMREAD_COLOR) if old else None
                    valid = decoded is not None and decoded.shape == (112, 112, 3)
                    verified = (trusted and prior and prior.get('status') == 'prepared'
                                and prior.get('output') == str(target) and valid
                                and hashlib.sha256(old).hexdigest() == prior.get('output_sha256'))
                    if verified:
                        row = prior
                    else:
                        crop, details = aligner.process(image, source) if hasattr(aligner, 'process') else aligner(image)
                        if crop.shape != (112, 112, 3) or crop.dtype != np.uint8:
                            raise ValueError('Invalid prepared image shape/type')
                        verified = valid and np.array_equal(decoded, crop)
                        if not verified:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            temp = target.with_name(target.stem + '.partial.png')
                            if not cv2.imwrite(str(temp), crop):
                                raise OSError('PNG write failed')
                            with temp.open('r+b') as f:
                                os.fsync(f.fileno())
                            if target.exists():
                                backup = attempt / 'replaced' / target.parent.name / target.name
                                backup.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(target, backup)
                            os.replace(temp, target)
                        row.update(status='prepared', output=str(target), output_sha256=sha256(target), **details)
                    reused += int(bool(verified))
                    success += 1
                    targets.write(f'{target.as_posix()} {label}\n')
                except (ValueError, OSError, cv2.error) as exc:
                    row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                records.write(json.dumps(row, ensure_ascii=False) + '\n')
                processed += 1
                if processed % 1000 == 0:
                    for f in (records, targets):
                        f.flush()
                        os.fsync(f.fileno())
                if time.monotonic() - last >= 10:
                    last = time.monotonic()
                    print(f'Recovery {processed}/{total}: reused={reused}, prepared={success}, failed={processed-success}, images/s={processed/(last-start):.2f}', flush=True)
            for reader, _ in readers:
                if next(reader, None) is not None:
                    raise ValueError('Recovery manifest exceeds input coverage')
            for f in (records, targets):
                f.flush()
                os.fsync(f.fileno())
        if sha256(source_list) != contract['source_list_sha256']:
            raise ValueError('Source list changed during recovery')
        report = dict(completed=total == processed == success and total > 0, total=total,
                      processed=processed, prepared=success, failed=processed-success,
                      reused_images=reused, regenerated_images=success-reused,
                      policy=aligner.policy, execution=getattr(aligner, 'execution', {}),
                      source_list=str(source_list), source_list_sha256=contract['source_list_sha256'],
                      raw_root=str(root), storage_schema='shared-preprocessing-v1', recovery_attempt=str(attempt))
        for name, key in [('manifest.jsonl', 'manifest'), ('train_list.txt', 'train_list')]:
            dest = out / name
            if dest.exists():
                shutil.copy2(dest, attempt / ('previous-' + name))
            temp = out / (name + '.partial')
            shutil.copyfile(attempt / name, temp)
            os.replace(temp, dest)
            report[key], report[key + '_sha256'] = str(dest), sha256(dest)
        write_derivation(out, 'image_preprocessing_resume', contract, {},
                         dict(total=total, prepared=success, reused=reused),
                         'completed' if report['completed'] else 'partial_or_failed')
        if (out / 'report.json').exists():
            shutil.copy2(out / 'report.json', attempt / 'previous-report.json')
        atomic_json(out / 'report.json', report)
        if report['completed']:
            (out / 'COMPLETED').touch()
        return report


def _verify_completed(out, source_list, root, aligner):
    # Do not mutate a completed artifact. Verify under our lock using a report copy
    # without the storage publishing guard, through the ordinary row/hash verifier.
    report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    if Path(report['raw_root']).resolve() != root:
        raise ValueError('Incomplete publication or different raw root')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        copy = dict(report)
        copy.pop('storage_schema', None)
        path = Path(tmp) / 'report.json'
        atomic_json(path, copy)
        verify_alignment(path, source_list, aligner.policy)
    if not (out / 'COMPLETED').exists():
        (out / 'COMPLETED').touch()  # Recover only a fully verified publication.
    return report
