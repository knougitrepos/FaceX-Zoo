"""Wait for one explicit parent process, then evaluate its saved pilot checkpoints.

Never resumes training or chooses a final test operating point. The paired
diagnostics provide evidence for the next training decision.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'fine-tune')]


def main():
    import psutil
    from modules.parent_controls import checkpoint_health
    from modules.parent_validation import evaluate_parent, compare_parents, verify_validation
    from modules.checkpoint_manager import compute_sha256
    from modules.notebook_workflow import write_json
    from modules.shared_storage import artifact_lock
    import modules.parent_validation as validation_module
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch',required=True)
    args = parser.parse_args()
    launch_path = Path(args.launch).resolve()
    launch = json.loads(launch_path.read_text(encoding='utf-8'))
    request = json.loads(Path(launch['request']).read_text(encoding='utf-8'))
    cfg = request['settings']
    out = Path(cfg['parent_control']['output_root'])
    if cfg['runtime'].get('stop_after_epoch') != 2 or request['stage'] != 'pretrain':
        raise ValueError('This monitor only handles the planned two-epoch Parent pilot')
    validation = out/'validation/vggface2_fixed_200x10_v1'
    verify_validation(validation)
    protocol_hash = compute_sha256(str(validation/'protocol.json'))
    source_hash = compute_sha256(validation_module.__file__)
    with artifact_lock(out/'pilot_monitor'):
        state = dict(pid=launch['pid'], launch=str(launch_path), protocol_sha256=protocol_hash,
                     validation_source_sha256=source_hash, phase='waiting_for_pilot')
        write_json(out/'pilot_monitor/status.json',state)
        try:
            process = psutil.Process(launch['pid'])
            cmdline = process.cmdline()
            if str(launch['request']) not in cmdline or 'run_parent_control.py' not in ' '.join(cmdline):
                raise ValueError('PID does not belong to this exact Parent launch')
            logging.info('Watching PID %s created %s',process.pid,process.create_time())
            while True:
                try:
                    code = process.wait(timeout=30)
                    logging.info('Parent process exited: %s',code)
                    break
                except psutil.TimeoutExpired:
                    logging.info('Verified Parent PID %s remains live',process.pid)
        except psutil.NoSuchProcess:
            logging.info('Parent process is no longer present; checking its terminal result')
        result_file = Path(launch['result'])
        if not result_file.is_file():
            raise RuntimeError('Parent exited without a pilot result; inspect its traceback/recovery before resuming')
        results = json.loads(result_file.read_text(encoding='utf-8'))
        if len(results)!=1 or results[0].get('status')!='paused_epoch_limit' or results[0].get('epoch')!=2:
            raise ValueError('Expected exactly one saved two-epoch pilot, not a completed Parent')
        if compute_sha256(results[0]['checkpoint']) != results[0]['sha256']:
            raise ValueError('Pilot checkpoint changed after saving')
        if (compute_sha256(str(validation/'protocol.json'))!=protocol_hash
                or compute_sha256(validation_module.__file__)!=source_hash):
            raise ValueError('Fixed diagnostic protocol/implementation changed during training')
        state['phase']='evaluating_fixed_diagnostic'
        write_json(out/'pilot_monitor/status.json',state)
        reference = Path(cfg['parent_control']['reference']['path']).parent
        control = Path(results[0]['checkpoint']).parent
        rows=[]
        for epoch in (1,2):
            filename=f'pretrain_mobilefacenet_1024d_epoch_{epoch}.pt'
            a,b = reference/filename,control/filename
            old=evaluate_parent(a,validation,out/'validation/evaluations',keep_raw_results=cfg['project']['keep_raw_results'])
            new=evaluate_parent(b,validation,out/'validation/evaluations',keep_raw_results=cfg['project']['keep_raw_results'])
            paired=out/'validation/comparisons'/f'epoch_{epoch}_{new.parent.name}.json'
            compare_parents(old,new,paired)
            rows.append(dict(epoch=epoch,reference_health=checkpoint_health(a),control_health=checkpoint_health(b),
                             paired_comparison=str(paired),comparison=json.loads(paired.read_text(encoding='utf-8'))))
        report=dict(status='pilot_evaluated',parent_complete=False,launch=str(launch_path),
                    protocol_sha256=protocol_hash,rows=rows,training_decision='pending_evidence_review',
                    note='No continuation launched. This diagnostic does not establish UCFace reproduction or final SurvFace improvement.')
        write_json(out/'pilot_report.json',report,immutable=True)
        state.update(phase='pilot_evaluated',report=str(out/'pilot_report.json'))
        write_json(out/'pilot_monitor/status.json',state)
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
