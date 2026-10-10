"""Bounded actual-Trainer throughput trials; never save formal trained weights."""
import argparse
import copy
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'fine-tune')]


class LimitedLoader:
    def __init__(self, loader, warmup, steps):
        self.loader = loader
        self.dataset, self.batch_sampler = loader.dataset, loader.batch_sampler
        self.warmup, self.steps = warmup, steps
        self.started = None
        self.wait_seconds = 0.0

    def __len__(self):
        return self.warmup + self.steps

    def __iter__(self):
        import torch
        iterator = iter(self.loader)
        for i in range(self.warmup + self.steps):
            if i == self.warmup:
                torch.cuda.synchronize()
                self.started = time.perf_counter()
            started = time.perf_counter()
            batch = next(iterator)
            if i >= self.warmup:
                self.wait_seconds += time.perf_counter() - started
            yield batch


class Utilization:
    def __init__(self):
        self.rows = []
        self.process = subprocess.Popen(
            ['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits','--loop-ms=500'], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        import psutil
        for line in self.process.stdout:
            try:
                gpu, mem, power, temp = [float(x.strip()) for x in line.split(',')]
                self.rows.append(dict(time=time.perf_counter(), gpu=gpu, vram_mb=mem,
                                      power_w=power, temperature=temp,
                                      cpu=psutil.cpu_percent(), ram_gb=psutil.virtual_memory().used/2**30))
            except ValueError:
                pass

    def finish(self, since):
        self.process.terminate()
        self.process.wait(timeout=10)
        self.thread.join(timeout=2)
        self.process.stdout.close()
        rows = [row for row in self.rows if row['time']>=since]
        return {key: statistics.mean(row[key] for row in rows) for key in
                ('gpu','vram_mb','power_w','temperature','cpu','ram_gb')} if rows else {}


def main():
    import torch
    import cv2
    from modules.notebook_workflow import resolve_settings, make_loaders, validate_paper_parameters
    from modules.model_builder import build_model
    from modules.trainer import Trainer, set_seed
    from modules.checkpoint_manager import CheckpointManager, compute_sha256

    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    output = Path(request['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    if (ROOT/'fine-tune/checkpoints/shared_parent_v2/.campaign.lock').exists():
        raise RuntimeError('Formal campaign is active or stale lock requires review; do not contend for GPU')
    logging.basicConfig(level=logging.WARNING)
    assert torch.cuda.is_available(), 'CUDA required'
    rows = []
    for index, variant in enumerate(request['variants']):
        cfg = resolve_settings(ROOT, overrides=request['settings'])
        cfg['training'].update(variant)
        cfg['training']['recovery_minutes'] = 0
        cfg['runtime'] = {}
        validate_paper_parameters(cfg)
        torch.set_num_threads(cfg['training']['torch_threads'])
        cv2.setNumThreads(1)
        set_seed(cfg['project']['seed'])
        model = build_model(cfg['backbone'],dict(cfg['heads']['arcface'],num_class=cfg['dataset']['num_classes']))
        train = val = monitor = trainer = bounded = None
        print('PROFILE START '+json.dumps(variant), flush=True)
        try:
            train, val = make_loaders(cfg)
            bounded = LimitedLoader(train, request.get('warmup',50),request.get('steps',300))
            trainer = Trainer(model,bounded,val,cfg,CheckpointManager(str(output/f'trial_{index}')),
                              torch.device('cuda:0'),head_type='arcface',stage='pretrain')
            if request.get('resume_checkpoint'):
                trainer.resume_from_checkpoint(request['resume_checkpoint'])
            if 'profile_start_batch' in variant:
                # Diagnostic sample window only; never persist this as a resume.
                trainer.resume_batch = int(variant['profile_start_batch'])
                trainer.resume_meter = None
            start_batch = trainer.resume_batch
            torch.cuda.reset_peak_memory_stats()
            monitor = Utilization()
            trainer.train_one_epoch(trainer.cur_epoch)
            torch.cuda.synchronize()
            duration = time.perf_counter()-bounded.started
            telemetry = monitor.finish(bounded.started)
            monitor = None
            weights = hashlib.sha256()
            for name, tensor in sorted(trainer.model.state_dict().items()):
                weights.update(name.encode())
                weights.update(tensor.detach().cpu().numpy().tobytes())
            row = dict(variant=variant, measured_steps=bounded.steps, warmup=bounded.warmup,
                       images_per_second=bounded.steps*cfg['training']['batch_size']/duration,
                       elapsed_seconds=duration, host_loader_wait_seconds=bounded.wait_seconds,
                       peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20, telemetry=telemetry,
                       weights_sha256=weights.hexdigest(),
                       cumulative_loss_sum=trainer._finished_train_meter.sum,
                       cumulative_loss_count=trainer._finished_train_meter.count,
                       config_hash=trainer.config_hash, completed=True, formal_training=False)
            row['start_batch'] = start_batch
            rows.append(row)
            print('PROFILE RESULT '+json.dumps(row),flush=True)
        finally:
            if monitor is not None:
                monitor.finish(0)
            for loader in (train,val):
                iterator = getattr(loader,'_iterator',None)
                if iterator is not None:
                    iterator._shutdown_workers()
            del bounded,trainer,model,train,val
            gc.collect()
            torch.cuda.empty_cache()
            (output/'results.json').write_text(json.dumps(dict(
                rows=rows,request=str(request_path),request_sha256=compute_sha256(str(request_path)),
                resume_sha256=compute_sha256(request['resume_checkpoint']) if request.get('resume_checkpoint') else None,
                device=torch.cuda.get_device_name(), is_formal_training=False,
                scope='actual Trainer and recovery weights; optional diagnostic sample windows; short trials, not full epoch ETA'),
                indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
