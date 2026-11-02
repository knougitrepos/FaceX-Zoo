"""Notebook controls; training and spawn workers live in a separate Python process."""
import copy
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path


def launch_alignment(base_dir, cfg, dataset, max_images=32, output_dir=None, device="cpu", resume=False):
    """Run dataset-specific CPU preprocessing with the notebook's exact settings."""
    from .notebook_workflow import save_settings
    root = Path(base_dir)
    ticket = Path(cfg["paths"].get("launch_dir", root / "fine-tune/data/adjust1/alignment_launches")) / uuid.uuid4().hex
    ticket.mkdir(parents=True)
    request = ticket / "settings.json"
    save_settings(request, cfg)
    if resume and max_images is not None:
        raise ValueError("Resume requires max_images=None")
    if max_images is not None and output_dir is None:
        ds = cfg["dataset"] if dataset == "vggface2" else cfg["finetune_profiles"][dataset]
        output_dir = Path(ds["alignment_report"]).parent.parent / "audits" / ticket.name
    cmd = [sys.executable, "-u", str(root / "fine-tune/scripts/prepare_alignment.py"),
           "--settings", str(request), "--dataset", dataset, "--device", device]
    if max_images is not None:
        cmd += ["--max-images", str(max_images)]
    if output_dir is not None:
        cmd += ["--output-dir", str(output_dir)]
    if resume:
        cmd += ["--resume"]
    print(f"Alignment settings: {request}; bounded limit: {max_images}", flush=True)
    _stream_alignment(cmd, root, ticket / "console.log")


def _stream_alignment(cmd, root, log_path):
    """Explicit pipes work with Jupyter/CREATE_NO_WINDOW and retain child tracebacks."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with Path(log_path).open("w", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print(f"Preprocessing PID {proc.pid}; log: {log_path}", flush=True)
        try:
            with proc.stdout:
                for line in proc.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
            code = proc.wait()
        except KeyboardInterrupt:
            # Do not leave a writer alive after losing its pipe/log owner.
            proc.terminate()
            proc.wait()
            print(f"Preprocessing interrupted; preserved outputs can resume. Log: {log_path}", flush=True)
            raise
    if code:
        raise RuntimeError(f"Preprocessing exited with code {code}. Full child traceback: {log_path}")


def launch_campaign(base_dir, cfg, stage, resume_checkpoints=None, smoke=False,
                    benchmark_workers=None, benchmark_steps=100):
    root = Path(base_dir)
    ticket = Path(cfg["paths"].get("launch_dir", root / "fine-tune/data/launches")) / uuid.uuid4().hex
    ticket.mkdir(parents=True)
    effective = copy.deepcopy(cfg)
    effective["runtime"] = {"stop_file": str(ticket / "stop.request"),
                            "log_file": str(ticket / "console.log"),
                            "result_file": str(ticket / "result.json")}
    request = {"settings": effective, "stage": stage, "resume": resume_checkpoints or {}, "smoke": smoke,
               "benchmark_workers": benchmark_workers, "benchmark_steps": benchmark_steps}
    request_path = ticket / "request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [sys.executable, "-u", str(root / "fine-tune/scripts/run_campaign.py"), "--request", str(request_path)]
    proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(f"Child PID {proc.pid}; log: {effective['runtime']['log_file']}; stop request: {effective['runtime']['stop_file']}", flush=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        Path(effective["runtime"]["stop_file"]).touch()
        print("Requested safe stop. Saving optimizer/RNG/next-batch state at a completed step.", flush=True)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"Child {proc.pid} still finishing; inspect log/progress.json. Stop flag remains set; no forced kill.", flush=True)
            # Drain without notebook output so the child cannot block on a full
            # pipe or receive BrokenPipe while writing its recovery checkpoint.
            def drain():
                for _ in proc.stdout:
                    pass
                proc.wait()
                proc.stdout.close()
            threading.Thread(target=drain, daemon=True).start()
        raise
    finally:
        if proc.poll() is not None:
            proc.stdout.close()
    if code:
        raise RuntimeError(f"Child exited with {code}; inspect {effective['runtime']['log_file']} and run progress.json")
    return json.loads(Path(effective["runtime"]["result_file"]).read_text(encoding="utf-8"))
