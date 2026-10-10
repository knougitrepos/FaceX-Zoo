"""Notebook controls; training and spawn workers live in a separate Python process."""
import copy
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path


def launch_campaign(base_dir, cfg, stage, resume_checkpoints=None, smoke=False,
                    benchmark_workers=None, benchmark_steps=100):
    root = Path(base_dir)
    ticket = root / "fine-tune/data/launches" / uuid.uuid4().hex
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
