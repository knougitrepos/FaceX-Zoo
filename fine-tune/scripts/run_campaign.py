"""Spawn-safe entry point for notebook training/export and bounded throughput trials."""
import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "fine-tune")]


def benchmark(cfg, workers, steps):
    import torch
    import cv2
    from modules.notebook_workflow import make_loaders, _formal_device
    from modules.model_builder import build_model
    from modules.trainer import set_seed
    results = []
    for count in workers:
        current = copy.deepcopy(cfg)
        current["training"]["num_workers"] = count
        torch.set_num_threads(current["training"].get("torch_threads", 2))
        cv2.setNumThreads(1)
        set_seed(current["project"]["seed"])
        device = _formal_device(current)
        model = build_model(current["backbone"], dict(current["heads"]["arcface"], num_class=current["dataset"]["num_classes"])).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=current["training"]["lr"], weight_decay=current["training"]["weight_decay"])
        loader = val = iterator = None
        try:
            loader, val = make_loaders(current)
            iterator = iter(loader)
            measured_images = 0
            started = None
            torch.cuda.reset_peak_memory_stats(device)
            for step in range(steps + 10):
                stop = current.get("runtime", {}).get("stop_file")
                if stop and Path(stop).exists():
                    raise RuntimeError("Benchmark stop requested; no formal weights were produced")
                images, labels = next(iterator)
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits, aux = model(images, labels)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if aux is not None:
                    loss = loss + aux.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step == 9:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                elif step >= 10:
                    measured_images += len(images)
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            row = {"workers": count, "steps": steps, "images_per_second": measured_images / duration,
                   "seconds": duration, "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                   "dataset": current["dataset"]["name"], "classes": current["dataset"]["num_classes"],
                   "dimension": current["backbone"]["feat_dim"], "batch_size": current["training"]["batch_size"],
                   "is_formal_training": False, "head": "arcface", "scope": "bounded throughput only; no benchmark accuracy"}
            print(json.dumps(row), flush=True)
            results.append(row)
        finally:
            for item in (loader, val):
                worker_iter = getattr(item, "_iterator", None)
                if worker_iter is not None:
                    worker_iter._shutdown_workers()
            del model, optimizer, iterator, loader, val
            torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    cfg = request["settings"]
    runtime = cfg["runtime"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(runtime["log_file"], encoding="utf-8")], force=True)
    from modules.notebook_workflow import run_training_matrix, export_matrix, validate_settings
    validate_settings(cfg)
    lock = Path(cfg["paths"]["checkpoint_dir"]) / ".campaign.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Explicit lock removal is required after a hard-killed process; never assume
    # an old PID is dead or erase another run's lock.
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Campaign lock exists: {lock}. Check recorded PID before removing a stale lock.")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        print(f"Device and runtime checks precede execution; PID={os.getpid()}", flush=True)
        if request["stage"] in ("finetune", "export") and not request.get("smoke"):
            from modules.finetune_protocol import prepare_selected_protocols
            prepare_selected_protocols(cfg)
        if request["stage"] == "benchmark":
            result = benchmark(cfg, request.get("benchmark_workers") or [0, 2, 4], request.get("benchmark_steps", 100))
        elif request["stage"] == "export":
            result = export_matrix(cfg)
        else:
            result = run_training_matrix(cfg, request["stage"], request["resume"], request["smoke"])
        Path(runtime["result_file"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(encoding="utf-8")
    main()
