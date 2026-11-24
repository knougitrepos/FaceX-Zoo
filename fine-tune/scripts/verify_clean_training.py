"""Bounded real-data smoke. It cannot publish a parent or fine-tune checkpoint."""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.clean_benchmarks import monitor, prepare_panels
from modules.clean_training import (
    files_for_protocol,
    input_list,
    make_model,
    optimizer_for,
    recipe,
    train_epoch,
)
from modules.dataset import build_dataloader
from modules.notebook_workflow import write_json
from modules.trainer import set_seed


def sample_loader(root, temp, stage, r):
    source = input_list(root, r)
    wanted = set(random.Random(8972).sample(range(r["images"]), 2 * r["batch"]))
    with source.open(encoding="utf-8") as stream:
        records = [line for i, line in enumerate(stream) if i in wanted]
    file = temp / (stage + ".txt")
    file.write_text("".join(records), encoding="utf-8")
    return build_dataloader(
        str(root),
        str(file),
        batch_size=r["batch"],
        num_workers=4,
        cache_dir=str(temp / "index"),
        prefetch_factor=4,
        augmentation=r["augmentation"],
        seed=8972,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    panels, prep = prepare_panels(args.data)
    set_seed(8972)
    report = dict(
        scope="random initialization, bounded GPU smoke only",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        preparation=prep,
        implementation=files_for_protocol(),
    )
    with tempfile.TemporaryDirectory() as temp:
        r = recipe("parent")
        loader = sample_loader(args.data, Path(temp), "parent", r)
        model = make_model("parent", r)
        opt = optimizer_for(model, "parent", r)
        report["parent_loss"] = train_epoch(model, opt, "parent", r, loader, 1)
        report["parent_last_lr"] = opt.param_groups[0]["lr"]
        opt.zero_grad(set_to_none=True)
        report["parent_monitor"] = monitor(model.backbone, panels)
        parent_state = {k: v.cpu() for k, v in model.backbone.state_dict().items()}
        del model, opt, loader
        torch.cuda.empty_cache()
        r = recipe("finetune")
        loader = sample_loader(args.data, Path(temp), "finetune", r)
        model = make_model("finetune", r, parent_state)
        opt = optimizer_for(model, "finetune", r)
        report["phase_losses"] = [train_epoch(model, opt, "finetune", r, loader, epoch) for epoch in (1, 4)]
        del loader
    write_json(args.output, report, immutable=True)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                prepare_s=prep["prepare_seconds"],
                monitor_s=report["parent_monitor"]["seconds"],
                phase_losses=report["phase_losses"],
            )
        )
    )


if __name__ == "__main__":
    main()
