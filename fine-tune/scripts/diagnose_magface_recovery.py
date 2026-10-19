"""Bounded, in-memory replay; never saves production training state.

Run with a launch request and an epoch checkpoint. The unmodified main-branch
MagFace head is evaluated on the very same features at the failing batch.
The only output is a new diagnostic JSON report; existing files are preserved.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fine-tune"))

import cv2
import torch
import torch.nn.functional as F

from modules.checkpoint_manager import CheckpointManager, load_checkpoint
from modules.model_builder import build_model
from modules.notebook_workflow import make_loaders, merge_settings
from modules.trainer import Trainer, TRAINING_NUMERICS, set_seed


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-batch", type=int, default=271)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.target_batch <= args.steps <= 1000:
        raise ValueError("Require 1 <= target batch <= steps <= 1000")
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(report_path)
    checkpoint = Path(args.checkpoint).resolve()
    if (checkpoint.parent.parent / ".campaign.lock").exists():
        raise RuntimeError("Production campaign lock exists; do not run diagnostic concurrently")
    protected = [checkpoint, checkpoint.parent / "progress.json",
                 ROOT / "fine-tune/notebooks/03_finetune.ipynb"]
    before = {str(p): digest(p) for p in protected if p.exists()}
    payload = load_checkpoint(str(checkpoint))
    if payload.get("head") != "magface" or payload.get("batch_idx", 0):
        raise ValueError("Expected an end-of-epoch MagFace checkpoint")
    cfg = copy.deepcopy(json.loads(Path(args.request).read_text(encoding="utf-8"))["settings"])
    cfg["project"]["seed"] = int(payload["seed"])
    dataset_name = payload["config"]["dataset"]["name"]
    cfg["dataset"] = merge_settings(cfg["dataset"], cfg["finetune_profiles"][dataset_name])
    torch.set_num_threads(cfg["training"].get("torch_threads", 2))
    cv2.setNumThreads(1)
    set_seed(payload["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("This replay requires the original CUDA execution device")
    device = torch.device("cuda:0")
    head_cfg = dict(cfg["heads"]["magface"], num_class=cfg["dataset"]["num_classes"])
    model = build_model(cfg["backbone"], head_cfg).to(device)
    # Read the original implementation from Git, without editing the worktree.
    original_source = subprocess.run(
        ["git", "-C", str(ROOT), "show", "main:head/MagFace.py"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    namespace = {}
    exec(compile(original_source, "main:head/MagFace.py", "exec"), namespace)
    reference = namespace["MagFace"](
        feat_dim=cfg["backbone"]["feat_dim"], num_class=cfg["dataset"]["num_classes"],
        **{k: head_cfg[k] for k in ["scale", "l_a", "u_a", "l_margin", "u_margin", "lamda"]},
        margin_am=head_cfg.get("margin_am", 0.),
    ).to(device)
    train_loader, val_loader = make_loaders(cfg)
    manager = CheckpointManager(str(checkpoint.parent))
    trainer = Trainer(model, train_loader, val_loader, cfg, manager, device,
                      parent_hash=payload["parent_hash"], head_type="magface", stage="finetune")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Diagnostic must not write production state")

    manager.save_checkpoint = forbidden
    trainer.save_state_checkpoint = forbidden
    trainer._progress = forbidden
    epoch = trainer.resume_from_checkpoint(str(checkpoint))  # strict config + full RNG/state restore
    old_policy = payload["extra_meta"].get("training_numerics")
    del payload
    model.train()
    train_loader.batch_sampler.set_epoch(epoch, 0)
    features = []

    def capture(_module, inputs):
        features[:] = [inputs[0].detach()]

    hook = model.head.register_forward_pre_hook(capture)
    evidence = None
    losses = {}
    try:
        for index, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()
            trainer.optimizer.zero_grad(set_to_none=True)
            logits, regularizer = model(images, labels)
            loss = trainer.criterion(logits, labels) + regularizer.mean()
            if index == args.target_batch:
                reference.load_state_dict(model.head.state_dict(), strict=True)
                x = features[0].clone().requires_grad_()
                old_logits, old_g = reference(x, labels)
                old_loss = F.cross_entropy(old_logits, labels) + old_g.mean()
                old_loss.backward()
                with torch.no_grad():
                    raw_cos = F.normalize(features[0]) @ F.normalize(model.head.weight, dim=0)
                    cos = raw_cos.clamp(-1, 1)
                    coords = torch.nonzero(1 - cos.pow(2) == 0).tolist()
                evidence = {
                    "batch": index, "original_loss": old_loss.item(), "patched_loss": loss.item(),
                    "original_logits_equal_patched": torch.equal(old_logits, logits),
                    "original_feature_gradient_finite": bool(torch.isfinite(x.grad).all()),
                    "original_head_gradient_finite": bool(torch.isfinite(reference.weight.grad).all()),
                    "boundary_entries": [dict(sample=i, class_id=j, label=int(labels[i]),
                                              raw_cos=float(raw_cos[i, j]), is_target=j == int(labels[i]))
                                         for i, j in coords],
                }
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite patched loss at batch {index}")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not bool(torch.isfinite(torch.nn.utils.get_total_norm(grads, norm_type=float("inf")))):
                raise RuntimeError(f"Non-finite patched gradient at batch {index}")
            if index == args.target_batch:
                evidence["patched_full_model_gradient_finite"] = True
            trainer.optimizer.step()  # memory only, never exported as a training checkpoint
            if index == 1 or index % 50 == 0 or index == args.target_batch:
                losses[str(index)] = loss.item()
                print(f"Diagnostic epoch {epoch} batch {index}/{args.steps}: finite", flush=True)
            if index >= args.steps:
                break
        if index != args.steps or evidence is None:
            raise RuntimeError("Loader exhausted before the diagnostic bound")
        if evidence["original_feature_gradient_finite"] and evidence["original_head_gradient_finite"]:
            raise RuntimeError("Original failure was not reproduced; do not claim a successful fix")
    finally:
        hook.remove()
        iterator = getattr(train_loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    after = {p: digest(p) for p in before}
    if before != after:
        raise RuntimeError("Protected files changed during diagnosis")
    report = dict(status="passed_bounded_replay", production_training_completed=False,
                  checkpoint=str(checkpoint), checkpoint_sha256=before[str(checkpoint)],
                  epoch=epoch, optimizer_steps_in_memory=args.steps,
                  source_numerics=old_policy, applied_numerics=TRAINING_NUMERICS,
                  torch_version=torch.__version__, device=torch.cuda.get_device_name(device),
                  original_head_source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                  evidence=evidence, sampled_batch_losses=losses, protected_hashes_unchanged=before)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"Diagnostic report saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
