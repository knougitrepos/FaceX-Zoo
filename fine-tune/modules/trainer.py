"""
Trainer Implementation for Face Recognition with ArcFace, AdaFace, and MagFace.
Follows UCFace (IEEE TIFS 2024) optimization settings: Adam optimizer, lr=1e-3, 40 epochs.
"""

import os
import time
import logging
import random
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from modules.checkpoint_manager import (
        CheckpointManager,
        compute_config_hash,
        extract_train_config,
        extract_model_and_train_config,
    )
except ImportError:
    try:
        from checkpoint_manager import (
            CheckpointManager,
            compute_config_hash,
            extract_train_config,
            extract_model_and_train_config,
        )
    except ImportError:
        from .checkpoint_manager import (
            CheckpointManager,
            compute_config_hash,
            extract_train_config,
            extract_model_and_train_config,
        )

logger = logging.getLogger("fine_tune.trainer")


class TrainingStopped(RuntimeError):
    """A requested stop after an optimizer step, with resumable state saved."""


def set_seed(seed: int = 8972):
    """Set random seed for reproducibility across random, numpy, torch, cuda (goal.md default: 8972)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Reproducibility seed set to: {seed}")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


class DeferredLossMeter(AverageMeter):
    """Batch scalar transfers, retaining the original ordered Python summation."""

    def __init__(self, sync_batches=20):
        super().__init__()
        if type(sync_batches) is not int or sync_batches < 1:
            raise ValueError("LOSS_SYNC_BATCHES must be a positive integer")
        self.sync_batches = sync_batches
        self.pending = []

    def update_tensor(self, loss, n):
        self.pending.append((loss.detach(), n))
        if len(self.pending) >= self.sync_batches:
            self.flush()

    def flush(self):
        if self.pending:
            values = torch.stack([value for value, _ in self.pending]).cpu().tolist()
            for value, (_, count) in zip(values, self.pending):
                self.update(value, count)
            self.pending.clear()


class Trainer:
    """Trainer supporting ArcFace, AdaFace, and MagFace heads on MobileFaceNet."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        checkpoint_manager: CheckpointManager,
        device: torch.device,
        parent_hash: Optional[str] = None,
        head_type: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.parent_hash = parent_hash
        self.history: Dict[str, Any] = {"epochs": [], "train_loss": [], "lr": []}
        self.cur_epoch = 1
        self.epochs_trained_in_run = 0

        # Resolve head_type and stage
        if head_type is not None:
            self.head_type = str(head_type)
        elif hasattr(model, "head_type"):
            self.head_type = str(model.head_type)
        elif hasattr(model, "head"):
            self.head_type = model.head.__class__.__name__
        else:
            self.head_type = "unknown"

        if stage is not None:
            self.stage = str(stage)
        else:
            self.stage = str(config.get("stage", config.get("project", {}).get("stage", "finetune")))

        train_cfg = config.get("training", {})
        opt_name = train_cfg.get("optimizer", "adam").lower()
        lr = float(train_cfg.get("lr", 0.001))
        weight_decay = float(train_cfg.get("weight_decay", 0.0001))

        # Select optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = float(train_cfg.get("momentum", 0.9))
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            logger.error(f"Unsupported optimizer: '{opt_name}'. Allowed: 'adam', 'sgd'")
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # Scheduler: StepLR / MultiStepLR (UCFace: 10 epoch마다 0.1 감소, 총 40 epoch)
        milestones = train_cfg.get("milestones", [10, 20, 30])
        gamma = float(train_cfg.get("lr_decay_ratio", 0.1))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=gamma
        )

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.total_epochs = int(train_cfg.get("epoches", 40))
        self.print_freq = int(train_cfg.get("print_freq", 20))
        self.loss_sync_batches = train_cfg.get("loss_sync_batches", 20)
        self.save_freq = int(train_cfg.get("save_freq", 5))
        self.keep_raw_results = bool(config.get("project", {}).get("keep_raw_results", True))
        self.seed = int(config.get("project", {}).get("seed", 8972))
        self.last_periodic_ckpt: Optional[str] = None
        self.saved_checkpoints: Dict[str, str] = {}
        self.last_saved_ckpt: Optional[str] = None
        self.resume_batch = 0
        self.resume_meter = None
        self._recovery_clock = time.monotonic()
        self._owned_recovery = None
        self._model_prefix = "model"
        self.progress_path = Path(self.checkpoint_manager.checkpoint_dir) / "progress.json"

        # Build full training config and canonical config hash
        self.train_config = extract_model_and_train_config(
            model=self.model,
            config=self.config,
            seed=self.seed,
            stage=self.stage,
            head_type=self.head_type,
            parent_hash=self.parent_hash,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            scheduler=self.scheduler,
            optimizer=self.optimizer,
        )
        self.config_hash = compute_config_hash(self.train_config)
        self.train_config["config_hash"] = self.config_hash

    def _progress(self, phase, **values):
        payload = {"phase": phase, "seed": self.seed, "head": self.head_type, "stage": self.stage,
                   "epoch": self.cur_epoch, "total_epochs": self.total_epochs, "time_unix": time.time(),
                   "last_checkpoint": self.last_saved_ckpt, **values}
        temp = self.progress_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.progress_path)

    def _recovery(self, next_batch, meter):
        import copy
        filename = f"{self._model_prefix}_recovery_e{self.cur_epoch}_b{next_batch}_{uuid.uuid4().hex[:8]}.pt"
        path = self.checkpoint_manager.save_checkpoint(
            filename, self.model, self.optimizer, self.scheduler, epoch=self.cur_epoch - 1,
            batch_idx=next_batch, extra_meta={"completed": False, "is_smoke": False,
                "seed": self.seed, "stage": self.stage, "head": self.head_type,
                "parent_hash": self.parent_hash, "config": self.train_config,
                "history": copy.deepcopy(self.history), "total_epochs": self.total_epochs,
                "execution_settings": self.config,
                "epoch_state": {"epoch": self.cur_epoch, "next_batch": next_batch,
                                "loss_sum": meter.sum, "loss_count": meter.count}})
        old = self._owned_recovery
        self._owned_recovery = self.last_saved_ckpt = path
        # Recovery is a rolling resume cache, distinct from KEEP_RAW_RESULTS.
        # Only a prior cache created by this process can be removed.
        if old and old != path:
            Path(old).unlink(missing_ok=True)
        self._recovery_clock = time.monotonic()
        logger.info("Recovery checkpoint: %s", path)

    def train_one_epoch(self, epoch: int) -> float:
        """Run training for one epoch."""
        self.model.train()
        loss_meter = DeferredLossMeter(self.loss_sync_batches)
        offset = self.resume_batch
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch, offset)
        elif offset:
            raise ValueError("Mid-epoch resume requires EpochBatchSampler")
        if self.resume_meter:
            loss_meter.sum, loss_meter.count = self.resume_meter
            loss_meter.avg = loss_meter.sum / max(loss_meter.count, 1)
        self.resume_batch, self.resume_meter = 0, None
        total = getattr(sampler, "total_batches", len(self.train_loader))
        start_time = time.time()
        self._progress("training", batches_completed=offset, total_batches=total)

        for batch_idx, (images, labels) in enumerate(self.train_loader, start=offset):
            try:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                # Forward pass
                logits, loss_g = self.model(images, labels)

                # MagFace includes loss_g regularization term
                if loss_g is not None:
                    loss = self.criterion(logits, labels) + torch.mean(loss_g)
                else:
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_meter.update_tensor(loss, images.size(0))
            except Exception as e:
                logger.error(f"Training step error [epoch {epoch}, batch {batch_idx}]: {e}", exc_info=True)
                raise

            done = batch_idx + 1
            stop_path = self.config.get("runtime", {}).get("stop_file")
            stopping = bool(stop_path and Path(stop_path).exists())
            recovery_seconds = float(self.config["training"].get("recovery_minutes", 10)) * 60
            if stopping or (recovery_seconds > 0 and time.monotonic() - self._recovery_clock >= recovery_seconds):
                loss_meter.flush()  # Recovery must include every completed optimizer step.
                self._recovery(done, loss_meter)
            if stopping:
                self._progress("stopped", batches_completed=done, total_batches=total, completed=False)
                raise TrainingStopped(f"Safe stop; resume explicitly from {self.last_saved_ckpt}")
            if batch_idx % self.print_freq == 0 or done == total:
                loss_meter.flush()
                current_lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                rate = (done - offset) * self.train_loader.batch_sampler.batch_size / max(elapsed, 1e-9)
                eta = (total - done) * self.train_loader.batch_sampler.batch_size / max(rate, 1e-9)
                vram_mb = 0.0
                if torch.cuda.is_available() and getattr(self.device, "type", "") == "cuda":
                    vram_mb = torch.cuda.memory_allocated(self.device) / (1024**2)

                logger.info(
                    f"Epoch [{epoch}/{self.total_epochs}] "
                    f"Step [{done}/{total}] {100 * done / total:.2f}% "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"LR: {current_lr:.6f} "
                    f"VRAM: {vram_mb:.1f}MB "
                    f"Time: {elapsed:.1f}s Images/s: {rate:.1f} ETA(epoch train): {eta / 60:.1f}min"
                )
                self._progress("training", batches_completed=done, total_batches=total,
                               epoch_percent=100 * done / total, loss=loss_meter.avg,
                               images_per_second=rate, epoch_train_eta_seconds=eta)

        loss_meter.flush()
        self._finished_train_meter = loss_meter
        return loss_meter.avg

    def validate(self) -> float:
        """Run validation evaluation if val_loader is available."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return 0.0

        self.model.eval()
        self._progress("validation", batches_completed=0, total_batches=len(self.val_loader))
        val_loss_meter = DeferredLossMeter(self.loss_sync_batches)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                stop = self.config.get("runtime", {}).get("stop_file")
                if stop and Path(stop).exists():
                    self._recovery(self.train_loader.batch_sampler.total_batches, self._finished_train_meter)
                    self._progress("stopped", completed=False)
                    raise TrainingStopped(f"Stopped during validation; resume from {self.last_saved_ckpt}")
                try:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    logits, loss_g = self.model(images, labels)
                    if loss_g is not None:
                        loss = self.criterion(logits, labels) + torch.mean(loss_g)
                    else:
                        loss = self.criterion(logits, labels)
                    val_loss_meter.update_tensor(loss, images.size(0))
                    if batch_idx % self.print_freq == 0:
                        val_loss_meter.flush()
                        logger.info("Validation epoch %s step %s/%s loss %.4f", self.cur_epoch, batch_idx + 1, len(self.val_loader), val_loss_meter.avg)
                        self._progress("validation", batches_completed=batch_idx + 1, total_batches=len(self.val_loader))
                except Exception as e:
                    logger.error(f"Validation step error [batch {batch_idx}]: {e}", exc_info=True)
                    raise

        val_loss_meter.flush()
        logger.info(f"Validation Loss: {val_loss_meter.avg:.4f}")
        return val_loss_meter.avg

    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        expected_head: Optional[str] = None,
        expected_stage: Optional[str] = None,
        strict_config: bool = True,
        allow_unverified_resume: bool = False,
    ) -> int:
        """Validate origin & settings, restore model, optimizer, scheduler, RNG, and prior history."""
        target_head = expected_head or self.head_type
        target_stage = expected_stage or self.stage
        start_epoch, meta = self.checkpoint_manager.resume_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_seed=self.seed,
            expected_head=target_head,
            expected_parent_hash=self.parent_hash,
            expected_stage=target_stage,
            expected_config=self.train_config,
            return_meta=True,
            strict_config=strict_config,
            allow_unverified_resume=allow_unverified_resume,
        )
        # Restore accumulated history if available
        loaded_history = meta.get("history") if meta else None
        if loaded_history and isinstance(loaded_history, dict) and "epochs" in loaded_history and len(loaded_history["epochs"]) > 0:
            import copy
            self.history = copy.deepcopy(loaded_history)
            logger.info(f"Restored prior training history for epochs: {self.history.get('epochs', [])}")
        else:
            # Mark missing prior epoch history explicitly
            prior_epochs = list(range(1, start_epoch))
            self.history = {
                "epochs": [],
                "train_loss": [],
                "lr": [],
            }
            if prior_epochs:
                self.history["missing_epochs"] = prior_epochs
                logger.warning(
                    f"Resume checkpoint does not contain prior history logs. "
                    f"Marking epochs {prior_epochs} as unrecorded in training history."
                )
        if self.val_loader is not None and len(self.val_loader) > 0:
            self.history.setdefault("val_loss", [])
        self.cur_epoch = min(start_epoch, self.total_epochs)
        state = meta.get("epoch_state") if meta else None
        if state:
            if int(state["epoch"]) != start_epoch:
                raise ValueError("Recovery epoch metadata is inconsistent")
            self.resume_batch = int(state["next_batch"])
            self.resume_meter = (float(state["loss_sum"]), int(state["loss_count"]))
            # A stop during validation can resume with the training epoch already
            # consumed. The optimizer steps happened before the saved boundary.
            if self.resume_batch > 0 and self.optimizer.state:
                self.optimizer._opt_called = True
        return start_epoch

    def save_state_checkpoint(
        self,
        filename: str,
        epoch: Optional[int] = None,
        loss: float = 0.0,
        extra_meta: Optional[Dict[str, Any]] = None,
        allow_overwrite: bool = False,
    ) -> str:
        """Save a checkpoint using the live optimizer, scheduler, and training metadata."""
        import copy
        target_epoch = epoch if epoch is not None else self.cur_epoch
        if self.epochs_trained_in_run == 0 and target_epoch > self.total_epochs:
            target_epoch = self.total_epochs

        save_path = os.path.join(self.checkpoint_manager.checkpoint_dir, filename)

        # Handle duplicate save: if this exact filename was already saved during this active training run
        # at the same epoch, avoid duplicate FileExistsError collision by returning the saved path
        if filename in self.saved_checkpoints and os.path.exists(save_path) and not allow_overwrite:
            logger.info(
                f"[Save Coordinated] Checkpoint '{filename}' already saved by active training loop. "
                f"Reusing existing verified checkpoint: {save_path}"
            )
            return save_path

        # Check actual completion: target_epoch must reach total_epochs and history must confirm actual training across all scheduled epochs
        epochs_in_history = set(self.history.get("epochs", []))
        missing_epochs = set(self.history.get("missing_epochs", []))
        all_covered = epochs_in_history.union(missing_epochs)
        is_completed = (target_epoch >= self.total_epochs) and (len(all_covered) >= self.total_epochs)

        meta = {
            "loss": loss,
            "lr": self.optimizer.param_groups[0]["lr"],
            "parent_hash": self.parent_hash,
            "history": copy.deepcopy(self.history),
            "total_epochs": self.total_epochs,
            "epoch": target_epoch,
            "seed": self.seed,
            "stage": self.stage,
            "head": self.head_type,
            "is_smoke": False,
            "completed": is_completed,
            "config": self.train_config,
            "config_hash": self.config_hash,
            "execution_settings": self.config,
        }
        if extra_meta:
            meta.update(extra_meta)
            # Enforce that completed cannot be set to True if actual training did not reach total_epochs
            if not is_completed:
                meta["completed"] = False

        saved_path = self.checkpoint_manager.save_checkpoint(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=target_epoch,
            extra_meta=meta,
            allow_overwrite=allow_overwrite,
        )
        self.saved_checkpoints[filename] = saved_path
        self.last_saved_ckpt = saved_path
        return saved_path

    def train(self, start_epoch: int = 1, model_prefix: str = "model") -> Dict[str, Any]:
        """Execute full training loop across epochs."""
        import copy
        self._model_prefix = model_prefix
        # Preserve restored history if available; otherwise initialize
        if self.history and self.history.get("epochs") and len(self.history["epochs"]) > 0:
            history = self.history
        else:
            history = {"epochs": [], "train_loss": [], "lr": []}
            if start_epoch > 1:
                history["missing_epochs"] = list(range(1, start_epoch))
            self.history = history

        if self.val_loader is not None and len(self.val_loader) > 0:
            history.setdefault("val_loss", [])

        if start_epoch > self.total_epochs:
            logger.info(f"Target total_epochs ({self.total_epochs}) already completed. No additional epochs needed.")
            self.cur_epoch = self.total_epochs
            return history

        self.cur_epoch = start_epoch
        logger.info(f"Starting training from epoch {start_epoch} to {self.total_epochs} (Seed: {self.seed}, Keep Raw: {self.keep_raw_results})...")

        try:
            for epoch in range(start_epoch, self.total_epochs + 1):
                self.cur_epoch = epoch
                self.epochs_trained_in_run += 1
                avg_loss = self.train_one_epoch(epoch)
                current_lr = self.optimizer.param_groups[0]["lr"]
                val_loss = None
                if self.val_loader is not None and len(self.val_loader) > 0:
                    val_loss = self.validate()
                history["epochs"].append(epoch)
                history["train_loss"].append(avg_loss)
                history["lr"].append(current_lr)
                if val_loss is not None:
                    history["val_loss"].append(val_loss)

                self.scheduler.step()

                # Periodic saving
                if epoch % self.save_freq == 0 or epoch == self.total_epochs:
                    filename = f"{model_prefix}_epoch_{epoch}.pt"
                    saved_path = self.checkpoint_manager.save_checkpoint(
                        filename=filename,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        extra_meta={
                            "loss": avg_loss,
                            "lr": current_lr,
                            "parent_hash": self.parent_hash,
                            "epoch": epoch,
                            "total_epochs": self.total_epochs,
                            "seed": self.seed,
                            "stage": self.stage,
                            "head": self.head_type,
                            "is_smoke": False,
                            "completed": (epoch >= self.total_epochs),
                            "history": copy.deepcopy(self.history),
                            "config": self.train_config,
                            "config_hash": self.config_hash,
                            "execution_settings": self.config,
                        },
                        allow_overwrite=False,
                    )
                    self.saved_checkpoints[filename] = saved_path
                    self.last_saved_ckpt = saved_path

                    # When KEEP_RAW_RESULTS is False, prune older intermediate periodic checkpoints
                    if not self.keep_raw_results and epoch < self.total_epochs:
                        if self.last_periodic_ckpt and os.path.exists(self.last_periodic_ckpt):
                            try:
                                os.remove(self.last_periodic_ckpt)
                                pruned_fname = os.path.basename(self.last_periodic_ckpt)
                                self.saved_checkpoints.pop(pruned_fname, None)
                                logger.info(f"[Prune] Cleaned up older intermediate checkpoint: {self.last_periodic_ckpt}")
                            except Exception as pe:
                                logger.warning(f"Could not remove old checkpoint {self.last_periodic_ckpt}: {pe}")
                        self.last_periodic_ckpt = saved_path
                    elif epoch == self.total_epochs:
                        logger.info(f"[Formal Complete] Saved final epoch {epoch} checkpoint to {saved_path}")

            self._progress("completed", completed=True)
            logger.info("Training loop completed successfully.")
            return history
        except Exception as e:
            logger.error(f"Training loop failed: {e}", exc_info=True)
            raise
