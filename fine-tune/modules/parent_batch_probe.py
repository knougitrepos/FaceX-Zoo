"""Short FP32 training-step benchmark for the local WD0 VGGFace2 parent."""
import math
import statistics
import time

import torch
from torch.nn.functional import cross_entropy

from .model_builder import build_model
from .trainer import set_seed


def measure_batch(batch, loader=None):
    """Compare equal image counts with GPU tensors or four blocks from a real loader."""
    if batch not in (64, 256):
        raise ValueError("Benchmark supports 64 and 256")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    set_seed(8972)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = dict(batch=batch, device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  cuda=torch.version.cuda, images_per_epoch=3141890,
                  steps_per_epoch=math.ceil(3141890 / batch),
                  input="synthetic GPU tensors, FP32, Adam WD0, finite loss and gradient checks")
    try:
        model = build_model({"type": "MobileFaceNet", "feat_dim": 1024, "dropout_rate": .6},
                            {"type": "ArcFace", "num_class": 8631, "margin_arc": .35, "scale": 32}).cuda().train()
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        if loader is None:
            images = torch.randn(batch, 3, 112, 112, device="cuda")
            labels = torch.arange(batch, device="cuda") % 8631
        else:
            result["input"] = "warm-cache aligned VGGFace2, four workers, augmentation and H2D included"
            iterator = iter(loader)
        times = []
        for block in range(4):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(1024 // batch):
                if loader is not None:
                    x, y = next(iterator)
                    images, labels = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits, _ = model(images, labels)
                loss = cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None], float("inf"))
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradient")
                opt.step()
            torch.cuda.synchronize()
            if block:
                times.append(time.perf_counter() - start)
        result.update(status="completed", measured_images=3072, block_seconds=times,
                      images_per_second=1024 / statistics.median(times))
    except torch.cuda.OutOfMemoryError:
        result.update(status="cuda_out_of_memory")
    result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    return result
