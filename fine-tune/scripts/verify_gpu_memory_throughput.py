"""
Verification of GPU VRAM footprint and throughput for MobileFaceNet 1024D + 8631 classes + batch size 64.
Verifies goal.md Section 7 requirement:
"작은 forward/backward 시험 후 1024D·실제 클래스 수·batch 64에서 메모리와 처리량을 검증한다."
"""

import sys
import os
import time
import torch
import torch.nn as nn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backbone.MobileFaceNets import MobileFaceNet
from head.ArcFace import ArcFace
from head.AdaFace import AdaFace
from head.MagFace import MagFace

def profile_head(head_name: str, head_module: nn.Module, backbone: nn.Module, device: torch.device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    batch_size = 64
    feat_dim = 1024
    num_classes = 8631
    
    inputs = torch.randn(batch_size, 3, 112, 112, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head_module.parameters()), lr=1e-3
    )
    
    # Warmup
    backbone.train()
    head_module.train()
    optimizer.zero_grad()
    
    start_time = time.time()
    feats = backbone(inputs)
    
    if head_name == "AdaFace":
        norms = torch.norm(feats, p=2, dim=1, keepdim=True)
        feats_norm = feats / (norms + 1e-12)
        outputs = head_module(feats_norm, norms, labels)
    elif head_name == "MagFace":
        outputs, loss_g = head_module(feats, labels)
    else:  # ArcFace
        outputs = head_module(feats, labels)
        
    loss = criterion(outputs, labels)
    if head_name == "MagFace":
        loss = loss + torch.mean(loss_g)
        
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    step_time = (time.time() - start_time) * 1000.0  # ms
    
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    throughput = batch_size / (step_time / 1000.0)
    
    print(f"[{head_name:7s}] Step Time: {step_time:6.2f} ms | Throughput: {throughput:5.1f} img/s | Allocated: {allocated:6.1f} MB | Peak VRAM: {peak:6.1f} MB ({peak/1024:4.2f} GB)")
    return {
        "head": head_name,
        "step_time_ms": step_time,
        "throughput_img_per_s": throughput,
        "allocated_mb": allocated,
        "peak_vram_mb": peak,
        "peak_vram_gb": peak / 1024,
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU profiling.")
        return
        
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print("=" * 80)
    print(f"GPU Device: {props.name} (Compute Capability: {props.major}.{props.minor})")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"Target Configuration: MobileFaceNet 1024D | Num Classes: 8631 | Batch Size: 64")
    print("=" * 80)
    
    backbone = MobileFaceNet(embedding_size=1024, out_h=7, out_w=7, dropout_rate=0.6).to(device)
    
    heads = {
        "ArcFace": ArcFace(feat_dim=1024, num_class=8631, margin_arc=0.6, scale=64.0).to(device),
        "AdaFace": AdaFace(feat_dim=1024, num_class=8631, m=0.4, h=0.333, scale=64.0, t_alpha=0.01).to(device),
        "MagFace": MagFace(feat_dim=1024, num_class=8631, scale=64.0, l_a=10.0, u_a=110.0, l_margin=0.45, u_margin=0.8, lamda=20.0).to(device),
    }
    
    results = []
    for name, head in heads.items():
        res = profile_head(name, head, backbone, device)
        results.append(res)
        
    print("=" * 80)
    print("SUMMARY & OOM SAFETY VERIFICATION:")
    for res in results:
        head = res["head"]
        peak_gb = res["peak_vram_gb"]
        usage_pct = (peak_gb / total_vram_gb) * 100
        print(f"- {head:7s}: Peak VRAM {peak_gb:.2f} GB / {total_vram_gb:.2f} GB ({usage_pct:.1f}% used) -> SAFE (NO OOM)")
    print("=" * 80)

if __name__ == "__main__":
    main()
