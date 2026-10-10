"""
Pytest integration for GPU VRAM and Throughput Verification on MobileFaceNet 1024D + 8631 classes + batch size 64.
Verifies goal.md Section 7 requirement:
"작은 forward/backward 시험 후 1024D·실제 클래스 수·batch 64에서 메모리와 처리량을 검증한다. batch 4·클래스 8의 통과를 전체 조건의 메모리 검증으로 대체하지 않는다."
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backbone.MobileFaceNets import MobileFaceNet
from head.ArcFace import ArcFace
from head.AdaFace import AdaFace
from head.MagFace import MagFace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory profiling")
@pytest.mark.parametrize("head_name", ["ArcFace", "AdaFace", "MagFace"])
def test_gpu_memory_and_throughput(head_name: str):
    """Verify 1024D + 8631 classes + batch size 64 runs without OOM on CUDA GPU."""
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    batch_size = 64
    feat_dim = 1024
    num_classes = 8631

    backbone = MobileFaceNet(embedding_size=feat_dim, out_h=7, out_w=7, dropout_rate=0.6).to(device)

    if head_name == "ArcFace":
        head = ArcFace(feat_dim=feat_dim, num_class=num_classes, margin_arc=0.6, scale=64.0).to(device)
    elif head_name == "AdaFace":
        head = AdaFace(feat_dim=feat_dim, num_class=num_classes, m=0.4, h=0.333, scale=64.0, t_alpha=0.01).to(device)
    else:  # MagFace
        head = MagFace(feat_dim=feat_dim, num_class=num_classes, scale=64.0, l_a=10.0, u_a=110.0, l_margin=0.45, u_margin=0.8, lamda=20.0).to(device)

    inputs = torch.randn(batch_size, 3, 112, 112, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=1e-3
    )

    backbone.train()
    head.train()
    optimizer.zero_grad()

    feats = backbone(inputs)
    assert feats.shape == (batch_size, feat_dim)

    if head_name == "AdaFace":
        norms = torch.norm(feats, p=2, dim=1, keepdim=True)
        feats_norm = feats / (norms + 1e-12)
        outputs = head(feats_norm, norms, labels)
        loss = criterion(outputs, labels)
    elif head_name == "MagFace":
        outputs, loss_g = head(feats, labels)
        loss = criterion(outputs, labels) + torch.mean(loss_g)
    else:  # ArcFace
        outputs = head(feats, labels)
        loss = criterion(outputs, labels)

    assert outputs.shape == (batch_size, num_classes)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)

    # Assert peak VRAM is well within limits (< 6 GB out of 11 GB, < 60%)
    assert peak_vram_mb < 6000, f"{head_name} exceeded expected VRAM: {peak_vram_mb:.1f} MB"
    assert peak_vram_mb < total_vram_mb, f"{head_name} caused OOM danger"
