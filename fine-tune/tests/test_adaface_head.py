"""
Unit tests for AdaFace Head Integration in FaceX-Zoo.
Verifies numerical consistency with official AdaFace, gradient flow, and EMA moving statistics.
"""

import sys
import os
import math
import pytest
import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from head.AdaFace import AdaFace, l2_norm


def test_adaface_initialization():
    """Verify AdaFace initialization parameters and buffer registration."""
    head = AdaFace(feat_dim=1024, num_class=8631, m=0.4, h=0.333, scale=64.0, t_alpha=0.01)
    
    assert head.feat_dim == 1024
    assert head.num_class == 8631
    assert math.isclose(head.m, 0.4, rel_tol=1e-5)
    assert math.isclose(head.h, 0.333, rel_tol=1e-5)
    assert math.isclose(head.s, 64.0, rel_tol=1e-5)
    assert math.isclose(head.t_alpha, 0.01, rel_tol=1e-5)
    
    # Check weight / kernel shape
    assert head.kernel.shape == (1024, 8631)
    assert head.weight.shape == (1024, 8631)
    
    # Check registered buffers
    assert hasattr(head, "batch_mean")
    assert hasattr(head, "batch_std")
    assert hasattr(head, "t")
    assert math.isclose(head.batch_mean.item(), 20.0, rel_tol=1e-5)
    assert math.isclose(head.batch_std.item(), 100.0, rel_tol=1e-5)


def test_adaface_adapter_forward_shape():
    """Verify that FaceX-Zoo adapter forward(feats, labels) accepts unnormalized feats and returns [B, num_classes]."""
    B = 8
    feat_dim = 1024
    num_class = 100
    head = AdaFace(feat_dim=feat_dim, num_class=num_class, m=0.4, h=0.333, scale=64.0, t_alpha=0.01)
    
    # Unnormalized features (e.g. from MobileFaceNet output)
    feats = torch.randn(B, feat_dim, requires_grad=True)
    labels = torch.randint(0, num_class, (B,))
    
    logits = head(feats, labels)
    
    assert logits.shape == (B, num_class)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_adaface_official_mode_equivalence():
    """Verify that adapter mode produces identical logits to manual normalization + direct calling."""
    torch.manual_seed(123)
    B = 4
    feat_dim = 512
    num_class = 50
    head = AdaFace(feat_dim=feat_dim, num_class=num_class, m=0.4, h=0.333, scale=64.0, t_alpha=0.01)
    head.eval()  # Freeze EMA update for exact comparison

    # Create unnormalized feats
    feats = torch.randn(B, feat_dim)
    labels = torch.tensor([0, 5, 12, 49])

    # Mode 1: FaceX-Zoo adapter mode
    logits_adapter = head(feats, labels)

    # Mode 2: Manual normalization matching official AdaFace calling convention
    norm = torch.norm(feats, p=2, dim=1, keepdim=True)
    normed_feats = torch.div(feats, norm)
    logits_official = head.compute_logits(normed_feats, norm, labels)

    assert torch.allclose(logits_adapter, logits_official, atol=1e-5)


def test_adaface_backward_gradient():
    """Verify gradient flows properly back through unnormalized features and kernel."""
    B = 4
    feat_dim = 256
    num_class = 10
    head = AdaFace(feat_dim=feat_dim, num_class=num_class, m=0.4, h=0.333, scale=64.0, t_alpha=0.01)
    head.train()

    feats = torch.randn(B, feat_dim, requires_grad=True)
    labels = torch.tensor([1, 3, 7, 2])

    logits = head(feats, labels)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    assert feats.grad is not None
    assert feats.grad.shape == feats.shape
    assert not torch.isnan(feats.grad).any()
    assert head.kernel.grad is not None
    assert not torch.isnan(head.kernel.grad).any()


def test_adaface_ema_buffers_update():
    """Verify EMA buffers batch_mean and batch_std update during train and freeze during eval."""
    head = AdaFace(feat_dim=128, num_class=20, m=0.4, h=0.333, scale=64.0, t_alpha=0.1)
    init_mean = head.batch_mean.item()
    init_std = head.batch_std.item()

    # Training mode: buffers should update
    head.train()
    feats = torch.randn(16, 128) * 15.0  # distinctive scale
    labels = torch.randint(0, 20, (16,))
    _ = head(feats, labels)

    assert not math.isclose(head.batch_mean.item(), init_mean, rel_tol=1e-4)
    assert not math.isclose(head.batch_std.item(), init_std, rel_tol=1e-4)

    # Eval mode: buffers should remain frozen
    head.eval()
    frozen_mean = head.batch_mean.item()
    frozen_std = head.batch_std.item()
    _ = head(feats, labels)

    assert math.isclose(head.batch_mean.item(), frozen_mean, rel_tol=1e-7)
    assert math.isclose(head.batch_std.item(), frozen_std, rel_tol=1e-7)


def test_adaface_batch_size_1_ema_safety():
    """Verify that batch_size=1 during training does NOT corrupt batch_std to NaN."""
    head = AdaFace(feat_dim=128, num_class=20, m=0.4, h=0.333, scale=64.0, t_alpha=0.01)
    head.train()

    # Single sample
    feat_single = torch.randn(1, 128)
    label_single = torch.tensor([5])

    logits = head(feat_single, label_single)

    assert logits.shape == (1, 20)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(head.batch_std).any()
    assert not torch.isnan(head.batch_mean).any()


def test_adaface_1d_norms_shape_support():
    """Verify AdaFace handles 1D norms tensor of shape [B] without broadcast crash."""
    B = 4
    feat_dim = 256
    num_class = 30
    head = AdaFace(feat_dim=feat_dim, num_class=num_class, m=0.4, h=0.333, scale=64.0)
    head.eval()

    embs = torch.randn(B, feat_dim)
    norms_1d = torch.norm(embs, p=2, dim=1)  # Shape [B]
    embs_normed = embs / norms_1d.unsqueeze(1)
    labels = torch.randint(0, num_class, (B,))

    # Pass 1D norms tensor: shape [B]
    logits = head.compute_logits(embs_normed, norms_1d, labels)
    assert logits.shape == (B, num_class)
    assert not torch.isnan(logits).any()


def test_adaface_official_call_signature():
    """Verify calling head(embeddings, norms, labels) with official 3-argument signature."""
    B = 4
    feat_dim = 256
    num_class = 20
    head = AdaFace(feat_dim=feat_dim, num_class=num_class)
    head.eval()

    embs = torch.randn(B, feat_dim)
    norms = torch.norm(embs, p=2, dim=1, keepdim=True)
    normed_embs = embs / norms
    labels = torch.tensor([0, 3, 7, 19], dtype=torch.long)

    # Official calling signature: head(embeddings, norms, labels)
    logits = head(normed_embs, norms, labels)
    assert logits.shape == (B, num_class)
    assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
