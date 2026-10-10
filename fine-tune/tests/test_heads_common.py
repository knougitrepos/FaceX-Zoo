"""
Unit tests comparing ArcFace, AdaFace, and MagFace heads under identical conditions.
Verifies shapes, gradient computation, and baseline parameter compatibility.
"""

import sys
import os
import pytest
import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from head.ArcFace import ArcFace
from head.AdaFace import AdaFace
from head.MagFace import MagFace


@pytest.mark.parametrize("feat_dim", [512, 1024])
@pytest.mark.parametrize("batch_size", [4, 16, 64])
def test_all_heads_forward_shapes(feat_dim, batch_size):
    """Verify all three heads accept unnormalized feature embeddings and return proper shapes."""
    num_classes = 100
    labels = torch.randint(0, num_classes, (batch_size,))
    feats = torch.randn(batch_size, feat_dim)

    # 1. ArcFace (m=0.6, s=64.0 per UCFace paper)
    arcface = ArcFace(feat_dim=feat_dim, num_class=num_classes, margin_arc=0.6, margin_am=0.0, scale=64.0)
    out_arc = arcface(feats, labels)
    assert out_arc.shape == (batch_size, num_classes)
    assert not torch.isnan(out_arc).any()

    # 2. AdaFace (m=0.4, s=64.0, h=0.333 per UCFace paper)
    adaface = AdaFace(feat_dim=feat_dim, num_class=num_classes, m=0.4, h=0.333, scale=64.0, t_alpha=0.01)
    out_ada = adaface(feats, labels)
    assert out_ada.shape == (batch_size, num_classes)
    assert not torch.isnan(out_ada).any()

    # 3. MagFace (la=10, ua=110, lm=0.45, um=0.8, lambda=20, s=64.0 per UCFace paper)
    magface = MagFace(
        feat_dim=feat_dim,
        num_class=num_classes,
        margin_am=0.0,
        scale=64.0,
        l_a=10.0,
        u_a=110.0,
        l_margin=0.45,
        u_margin=0.8,
        lamda=20.0,
    )
    out_mag, loss_g = magface(feats, labels)
    assert out_mag.shape == (batch_size, num_classes)
    assert not torch.isnan(out_mag).any()
    assert loss_g.shape == (batch_size, 1)
    assert (loss_g > 0).all()


def test_loss_and_gradients_all_heads():
    """Verify loss and backward pass produce clean gradients for all three heads."""
    B = 8
    D = 1024
    C = 50

    feats_orig = torch.randn(B, D)
    labels = torch.randint(0, C, (B,))

    # 1. ArcFace
    f_arc = feats_orig.clone().requires_grad_(True)
    head_arc = ArcFace(D, C, margin_arc=0.6, margin_am=0.0, scale=64.0)
    out = head_arc(f_arc, labels)
    loss = F.cross_entropy(out, labels)
    loss.backward()
    assert f_arc.grad is not None
    assert not torch.isnan(f_arc.grad).any()

    # 2. AdaFace
    f_ada = feats_orig.clone().requires_grad_(True)
    head_ada = AdaFace(D, C, m=0.4, h=0.333, scale=64.0, t_alpha=0.01)
    out = head_ada(f_ada, labels)
    loss = F.cross_entropy(out, labels)
    loss.backward()
    assert f_ada.grad is not None
    assert not torch.isnan(f_ada.grad).any()

    # 3. MagFace
    f_mag = feats_orig.clone().requires_grad_(True)
    head_mag = MagFace(D, C, scale=64.0, l_a=10.0, u_a=110.0, l_margin=0.45, u_margin=0.8, lamda=20.0)
    out, loss_g = head_mag(f_mag, labels)
    loss = F.cross_entropy(out, labels) + torch.mean(loss_g)
    loss.backward()
    assert f_mag.grad is not None
    assert not torch.isnan(f_mag.grad).any()


def test_magface_norm_gradient_not_truncated():
    """Explicitly verify that MagFace norm regularization (loss_g) flows gradients back to features."""
    B = 4
    D = 512
    C = 20
    f_mag = torch.randn(B, D, requires_grad=True)
    labels = torch.randint(0, C, (B,))
    head_mag = MagFace(D, C, scale=64.0, l_a=10.0, u_a=110.0, l_margin=0.45, u_margin=0.8, lamda=20.0)
    out, loss_g = head_mag(f_mag, labels)

    # Backward solely through norm regularization loss
    loss_reg = torch.mean(loss_g)
    loss_reg.backward()

    assert f_mag.grad is not None
    assert (f_mag.grad != 0).any()
    assert not torch.isnan(f_mag.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
