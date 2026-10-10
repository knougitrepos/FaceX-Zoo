"""
@author: Adapted for FaceX-Zoo from mk-minchul/AdaFace
Paper: AdaFace: Quality Adaptive Margin for Face Recognition (CVPR 2022)
Original repository: https://github.com/mk-minchul/AdaFace
"""

import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter


def l2_norm(input_tensor, axis=1):
    """L2 normalization along a specified axis."""
    norm = torch.norm(input_tensor, 2, axis, True)
    output = torch.div(input_tensor, norm)
    return output


class AdaFace(Module):
    """AdaFace head implementation compatible with FaceX-Zoo training interface.

    AdaFace dynamically scales angular and additive margins based on feature norms,
    giving adaptive penalties to face samples with different image qualities.

    Mathematical formulation (from Kim et al., CVPR 2022):
        \\hat{||r_i||} = clip((||r_i|| - \\mu_r) / (\\sigma_r / h), -1, 1)
        g_{angular} = -m * \\hat{||r_i||}
        g_{additive} = m + m * \\hat{||r_i||}
        cosine = cos(theta + g_{angular}) - g_{additive}
        output = s * cosine

    Args:
        feat_dim (int): Dimension of feature embeddings (e.g. 1024 or 512). Default: 1024.
        num_class (int): Number of identity classes in the training dataset (e.g. 8631 for VGGFace2).
        m (float): Base margin parameter. Default: 0.4.
        h (float): Margin concentration hyperparameter. Default: 0.333.
        scale (float): Cosine scaling factor s. Default: 64.0.
        t_alpha (float): Moving average rate for feature norm statistics. Default: 0.01.
    """

    def __init__(
        self,
        feat_dim=1024,
        num_class=8631,
        m=0.4,
        h=0.333,
        scale=64.0,
        t_alpha=0.01,
        **kwargs
    ):
        super(AdaFace, self).__init__()
        # Support both naming conventions: feat_dim / embedding_size, num_class / classnum
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.m = float(m)
        self.h = float(h)
        self.s = float(scale if scale is not None else kwargs.get("s", 64.0))
        self.scale = self.s
        self.t_alpha = float(t_alpha)
        self.eps = 1e-3

        # Kernel / Weight matrix [feat_dim, num_class]
        self.kernel = Parameter(torch.Tensor(self.feat_dim, self.num_class))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        # Exponential moving statistics for feature norm
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    @property
    def weight(self):
        """Alias for compatibility with FaceX-Zoo heads that access self.weight."""
        return self.kernel

    @weight.setter
    def weight(self, value):
        self.kernel = value

    def compute_logits(self, embeddings, norms, labels):
        """Compute AdaFace scaled logits given normalized embeddings, norms, and labels.

        Args:
            embeddings (torch.Tensor): L2-normalized feature vectors, shape [B, feat_dim].
            norms (torch.Tensor): Feature norms, shape [B, 1].
            labels (torch.Tensor): Ground-truth class labels, shape [B] or [B, 1].

        Returns:
            torch.Tensor: Scaled cosine logits with adaptive margin, shape [B, num_class].
        """
        labels = labels.view(-1)
        norms = norms.view(-1, 1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # Stability clamp

        safe_norms = torch.clip(norms, min=0.001, max=100.0)
        safe_norms = safe_norms.clone().detach()

        # Update batch mean and batch std using EMA (in training mode)
        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean().detach()
                self.batch_mean = mean * self.t_alpha + (1.0 - self.t_alpha) * self.batch_mean
                # Guard against NaN when batch_size is 1 (torch.std with 1 sample produces NaN)
                if safe_norms.numel() > 1:
                    std = safe_norms.std().detach()
                    self.batch_std = std * self.t_alpha + (1.0 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1.0, 1.0)

        # g_angular: m * margin_scaler * -1
        m_arc = torch.zeros(labels.size(0), cosine.size(1), device=cosine.device)
        m_arc.scatter_(1, labels.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1.0
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive: m + (m * margin_scaler)
        m_cos = torch.zeros(labels.size(0), cosine.size(1), device=cosine.device)
        m_cos.scatter_(1, labels.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # Scale by s
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m

    def forward(self, feats, labels, norms=None):
        """Forward pass supporting both FaceX-Zoo adapter call and direct official call.

        - FaceX-Zoo interface:
            forward(feats, labels) where feats are unnormalized embeddings from backbone.
            L2 norm is calculated automatically and embeddings are normalized.
        - Direct call with explicit norms:
            forward(embeddings, labels, norms=norms) or forward(embeddings, norms, labels).
        """
        if norms is not None:
            # Check if caller used official signature: forward(embeddings, norms, labels)
            # where 2nd arg was norms and 3rd arg was labels
            if isinstance(labels, torch.Tensor) and labels.dtype in (torch.float32, torch.float64) and norms.dtype in (torch.int64, torch.int32, torch.long):
                return self.compute_logits(feats, labels, norms)
            return self.compute_logits(feats, norms, labels)

        # FaceX-Zoo Adapter mode: feats are unnormalized feature vectors from backbone
        norm = torch.norm(feats, p=2, dim=1, keepdim=True)
        embeddings = torch.div(feats, norm.clamp(min=1e-5))
        return self.compute_logits(embeddings, norm, labels)
