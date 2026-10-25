"""Target-only FP32 angular margins with an explicit zero-radicand safeguard.

Local numerical policy, not an author implementation. Preserve the original
clamp, fallback branches and MagFace norm graph. Replace only a zero radicand
BEFORE sqrt; never mask an already evaluated singular sqrt backward.
"""
import math
import torch
import torch.nn.functional as F
from head.ArcFace import ArcFace as OriginalArcFace
from head.MagFace import MagFace as OriginalMagFace

NUMERICAL_POLICY = "target_only_fp32_zero_radicand_eps_v1"
BOUNDARY_EPS = torch.finfo(torch.float32).eps


def target_sine(target):
    if target.dtype != torch.float32:
        raise TypeError("Angular margins require FP32 cosine computation")
    radicand = 1.0 - target.square()
    safe = torch.where(radicand == 0, torch.full_like(radicand, BOUNDARY_EPS), radicand)
    return safe.sqrt()


def _require_fp32(feats, weight):
    if feats.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("Angular margins require FP32 features and weights; do not cast a precomputed FP16 cosine")


class ArcFace(OriginalArcFace):
    numerical_policy = NUMERICAL_POLICY

    def forward(self, feats, labels):
        _require_fp32(feats, self.weight)
        with torch.autocast(device_type=feats.device.type, enabled=False):
            return self._forward_fp32(feats, labels)

    def _forward_fp32(self, feats, labels):
        cosine = (F.normalize(feats) @ F.normalize(self.weight, dim=0)).clamp(-1, 1)
        rows = torch.arange(cosine.shape[0], device=cosine.device)
        labels = labels.reshape(-1).long()
        target = cosine[rows, labels]
        margin = target * self.cos_margin - target_sine(target) * self.sin_margin
        margin = torch.where(target > self.min_cos_theta, margin, target - self.margin_am)
        output = cosine.clone()
        output[rows, labels] = margin
        return output * self.scale


class MagFace(OriginalMagFace):
    numerical_policy = NUMERICAL_POLICY

    def forward(self, feats, labels):
        _require_fp32(feats, self.weight)
        with torch.autocast(device_type=feats.device.type, enabled=False):
            return self._forward_fp32(feats, labels)

    def _forward_fp32(self, feats, labels):
        norm = feats.norm(dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        adaptive_margin = self.calc_margin(norm).squeeze(1)
        regularizer = self.lamda * (1 / self.u_a**2 * norm + 1 / norm)
        cosine = (F.normalize(feats) @ F.normalize(self.weight, dim=0)).clamp(-1, 1)
        rows = torch.arange(cosine.shape[0], device=cosine.device)
        labels = labels.reshape(-1).long()
        target = cosine[rows, labels]
        margin = target * adaptive_margin.cos() - target_sine(target) * adaptive_margin.sin()
        margin = torch.where(target > torch.cos(math.pi - adaptive_margin), margin, target - self.margin_am)
        output = cosine.clone()
        output[rows, labels] = margin
        return output * self.scale, regularizer
