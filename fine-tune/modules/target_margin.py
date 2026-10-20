"""Target-only FaceX-Zoo angular margins (local implementation, not author code).

No sqrt floor: non-target boundary derivatives are never evaluated. Target
cosine +/-1 remains singular and must be caught before an optimizer update.
Parameter names/shapes and the original clamp/branch/regularizer are preserved.
"""
import math
import torch
import torch.nn.functional as F
from head.ArcFace import ArcFace as OriginalArcFace
from head.MagFace import MagFace as OriginalMagFace

NUMERICAL_POLICY = "target_only_original_sqrt_v1"


class ArcFace(OriginalArcFace):
    numerical_policy = NUMERICAL_POLICY

    def forward(self, feats, labels):
        cosine = (F.normalize(feats) @ F.normalize(self.weight, dim=0)).clamp(-1, 1)
        rows = torch.arange(cosine.shape[0], device=cosine.device)
        labels = labels.reshape(-1).long()
        target = cosine[rows, labels]
        margin = target * self.cos_margin - torch.sqrt(1.0 - target.pow(2)) * self.sin_margin
        margin = torch.where(target > self.min_cos_theta, margin, target - self.margin_am)
        output = cosine.clone()
        output[rows, labels] = margin
        return output * self.scale


class MagFace(OriginalMagFace):
    numerical_policy = NUMERICAL_POLICY

    def forward(self, feats, labels):
        norm = feats.norm(dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        adaptive_margin = self.calc_margin(norm).squeeze(1)
        regularizer = self.lamda * (1 / self.u_a**2 * norm + 1 / norm)
        cosine = (F.normalize(feats) @ F.normalize(self.weight, dim=0)).clamp(-1, 1)
        rows = torch.arange(cosine.shape[0], device=cosine.device)
        labels = labels.reshape(-1).long()
        target = cosine[rows, labels]
        margin = target * adaptive_margin.cos() - torch.sqrt(1.0 - target.pow(2)) * adaptive_margin.sin()
        margin = torch.where(target > torch.cos(math.pi - adaptive_margin), margin, target - self.margin_am)
        output = cosine.clone()
        output[rows, labels] = margin
        return output * self.scale, regularizer
