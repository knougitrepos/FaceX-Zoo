"""
Unified Model Builder for Face Recognition Backbones and Margin Heads.
Supports MobileFaceNet (1024D / 512D) paired with ArcFace, AdaFace, MagFace, and CosFace.
"""

import sys
import os
import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

# Include parent directory to load FaceX-Zoo backbone and head modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backbone.MobileFaceNets import MobileFaceNet
from head.ArcFace import ArcFace
from head.AdaFace import AdaFace
from head.MagFace import MagFace
from head.AM_Softmax import AM_Softmax

logger = logging.getLogger("fine_tune.model")


class FaceRecognitionModel(nn.Module):
    """Encapsulates backbone and angular margin head into a cohesive training module.

    Attributes:
        backbone (nn.Module): Feature extractor (e.g. MobileFaceNet).
        head (nn.Module): Margin classifier (ArcFace, AdaFace, MagFace, CosFace).
        head_type (str): 'ArcFace', 'AdaFace', 'MagFace', or 'CosFace'.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, head_type: str):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = backbone
        self.head = head
        self.head_type = head_type

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Logits of shape [B, num_classes]
                - Auxiliary loss tensor (for MagFace: loss_g; for others: None)
        """
        feats = self.backbone(images)

        if self.head_type.lower() == "magface":
            logits, loss_g = self.head(feats, labels)
            return logits, loss_g
        else:
            logits = self.head(feats, labels)
            return logits, None

    def extract_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference feature extraction.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Normalized embeddings of shape [B, feat_dim]
                - L2 feature norms of shape [B, 1]
        """
        feats = self.backbone(images)
        norms = torch.norm(feats, p=2, dim=1, keepdim=True)
        normed_feats = torch.div(feats, norms.clamp(min=1e-5))
        return normed_feats, norms


def build_model(
    backbone_conf: Dict[str, Any],
    head_conf: Dict[str, Any],
) -> FaceRecognitionModel:
    """Build unified FaceRecognitionModel from configuration dictionaries.

    Args:
        backbone_conf (dict): Configuration for backbone (feat_dim, dropout_rate, etc.).
        head_conf (dict): Configuration for head (type, num_class, margins, scale, etc.).

    Returns:
        FaceRecognitionModel
    """
    feat_dim = int(backbone_conf.get("feat_dim", 1024))
    out_h = int(backbone_conf.get("out_h", 7))
    out_w = int(backbone_conf.get("out_w", 7))
    dropout_rate = float(backbone_conf.get("dropout_rate", 0.6))

    backbone_type = backbone_conf.get("type", "MobileFaceNet")
    if backbone_type == "MobileFaceNet":
        backbone = MobileFaceNet(
            embedding_size=feat_dim,
            out_h=out_h,
            out_w=out_w,
            dropout_rate=dropout_rate,
        )
    else:
        logger.error(f"Unsupported backbone type: {backbone_type}. Supported: 'MobileFaceNet'")
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    head_type = head_conf.get("type", "ArcFace")
    num_class = int(head_conf.get("num_class", 8631))

    if head_type.lower() == "arcface":
        margin_arc = float(head_conf.get("margin_arc", 0.6))
        margin_am = float(head_conf.get("margin_am", 0.0))
        scale = float(head_conf.get("scale", 64.0))
        head = ArcFace(
            feat_dim=feat_dim,
            num_class=num_class,
            margin_arc=margin_arc,
            margin_am=margin_am,
            scale=scale,
        )
    elif head_type.lower() == "cosface":
        # FaceX-Zoo's existing additive cosine-margin implementation.
        # UCFace IV-A-2 low-resolution baseline: m=0.3, s=64.
        head = AM_Softmax(
            feat_dim=feat_dim,
            num_class=num_class,
            margin=float(head_conf.get("margin", 0.3)),
            scale=float(head_conf.get("scale", 64.0)),
        )
    elif head_type.lower() == "adaface":
        m = float(head_conf.get("m", 0.4))
        h = float(head_conf.get("h", 0.333))
        scale = float(head_conf.get("scale", 64.0))
        t_alpha = float(head_conf.get("t_alpha", 0.01))
        head = AdaFace(
            feat_dim=feat_dim,
            num_class=num_class,
            m=m,
            h=h,
            scale=scale,
            t_alpha=t_alpha,
        )
    elif head_type.lower() == "magface":
        margin_am = float(head_conf.get("margin_am", 0.0))
        scale = float(head_conf.get("scale", 64.0))
        l_a = float(head_conf.get("l_a", 10.0))
        u_a = float(head_conf.get("u_a", 110.0))
        l_margin = float(head_conf.get("l_margin", 0.45))
        u_margin = float(head_conf.get("u_margin", 0.8))
        lamda = float(head_conf.get("lamda", 20.0))
        head = MagFace(
            feat_dim=feat_dim,
            num_class=num_class,
            margin_am=margin_am,
            scale=scale,
            l_a=l_a,
            u_a=u_a,
            l_margin=l_margin,
            u_margin=u_margin,
            lamda=lamda,
        )
    else:
        logger.error(f"Unsupported head type: {head_type}. Supported: 'ArcFace', 'AdaFace', 'MagFace', 'CosFace'")
        raise ValueError(f"Unsupported head type: {head_type}")

    logger.info(
        f"Built FaceRecognitionModel: Backbone={backbone_type}(dim={feat_dim}, dropout={dropout_rate}), "
        f"Head={head_type}(num_class={num_class})"
    )
    return FaceRecognitionModel(backbone=backbone, head=head, head_type=head_type)
