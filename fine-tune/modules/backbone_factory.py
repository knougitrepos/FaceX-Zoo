"""Explicit FaceX backbones; keep upstream architectures unchanged."""
from backbone.MobileFaceNets import MobileFaceNet
from backbone.ResNets import Resnet


def build_backbone(config):
    kind = config.get("type", "MobileFaceNet")
    dim = int(config.get("feat_dim", 1024))
    dropout = float(config.get("dropout_rate", .6))
    if dim < 1:
        raise ValueError("Positive feature dimension required")
    if kind == "MobileFaceNet":
        return MobileFaceNet(embedding_size=dim, out_h=7, out_w=7, dropout_rate=dropout)
    if kind not in ("Resnet", "ResNet"):
        raise ValueError(f"Unsupported backbone type: {kind}")
    depth, mode = int(config.get("num_layers", 50)), config.get("mode", "ir")
    if depth not in (50, 100, 152) or mode not in ("ir", "ir_se"):
        raise ValueError("Resnet requires depth 50/100/152 and ir/ir_se")
    model = Resnet(depth, dropout, mode, feat_dim=dim, out_h=7, out_w=7)
    model.num_layers, model.mode = depth, mode
    model.dropout_rate, model.out_h, model.out_w = dropout, 7, 7
    return model


def backbone_slug(config):
    if config.get("type") == "MobileFaceNet":
        return "mobilefacenet"
    return f"resnet{config.get('num_layers', 50)}_{config.get('mode', 'ir')}"
