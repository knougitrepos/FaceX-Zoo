"""Explicit FaceX backbones; keep upstream architectures unchanged."""
from backbone.MobileFaceNets import MobileFaceNet
from backbone.ResNets import Resnet


def canonical_backbone_type(value):
    """Case-insensitive input, stable names in newly generated metadata."""
    names = {"mobilefacenet": "MobileFaceNet", "resnet": "Resnet"}
    if not isinstance(value, str) or value.strip().lower() not in names:
        raise ValueError(f"Unsupported backbone type: {value!r}")
    return names[value.strip().lower()]


def baseline_learning_rate(config):
    """UCFace IV-A-3 downstream baseline learning rates."""
    if canonical_backbone_type(config.get("type", "MobileFaceNet")) == "MobileFaceNet":
        return 0.001
    if int(config.get("num_layers", 50)) not in (50, 100):
        raise ValueError("UCFace baseline LR is specified here only for Resnet-50/100")
    return 0.0001


def build_backbone(config):
    kind = canonical_backbone_type(config.get("type", "MobileFaceNet"))
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
    if canonical_backbone_type(config.get("type", "MobileFaceNet")) == "MobileFaceNet":
        return "mobilefacenet"
    return f"resnet{config.get('num_layers', 50)}_{config.get('mode', 'ir')}"
