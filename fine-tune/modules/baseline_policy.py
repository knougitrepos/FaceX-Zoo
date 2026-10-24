"""Versioned research choices for the adjust1 campaign."""
import hashlib
from pathlib import Path

VERSION = "adjust1_preprocessing_v2"
DEFAULT_AUGMENTATION = {
    "flip_probability": 0.5, "affine_probability": 0.5,
    "rotation_degrees": 10.0, "scale_range": [0.95, 1.05],
    "border_mode": "reflect", "center": "image_center",
}


def code_fingerprints():
    root = Path(__file__).resolve().parents[2]
    paths = ["fine-tune/modules/" + name + ".py" for name in (
        "target_margin", "baseline_policy", "alignment", "model_builder",
        "dataset", "input_pipeline", "trainer", "checkpoint_manager", "notebook_workflow",
        "backbone_factory", "finetune_protocol", "research_protocol")]
    paths += ["backbone/MobileFaceNets.py", "backbone/ResNets.py", "head/ArcFace.py", "head/MagFace.py",
              "head/AdaFace.py", "head/AM_Softmax.py"]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def validate_augmentation(options):
    import math
    if set(options) != set(DEFAULT_AUGMENTATION):
        raise ValueError("Specify all augmentation fields; unknown fields are not accepted")
    for name in ("flip_probability", "affine_probability"):
        if not math.isfinite(options[name]) or not 0 <= options[name] <= 1:
            raise ValueError(f"Invalid augmentation {name}")
    low, high = options["scale_range"]
    if not all(math.isfinite(v) for v in (low, high, options["rotation_degrees"])) or not 0 < low <= high or options["rotation_degrees"] < 0:
        raise ValueError("Invalid affine range")
    if options["border_mode"] not in ("reflect", "constant") or options["center"] != "image_center":
        raise ValueError("Unsupported affine padding/center")
