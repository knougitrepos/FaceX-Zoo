"""
FaceX-Zoo & UCFace Fine-Tuning and Evaluation Modules.
"""

from .dataset import FaceImageDataset, VGGFace2ArchiveInspector, build_dataloader
from .model_builder import FaceRecognitionModel, build_model
from .checkpoint_manager import CheckpointManager, compute_sha256, verify_parent_checkpoint, load_checkpoint
from .trainer import Trainer
from .evaluator import FeatureEvaluator

__all__ = [
    "FaceImageDataset",
    "VGGFace2ArchiveInspector",
    "build_dataloader",
    "FaceRecognitionModel",
    "build_model",
    "CheckpointManager",
    "compute_sha256",
    "load_checkpoint",
    "verify_parent_checkpoint",
    "Trainer",
    "FeatureEvaluator",
]
