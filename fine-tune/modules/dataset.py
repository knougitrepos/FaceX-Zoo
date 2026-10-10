"""
VGGFace2 Dataset Inspector, Preprocessing, and DataLoader Utilities for FaceX-Zoo.
"""

import os
import csv
import tarfile
import random
import logging
import hashlib
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("fine_tune.dataset")


class VGGFace2ArchiveInspector:
    """Helper to inspect VGGFace2 archives without requiring immediate full decompression."""

    def __init__(self, raw_root: str):
        self.raw_root = raw_root
        self.train_tar = os.path.join(raw_root, "data", "vggface2_train.tar.gz")
        self.test_tar = os.path.join(raw_root, "data", "vggface2_test.tar.gz")
        self.meta_csv = os.path.join(raw_root, "meta", "identity_meta.csv")

    def check_archives_exist(self) -> Dict[str, bool]:
        """Check presence of essential archives and metadata."""
        status = {
            "train_tar_exists": os.path.exists(self.train_tar),
            "test_tar_exists": os.path.exists(self.test_tar),
            "meta_csv_exists": os.path.exists(self.meta_csv),
        }
        if status["train_tar_exists"]:
            status["train_tar_size_bytes"] = os.path.getsize(self.train_tar)
            status["train_tar_size_gb"] = round(status["train_tar_size_bytes"] / (1024**3), 2)
        if status["test_tar_exists"]:
            status["test_tar_size_bytes"] = os.path.getsize(self.test_tar)
            status["test_tar_size_gb"] = round(status["test_tar_size_bytes"] / (1024**3), 2)
        return status

    def parse_identity_meta(self, train_only: bool = True) -> List[Dict[str, str]]:
        """Parse identity_meta.csv to retrieve class IDs and attributes.
        
        Args:
            train_only (bool): If True, returns only the 8,631 training identities (Flag == '1').
                               If False, returns all 9,131 identities (including 500 test IDs).
        """
        if not os.path.exists(self.meta_csv):
            logger.error(f"identity_meta.csv not found at {self.meta_csv}")
            raise FileNotFoundError(f"identity_meta.csv not found at {self.meta_csv}")
        
        identities = []
        try:
            with open(self.meta_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f, skipinitialspace=True)
                header = next(reader, None)
                for parts in reader:
                    if not parts or len(parts) < 2:
                        continue
                    parts = [p.strip() for p in parts]
                    flag = parts[3] if len(parts) > 3 else "0"
                    if train_only and flag != "1":
                        continue
                    identities.append({
                        "class_id": parts[0],
                        "name": parts[1] if len(parts) > 1 else "",
                        "sample_num": parts[2] if len(parts) > 2 else "0",
                        "flag": flag,
                    })
            logger.info(f"Parsed {len(identities)} identities from {self.meta_csv} (train_only={train_only})")
            return identities
        except Exception as e:
            logger.error(f"Error parsing identity metadata from {self.meta_csv}: {e}", exc_info=True)
            raise


def apply_face_augmentation(image: np.ndarray, is_train: bool = True) -> np.ndarray:
    """Apply data augmentation matching UCFace paper settings.
    Baseline margin classifiers use: horizontal flip, affine scaling and rotation.
    Input image shape: [H, W, C] in BGR.
    """
    if not is_train:
        return image

    # Horizontal flip (50% probability)
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Affine transformation: rotation (-10 to 10 deg) and slight scaling (0.95 to 1.05)
    if random.random() > 0.5:
        h, w = image.shape[:2]
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.95, 1.05)
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    return image


def resolve_image_path(data_root: str, rel_path: str) -> str:
    """Resolve full image path, seamlessly handling 'train/' prefix differences.

    Checks:
    1. os.path.join(data_root, rel_clean)
    2. os.path.join(data_root, 'train', rel_clean) (for archives unpacked into a train/ subfolder)
    3. Stripping 'train/' if data_root already ends with 'train'
    4. Fallback to candidate 1 (will be caught by fail-closed os.path.exists check)
    """
    rel_clean = rel_path.replace("\\", "/").strip()

    candidate1 = os.path.normpath(os.path.join(data_root, rel_clean))
    if os.path.exists(candidate1):
        return candidate1

    if not rel_clean.startswith("train/"):
        candidate2 = os.path.normpath(os.path.join(data_root, "train", rel_clean))
        if os.path.exists(candidate2):
            return candidate2

    if rel_clean.startswith("train/"):
        stripped = rel_clean[len("train/"):]
        candidate3 = os.path.normpath(os.path.join(data_root, stripped))
        if os.path.exists(candidate3):
            return candidate3

    return candidate1


class FaceImageDataset(Dataset):
    """PyTorch Dataset for Face Recognition.
    
    Reads images in BGR format, resizes to (112, 112), applies augmentation,
    and normalizes pixels using:
        normalized_pixel = (pixel - 127.5) / 128.0 (which equals (pixel - 127.5) * 0.0078125)
    Consistent across FaceX-Zoo and ronbun inference.
    """

    def __init__(
        self,
        data_root: str,
        index_file: str,
        target_size: Tuple[int, int] = (112, 112),
        is_train: bool = True,
        pixel_mean: float = 127.5,
        pixel_std: float = 128.0,
    ):
        self.data_root = data_root
        self.index_file = index_file
        self.target_size = tuple(int(x) for x in target_size)
        if len(self.target_size) != 2 or min(self.target_size) < 1:
            raise ValueError("target_size must be positive (height, width)")
        self.is_train = is_train
        self.pixel_mean = float(pixel_mean)
        self.pixel_std = float(pixel_std)
        if not np.isfinite(self.pixel_mean) or not np.isfinite(self.pixel_std) or self.pixel_std <= 0:
            raise ValueError("pixel_mean/std must be finite, with pixel_std > 0")
        self.samples: List[Tuple[str, int]] = []

        if not os.path.exists(index_file):
            logger.warning(f"Index file {index_file} does not exist. Initializing empty dataset.")
            return

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        self.samples.append((parts[0].strip(), int(parts[1].strip())))
            logger.info(f"Loaded {len(self.samples)} samples from {index_file}")
        except Exception as e:
            logger.error(f"Failed to read index file {index_file}: {e}", exc_info=True)
            raise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[index]
        full_path = resolve_image_path(self.data_root, rel_path)
        
        if not os.path.exists(full_path):
            logger.error(f"Image file does not exist: {full_path}")
            raise FileNotFoundError(f"Image file does not exist: {full_path}")

        image = cv2.imread(full_path)
        if image is None:
            logger.error(f"Failed to decode image at {full_path}. File is corrupted or unreadable.")
            raise RuntimeError(f"Failed to decode image at {full_path}. File is corrupted or unreadable.")

        # Resize to target 112x112 if needed (Note: This is resizing, not facial landmark alignment)
        if (image.shape[0], image.shape[1]) != self.target_size:
            image = cv2.resize(image, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)

        # Apply augmentation for training
        if self.is_train:
            image = apply_face_augmentation(image, is_train=True)

        # Handle grayscale (2D) images if encountered
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Normalize: (pixel - 127.5) / 128.0
        # Transpose from HWC to CHW
        image_tensor = (image.transpose((2, 0, 1)).astype(np.float32) - self.pixel_mean) / self.pixel_std
        return torch.from_numpy(image_tensor), label


def validate_dataset_files(data_root: str, index_file: str, max_check: Optional[int] = 100) -> Tuple[int, List[str]]:
    """Verify that images referenced in index_file actually exist on disk under data_root.
    Returns (total_checked, missing_paths).
    """
    missing = []
    checked = 0
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file does not exist: {index_file}")

    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            rel_path = parts[0].strip()
            full_path = resolve_image_path(data_root, rel_path)
            if not os.path.exists(full_path):
                missing.append(full_path)
            checked += 1
            if max_check is not None and checked >= max_check:
                break
    return checked, missing


def build_dataloader(
    data_root: str,
    index_file: str,
    batch_size: int = 64,
    num_workers: int = 0,
    is_train: bool = True,
    pin_memory: bool = True,
    target_size: Tuple[int, int] = (112, 112),
    pixel_mean: float = 127.5,
    pixel_std: float = 128.0,
) -> DataLoader:
    """Build a PyTorch DataLoader for face training or evaluation."""
    dataset = FaceImageDataset(
        data_root=data_root,
        index_file=index_file,
        is_train=is_train,
        target_size=target_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=is_train,
    )


__all__ = [
    "VGGFace2ArchiveInspector",
    "apply_face_augmentation",
    "resolve_image_path",
    "FaceImageDataset",
    "validate_dataset_files",
    "build_dataloader",
]

