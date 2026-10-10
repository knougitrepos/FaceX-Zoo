"""
Evaluation Utilities: Feature Extraction, Continuous FIQA (Norm-based), and Verification.
"""

import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("fine_tune.evaluator")


class FeatureEvaluator:
    """Evaluates face models on representation quality, norms, and recognition accuracy."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def extract_all_features(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract normalized embeddings, feature norms, and labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - embeddings: [N, D]
                - norms: [N, 1]
                - labels: [N]
        """
        self.model.eval()
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        all_embs = []
        all_norms = []
        all_labels = []

        try:
            for images, labels in data_loader:
                images = images.to(self.device)
                normed_feats, norms = raw_model.extract_features(images)

                all_embs.append(normed_feats.cpu().numpy())
                all_norms.append(norms.cpu().numpy())
                all_labels.append(labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels))

            embeddings = np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, 0))
            norms = np.concatenate(all_norms, axis=0) if all_norms else np.empty((0, 0))
            labels = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,))

            logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[-1] if len(embeddings) > 0 else 0}")
            return embeddings, norms, labels
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}", exc_info=True)
            raise

    @staticmethod
    def compute_feature_norm_diagnostics(norms: np.ndarray) -> Dict[str, float]:
        """Compute statistical diagnostics of L2 feature norms.

        Note:
        This function computes summary statistics (mean, std, min, max, median) of feature norms.
        It is a representation diagnostic tool and is distinct from the full Continuous FIQA
        experiment (which involves quality-adaptive continuous threshold calibration and verification
        in ronbun).
        """
        if len(norms) == 0:
            return {"mean_norm": 0.0, "std_norm": 0.0, "min_norm": 0.0, "max_norm": 0.0, "median_norm": 0.0}

        flat_norms = norms.flatten()
        return {
            "mean_norm": float(np.mean(flat_norms)),
            "std_norm": float(np.std(flat_norms)),
            "min_norm": float(np.min(flat_norms)),
            "max_norm": float(np.max(flat_norms)),
            "median_norm": float(np.median(flat_norms)),
        }

    # Backward compatibility alias
    compute_continuous_fiqa = compute_feature_norm_diagnostics

    @staticmethod
    def evaluate_rank1(
        probe_embs: np.ndarray,
        probe_labels: np.ndarray,
        gallery_embs: np.ndarray,
        gallery_labels: np.ndarray,
    ) -> float:
        """Compute Rank-1 Identification Accuracy based on highest cosine similarity."""
        if len(probe_embs) == 0 or len(gallery_embs) == 0:
            return 0.0

        # Ensure L2 normalization for true cosine similarity
        p_norms = np.linalg.norm(probe_embs, axis=1, keepdims=True)
        probe_unit = probe_embs / np.clip(p_norms, 1e-8, None)
        g_norms = np.linalg.norm(gallery_embs, axis=1, keepdims=True)
        gallery_unit = gallery_embs / np.clip(g_norms, 1e-8, None)

        # Cosine similarity matrix: [num_probes, num_gallery]
        sim_matrix = np.dot(probe_unit, gallery_unit.T)
        top1_indices = np.argmax(sim_matrix, axis=1)
        predicted_labels = gallery_labels[top1_indices]

        correct = np.sum(predicted_labels == probe_labels)
        rank1_acc = float(correct / len(probe_labels) * 100.0)
        logger.info(f"Rank-1 Identification Accuracy: {rank1_acc:.2f}% ({correct}/{len(probe_labels)})")
        return rank1_acc
