"""
Summary Report Generator for Model Training and Quality Evaluation.
Formats results for analysis and ChatGPT prompt sharing.
"""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("fine_tune.summary")


class ExperimentSummaryReporter:
    """Generates structured experiment summaries in Markdown and JSON formats."""

    @staticmethod
    def generate_chatgpt_summary(
        experiment_meta: Dict[str, Any],
        train_history: Dict[str, Any],
        eval_metrics: Dict[str, Any],
        output_md_path: str,
    ) -> str:
        """Create a Markdown summary suitable for LLM / ChatGPT analysis."""
        model_name = experiment_meta.get("model_name", "Unknown_Model")
        head_type = experiment_meta.get("head_type", "Unknown_Head")
        backbone_dim = experiment_meta.get("feat_dim", 1024)
        parent_hash = experiment_meta.get("parent_hash", "None (Pre-training)")
        export_file = experiment_meta.get("exported_file", "None")

        md = []
        md.append(f"# 📊 Experiment Summary: {model_name}\n")
        md.append("## 1. ⚙️ Model & Configuration Overview")
        md.append(f"- **Backbone**: MobileFaceNet ({backbone_dim}D)")
        md.append(f"- **Angular Margin Head**: {head_type}")
        md.append(f"- **Parent Checkpoint SHA-256**: `{parent_hash}`")
        md.append(f"- **Exported Checkpoint**: `{export_file}`")
        md.append(f"- **Batch Size**: {experiment_meta.get('batch_size', 64)}")
        md.append(f"- **Optimizer**: {experiment_meta.get('optimizer', 'Adam')} (lr={experiment_meta.get('lr', 1e-3)})")
        md.append(f"- **Total Epochs**: {experiment_meta.get('total_epochs', 40)}\n")

        md.append("## 2. 📈 Training Convergence Record")
        if train_history and "epochs" in train_history and train_history["epochs"]:
            if "missing_epochs" in train_history and train_history["missing_epochs"]:
                miss = train_history["missing_epochs"]
                md.append(f"*(참고: Epoch {min(miss)} ~ {max(miss)} 구간은 이전 체크포인트 재개(resume) 실행이며 상세 로그가 이전 파일에 미포함되어 누락 기록되었습니다.)*\n")
            md.append("| Epoch | Loss | Learning Rate |")
            md.append("| :--- | :--- | :--- |")
            for ep, loss, lr in zip(
                train_history.get("epochs", []),
                train_history.get("train_loss", []),
                train_history.get("lr", []),
            ):
                md.append(f"| {ep} | {loss:.4f} | {lr:.6f} |")
        else:
            if train_history and "missing_epochs" in train_history and train_history["missing_epochs"]:
                miss = train_history["missing_epochs"]
                md.append(f"*Epoch {min(miss)} ~ {max(miss)} 구간 재개 완료 (체크포인트 내 상세 로그 미기록)*")
            else:
                md.append("*No epoch logs recorded.*")
        md.append("")

        md.append("## 3. 🔍 특징 Norm 통계 진단 (Feature Norm Diagnostics)")
        if eval_metrics and ("norm_diagnostics" in eval_metrics or "fiqa" in eval_metrics):
            norm_stats = eval_metrics.get("norm_diagnostics", eval_metrics.get("fiqa", {}))
            md.append(f"- **Mean Feature Norm (L2)**: {norm_stats.get('mean_norm', 0.0):.4f}")
            md.append(f"- **Std Feature Norm (L2)**: {norm_stats.get('std_norm', 0.0):.4f}")
            md.append(f"- **Min / Max Norm**: {norm_stats.get('min_norm', 0.0):.4f} / {norm_stats.get('max_norm', 0.0):.4f}")
            md.append(f"- **Median Norm**: {norm_stats.get('median_norm', 0.0):.4f}")
            md.append("*(참고: 본 통계는 L2 Norm 기초 진단이며, 품질 적응적 임계값 연속 보정 및 오픈셋 평가는 ronbun의 Continuous FIQA 파이프라인에서 별도로 측정됩니다.)*")
        
        rank1 = eval_metrics.get("rank1_acc", None) if eval_metrics else None
        if rank1 is not None and isinstance(rank1, (int, float)):
            md.append(f"- **Rank-1 Identification Accuracy**: {rank1:.2f}%")
        else:
            md.append("- **Rank-1 Identification Accuracy**: *미실행 (실제 갤러리/프로브 평가 데이터셋 미연동)*")
        md.append("")

        md.append("## 4. 📚 Academic Paper References & Methodology (User Rule 8)")
        md.append("- **UCFace**: Ahn et al., *Uncertainty-Aware Face Embedding With Contrastive Learning for Open-Set Evaluation*, IEEE TIFS 2024. Method: $L_{total} = L_C + \\lambda L_U$, using feature norm as uncertainty proxy.")
        md.append("- **ArcFace**: Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019. Margin formulation: $\\cos(\\theta + m)$, $m=0.6, s=64.0$.")
        md.append("- **AdaFace**: Kim et al., *AdaFace: Quality Adaptive Margin for Face Recognition*, CVPR 2022. Quality adaptive margins: $g_{ang}=-m\\hat{\\|r\\|}, g_{add}=m+m\\hat{\\|r\\|}$, $m=0.4, s=64.0, h=0.333$.")
        md.append("- **MagFace**: Meng et al., *MagFace: A Universal Representation for Face Recognition and Quality Assessment*, CVPR 2021. Form: $L_{mag} = L_i + \\lambda_g g_{mag}$, $(\\lambda_g=20.0, s=64.0)$.")
        md.append("- **VGGFace2**: Cao et al., *VGGFace2: A dataset for recognising faces across pose and age*, FG 2018. 8,631 training identities, 500 test identities.")
        md.append("")

        md.append("## 5. 💡 ChatGPT Analysis Questions")
        md.append("1. How does the norm distribution compare against typical high-resolution vs low-resolution face embeddings?")
        md.append("2. Does the loss convergence rate show proper margin constraint enforcement without gradient vanishing/explosion?")
        md.append("3. How do the angular margin dynamics align with the theoretical formulations of ArcFace, AdaFace, and MagFace?")
        md.append("")

        content = "\n".join(md)
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Generated ChatGPT experiment summary at {output_md_path}")
        return content
