"""Prepare official-landmark crops or resize-only inputs without face detection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fine-tune"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--dataset", choices=["vggface2", "tinyface", "survface", "scface"], required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output-dir", help="Separate directory for bounded audits")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="CPU preprocessing; no detector models")
    parser.add_argument("--resume", action="store_true", help="Verify and recover incomplete preprocessing without deleting existing PNGs")
    args = parser.parse_args()
    from modules.notebook_workflow import resolve_settings
    from modules.alignment import create_preprocessor, prepare_alignment
    cfg = resolve_settings(ROOT, Path(args.settings).resolve(), require_saved=True)
    ds = cfg["dataset"] if args.dataset == "vggface2" else cfg["finetune_profiles"][args.dataset]
    if not ds.get("enabled", True):
        raise ValueError("Dataset is not enabled/verified")
    if args.dataset != "vggface2":
        from modules.finetune_protocol import verify_protocol
        verify_protocol(ds)
    if args.max_images and not args.output_dir:
        raise ValueError("Use a separate --output-dir for a bounded audit")
    output = Path(args.output_dir or Path(ds["alignment_report"]).parent).resolve()
    source_list, raw_root = Path(ds["train_list"]).resolve(), Path(ds["raw_root"]).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Preparation output already exists: {output}")
    # An already completed dataset is a verification task, not a new alignment
    # task. Honor the same receipt/force policy used by training and export.
    report_path = Path(ds["alignment_report"])
    if (args.resume and args.max_images is None and output == report_path.parent.resolve()
            and cfg.get("data_integrity", {}).get("mode") == "immutable_verified_reuse"
            and report_path.is_file()):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("completed") is True:
            from modules.dataset_verification import bind_aligned_dataset
            selected = dict(cfg, dataset={**cfg["dataset"], **ds})
            checked = bind_aligned_dataset(selected)
            print(json.dumps(checked["data_integrity_verification"], indent=2))
            return 0
    aligner = create_preprocessor(ROOT, cfg, ds)
    try:
        if args.resume:
            from modules.alignment_resume import resume_alignment
            result = resume_alignment(source_list, raw_root, output, aligner, args.max_images)
        else:
            result = prepare_alignment(source_list, raw_root, output, aligner, args.max_images)
    finally:
        if hasattr(aligner, "close"):
            aligner.close()
    print(json.dumps({k: result[k] for k in ("completed", "total", "processed", "prepared", "failed")}, indent=2))
    return 0 if result["failed"] == 0 and (result["completed"] or args.max_images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
