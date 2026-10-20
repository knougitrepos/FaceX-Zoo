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
    if output.exists():
        raise FileExistsError(f"Preparation output already exists: {output}")
    aligner = create_preprocessor(ROOT, cfg, ds)
    try:
        result = prepare_alignment(source_list, raw_root, output, aligner, args.max_images)
    finally:
        if hasattr(aligner, "close"):
            aligner.close()
    print(json.dumps({k: result[k] for k in ("completed", "total", "processed", "prepared", "failed")}, indent=2))
    return 0 if result["failed"] == 0 and (result["completed"] or args.max_images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
