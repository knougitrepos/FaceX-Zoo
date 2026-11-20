"""Detached parent control runner; delegates optimization to run_campaign."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "fine-tune")]

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(encoding="utf-8")
    from run_campaign import main
    from modules.parent_controls import summarize_parent_control
    request = Path(sys.argv[sys.argv.index("--request") + 1])
    cfg = json.loads(request.read_text(encoding="utf-8"))["settings"]
    main()
    print("Parent control summary:", summarize_parent_control(cfg), flush=True)
