# adjust1 validation — 2026-10-19

Code changes are local on branch `adjust1`, based on `365e939`. No commit or push was performed. All changes are under `fine-tune/`.

## Automated checks

Final command:

```text
python -m pytest fine-tune/tests -q --disable-warnings --maxfail=2
168 passed in 92.46s
```

Coverage includes target-only ArcFace/MagFace interior logits, loss, feature/weight gradients and MagFace regularizer; float32/float64 target/non-target +/-1 boundaries; stopping before a non-finite optimizer update; full training lists; incomplete last batches; all four heads resuming immediately before a final partial batch with identical weights/history; aligned-image/source mapping and hash/root mismatch rejection; parent/fine-tuning configuration separation; synthetic small-image training/reuse/export; execution of all five notebooks in plan mode; CUDA memory/throughput checks with 1024D, 8,631 classes and batch 64.

`git diff --check` passed. Git printed LF/CRLF conversion notices, not whitespace errors. These checks are implementation evidence, not a completed scientific training campaign or recognition-performance evaluation.

## Bounded real-image SDK audit

Local SDK defaults: `face_detection_1.0` and `face_alignment_1.0`, CPU, FaceX-Zoo ArcFace 112x112 cropper. Images were the first three entries of each existing training list, not a representative random sample.

- VGGFace2: 3 attempted, 3 aligned, 0 failed.
- TinyFace: 3 attempted, 0 aligned, 3 failed (no face detected).
- SurvFace: 3 attempted, 0 aligned, 3 failed (no face detected).

Detailed immutable audit outputs:

- `fine-tune/data/adjust1/audits/sdk_20261019_170222/vggface2/report.json`
- `fine-tune/data/adjust1/audits/sdk_20261019_170222/tinyface/manifest.jsonl`
- `fine-tune/data/adjust1/audits/sdk_20261019_170222/survface/manifest.jsonl`

All three reports have `completed=false`. The VGGFace2 audit used the existing 95:5 campaign's training list only as a source of bounded diagnostics; it is not the new full-training list and cannot satisfy the new formal preparation contract. An additional one-image CLI audit verified that relative output paths resolve against the calling repository directory:

`fine-tune/data/adjust1/audits/cli_relative_path_check/report.json` — 1 aligned, `completed=false`.

The SDK's old model serialization emits SourceChangeWarning messages under the current PyTorch/torchvision. The cropper emits a skimage deprecation warning. Inference produced the outcomes above. The SDK import-time logging configuration contains a broken handler name; the isolated adapter preserves the caller's logging configuration without editing SDK code.

These small tests do not estimate dataset-wide detection success. TinyFace/SurvFace alignment is **not ready for full training** with the tested policy. Review low-resolution alignment and detection-failure handling before preparing an entire campaign. No automatic resize fallback, image deletion, or speculative attribution to earlier training NaNs was introduced.

## Preservation and remaining execution

- The old campaign report still states `completed=40`, and all 40 referenced `shared_parent_v3` final checkpoints exist.
- Original notebook outputs were copied to `fine-tune/data/legacy_notebook_snapshots/pre_adjust1_365e939/`. All five copies match the original commit after newline normalization. Working notebooks have cleared outputs and use isolated adjust1 settings/paths.
- No new formal checkpoint exists under `fine-tune/checkpoints/adjust1`.
- Full dataset alignment, new parent training, full downstream training, formal export and recognition/FIQA evaluation were not run.
- A preserves the target-boundary singularity; it does not guarantee prevention of every NaN. Fail-before-update behavior remains mandatory.
- Offline preparation currently refuses to overwrite an existing preparation directory. Interrupted preparation requires a new output directory; it does not automatically resume.

Execution instructions: [adjust1.md](../adjust1.md).
