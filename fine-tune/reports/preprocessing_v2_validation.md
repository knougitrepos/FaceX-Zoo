# Preprocessing v2 validation — 2026-10-20

Implemented locally on `adjust1`. All source/document/notebook changes remain under `fine-tune/`; `git diff --name-only -- . ':!fine-tune'` is empty. No commit or push was performed. The previous `adjust1_validation.md` describes the superseded SDK detection policy and remains as historical evidence.

## Implemented policy

- VGGFace2: official estimated five-point coordinates from `meta/bb_landmark.tar.gz`, exact image-name lookup, original FaceX-Zoo `FaceRecImageCropper` producing 112×112. No new detector or landmark model.
- TinyFace/SurvFace/SCface: full provided face crop resized to 112×112, default OpenCV bicubic, no detection, alignment, additional crop, or padding. This is the user's approved low-resolution experiment policy, not a claim that UCFace's unpublished implementation was recovered.
- The baseline version is `adjust1_preprocessing_v2`. New notebook settings use `data/adjust1/notebook_settings_v2.json`; prepared data use `data/adjust1/preprocessed_v2/`. Older settings and preparation reports are not silently reused.
- Parent preprocessing is checked against the VGGFace2 policy; downstream training/export use their resize-only policy. Metadata binds interpolation, source/target file hashes, ordered labels, archive and cropper hashes, and library versions.
- Every attempted image has a manifest record. Decode/mapping/write failures make preparation incomplete; neither a partial audit nor a list with failed rows is accepted for formal training. Bounded CLI audits with actual failures return a nonzero exit status.

## Real-data audit

Outputs are under `fine-tune/data/adjust1/audits/preprocessing_v2_20261020_003544/`:

- `vggface2_final/report.json`: 32 attempted, 32 prepared, 0 failed. Official source list: 3,141,890 training images; provided landmark CSV: 3,141,890 unique finite-coordinate rows indexed. The earlier `vggface2/` audit in the same directory is superseded by `vggface2_final/`.
- `tinyface/report.json`: 32 attempted, 32 prepared, 0 failed; source training list: 7,804 images.
- `survface/report.json`: 32 attempted, 32 prepared, 0 failed; source training list: 220,888 images.

Each output was decoded and compared pixel-for-pixel with the corresponding original FaceX-Zoo cropper result (VGGFace2), or full-image OpenCV `INTER_CUBIC` resize (TinyFace/SurvFace). Source/output SHA-256 and final policy fingerprints were verified. Raw images were not modified. Temporary SQLite landmark indices were removed on normal completion.

These are the first 32 listed images, not a representative random sample or a complete dataset conversion. Each report correctly has `completed: false` because it is bounded. SCface has synthetic coverage only; no local SCface release was available for a real-image audit.

## Automated checks

Command: `python -m pytest fine-tune/tests -q --disable-warnings --maxfail=2 -o cache_dir=fine-tune/data/pytest_cache`

Result: **177 passed, 9 warnings in 190.53 seconds**. `git diff --check` also passed (Git's LF/CRLF notices are not whitespace failures). The original SDK cropper emits a scikit-image deprecation warning; its source was preserved.

Coverage includes real cropper and resize paths on synthetic data, exact landmark-key lookup, duplicate/missing coordinate rejection, decode-failure accounting, full-crop channel/pixel preservation, source/output tampering, interpolation-policy mismatch, distinct parent/target policies, all five notebook plan modes, and small synthetic parent → four heads → reuse/export workflows. These tests do not constitute a formal full-data training campaign or a face-recognition performance result.

## Remaining execution

Full prepared image sets, formal parent/fine-tuned model training, and recognition/FIQA evaluation have not been run. Begin with notebook 00 to save the new settings; follow `fine-tune/adjust1.md` for full preparation and training. External evaluation code must use the exported preprocessing policy and is outside this change's scope.
