# Fast Pipeline Notes

## Goal

Reduce review-before runtime without weakening the stable V3 production path.

The current stable path remains:

```powershell
python pipeline.py ... --mode production --infer-backend torch
```

## What changed

### Isolated detect worker

Use:

```powershell
python pipeline.py ... --detect-isolated
```

This runs detect+track in `detect_track_worker.py` as a subprocess. The worker writes `tracked_detections.json`, exits, and releases model/GPU/runtime memory before identity stitching, mask timeline building, and review pack generation begin.

This is mainly a stability foundation. It prevents TensorRT/ONNX/PyTorch runtime state from leaking into later CPU/OpenCV stages.

### `fast_review` preset

Use:

```powershell
python pipeline.py ... --mode fast_review --review-only --detect-isolated
```

Compared with `production`:

- Detection interval changes from `2` to `3`.
- Confidence changes from `0.25` to `0.23`.
- Motion compensation windows are slightly widened.
- Extra full-video edge Review scanning is disabled by default.
- Standalone low-confidence promotion scanning is disabled by default.
- Full-video appearance embedding enrichment is disabled by default
  (`--fast-identity`).

This should reduce detector load by roughly one third, but it must be quality-checked on factory videos before production use.

`--fast-identity` keeps identity stitching conservative: tracks are still
segmented and reviewed, but the pipeline does not reopen the whole video to
crop one appearance sample per track. This saves several minutes on long
factory videos and reduces drift caused by bad cross-track merges. Use
`--full-identity` only when comparing against the old behavior.

## Recommended validation command

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python pipeline.py -i "D:\path\to\input.MP4" -o ".\v4_fast_outputs" --mode fast_review --review-only --infer-backend torch --detect-isolated
```

If `tracked_detections.json` already exists, rerun the post-detection stages only:

```powershell
python pipeline.py -i "D:\path\to\input.MP4" -o ".\v4_fast_outputs" --mode fast_review --review-only --infer-backend torch --detect-isolated --reuse-tracks
```

## Current recommendation

Do not make TensorRT the default. The full-video TensorRT FP16 run was slower than torch and stalled later in review-only processing. Keep TensorRT as an explicit experiment only.

## Timing snapshot

Test video: `DJI_20260509094235_0002_D.MP4`
Duration: about 37 minutes
Command: `--mode fast_review --review-only --infer-backend torch --detect-isolated --detect-batch 4`

Measured on 2026-07-11:

- Total review-before runtime: `1215.482s` (`20m15s`)
- Detect + track: `1213.731s`
- Identity stitching: `0.182s`
- Candidate promotion: `0.002s`
- Mask timeline + review pack: `0.892s`

Previous bottlenecks removed:

- Full-video candidate promotion scan: about `1853s` -> near zero
- Full-video appearance embedding enrichment: about `350s` -> near zero

Remaining bottleneck is detector inference/video decode. Batch `4` is currently
the best measured default on the RTX 3060 Laptop GPU; larger batches were not
faster in short benchmarks.
