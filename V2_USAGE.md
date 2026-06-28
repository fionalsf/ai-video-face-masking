# V2.0 long-video usage

`pipeline.py` now supports runtime presets for long factory videos.

## Quick commands

```powershell
# Fast preview: renders Auto + Review by default; skips review thumbnails.
python pipeline.py -i "D:/video/input.mp4" -o "D:/mask_output" --device 0 --mode preview --no-review-pack

# Production: denser detection and higher recall.
python pipeline.py -i "D:/video/input.mp4" -o "D:/mask_output" --device 0 --mode production

# Privacy-first: lowest miss rate, more false positives if LowConf is enabled.
python pipeline.py -i "D:/video/input.mp4" -o "D:/mask_output" --device 0 --mode privacy --mask-lowconf

# Reuse tracked_detections.json when only render settings changed.
python pipeline.py -i "D:/video/input.mp4" -o "D:/mask_output" --device 0 --mode preview --reuse-tracks
```

## Presets

| mode | interval | conf | imgsz | default render |
|------|----------|------|-------|----------------|
| `legacy` | 5 | 0.35 | 1280 | Auto |
| `preview` | 5 | 0.25 | 960 | Auto + Review |
| `production` | 2 | 0.25 | 1280 | Auto + Review |
| `privacy` | 1 | 0.20 | 1536 | Auto + Review |

## Useful flags

- `--reuse-tracks`: reuse existing `tracked_detections.json` to avoid rerunning YOLO.
- `--mask-review`: render Review-tier events in `legacy` mode too.
- `--mask-lowconf`: render LowConf-tier events; safer privacy, more false positives.
- `--no-review-pack`: skip review thumbnails for faster long-video preview.
- `--render-extend-frames`: extend first/last trajectory frames to reduce boundary misses.

