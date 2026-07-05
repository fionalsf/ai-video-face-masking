# V2.0 long-video usage

`pipeline.py` now uses a timeline-first review flow for long factory videos.
The important contract is: review decisions apply to frozen `mask_timeline.json`
proposals, and final render does not recompute tracking or bbox refinement.

## Quick commands

```powershell
# Fast preview: renders Auto + Review by default; skips review thumbnails.
python pipeline.py -i "D:/video/input.mp4" -o "D:/mask_output" --device 0 --mode preview --no-review-pack

# Production: denser detection and higher recall. It exports mask_timeline.json
# and review/pending_events.json for review.
python pipeline.py -i "D:/video/input.mp4" -o "D:/mask_output" --device 0 --mode production --review-only

# Open review UI. Accept/Reject decisions are saved to confirmed_events.json.
streamlit run review_ui.py -- --review-dir "D:/mask_output/video_name/review"

# Render final.mp4 strictly from mask_timeline.json + confirmed_events.json.
python confirm.py --output-dir "D:/mask_output/video_name" --encoder auto

# Rebuild v2 review pack from an existing output without rerunning detection.
python prepare_mask_review.py --output-dir "D:/mask_output/video_name"

# Reuse tracked_detections.json when only render settings changed.
python pipeline.py -i "D:/video/input.mp4" -o "D:/mask_output" --device 0 --mode preview --reuse-tracks
```

## Presets

| mode | interval | conf | imgsz | review unit |
|------|----------|------|-------|----------------|
| `legacy` | 5 | 0.35 | 1280 | frozen mask proposals |
| `preview` | 5 | 0.25 | 960 | frozen mask proposals |
| `production` | 2 | 0.25 | 1280 | frozen mask proposals |
| `privacy` | 1 | 0.20 | 1536 | frozen mask proposals |

## Useful flags

- `--reuse-tracks`: reuse existing `tracked_detections.json` to avoid rerunning YOLO.
- `--mask-review`: render Review-tier events in `legacy` mode too.
- `--mask-lowconf`: render LowConf-tier events; safer privacy, more false positives.
- `--no-review-pack`: skip review thumbnails for faster long-video preview.
- `--render-extend-frames`: extend first/last trajectory frames to reduce boundary misses.
- `--review-only`: build detection/review artifacts without rendering a draft video.

## Review semantics

- `accepted`: proposal is rendered in final.
- `rejected`: proposal is not rendered in final.
- unreviewed: proposal is not rendered in final.
- `auto` proposals are rendered automatically.

The UI no longer visually marks unreviewed proposals as accepted. This avoids
the old failure mode where the button looked accepted but no decision was saved.
