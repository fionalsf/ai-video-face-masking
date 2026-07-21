# Smart Render Short-Clip Prototype Results

## Result

- Automated 60-second and 5-minute validation: **PASS**.
- Production integration: available as an opt-in mode with automatic fallback.
- Sample: source frames `3300..5099`, `60.060s`, starting at `00:01:50.110`.
- Hybrid plan: `15` total segments (`8` stream-copy, `7` reencoded).
- Source/copy codec: HEVC Main, 1920x1080, 30000/1001 fps.
- Render-segment codec: HEVC Main via `hevc_nvenc` at 12 Mbps.
- Intermediate/concat format: MPEG-TS; final container: MP4 with source AAC copied once.

## Automated Validation

| Check | Result |
|---|---:|
| Expected/output frames | `1800 / 1800` |
| Video duration | `60.060s` |
| Duration delta | `0.000s` |
| Audio/video drift | `0.006667s` |
| Video packet count | `1800` |
| PTS monotonic | pass |
| DTS monotonic | pass |
| Full decode | pass, no FFmpeg errors |

## Visual Inspection

- The mask comparison confirms the target face is mosaicked in the hybrid output.
- A contact sheet covering four representative copy/render boundaries shows continuous scene motion with no black, corrupt, or obviously repeated frames.
- The MP4 decodes successfully even though copied DJI HEVC regions and NVENC HEVC regions carry different in-band parameter sets.

## Important Limitation

The executor now renders masks directly from the source inside GOP-expanded
render regions. It no longer depends on an existing full-render output.

The 5-minute stress sample covered `9000` frames and `31` hybrid segments
(`16` copied, `15` directly rendered). It completed generation plus full decode
validation in `187.035s`, with exact duration, monotonic PTS/DTS, no decode
errors, and `0.030667s` A/V drift (less than one frame).

An audio-free integration smoke test exposed MP4 decode preroll adding ten
frames to a short render input. Direct render segments now enforce their planned
output frame count; the fixed smoke test passes at exactly `300/300` frames.

Production usage remains explicit:

```powershell
python confirm.py --output-dir <reviewed-output> --smart-render
```

- Without `--smart-render`, the existing full renderer is unchanged.
- Smart-render errors automatically fall back to the full renderer.
- `--smart-render-strict` disables fallback for diagnostic runs.
- `--smart-render-keep-work` keeps TS intermediates for debugging.
- Smart rendering currently accepts HEVC sources only; other codecs fall back.

## Artifacts

- Hybrid sample: `smart_render_sample_110s/smart_render_sample.mp4`
- Original comparison sample: `smart_render_sample_110s/source_sample.mp4`
- Machine-readable report: `smart_render_sample_110s/validation_report.json`
- Mask comparison: `smart_render_sample_110s/mask_comparison.jpg`
- Boundary contact sheet: `smart_render_sample_110s/boundary_contact_sheet.png`
- Direct 60-second sample: `smart_render_direct_110s_fixed/smart_render_sample.mp4`
- Direct 5-minute sample: `smart_render_direct_5min/smart_render_sample.mp4`
