# Smart Render Feasibility

## Scope

This report evaluates keyframe-aware segment rendering only. It does not replace the current production full-video renderer.

## Executive Conclusion

- Recommendation: `continue the independent minimum prototype`; do not integrate it into `render.py` or `confirm.py` yet.
- At the selected 1.0s merge threshold, only `10.39%` of source frames require reencoding after GOP-safe expansion; the theoretical avoided reencode share is `89.61%`.
- This is an encoding-work estimate, not a wall-clock speedup promise. Demux/copy, 86 segment operations, concat/remux, storage throughput, and validation remain overheads.
- Main risk: `43` reencoded regions and `43` copied regions create many codec/timestamp boundaries; correctness must be proven on a short representative clip before any full-video prototype.

## Source Video

- Path: `<input-video>`
- Duration: `2225.457s` (`00:37:05.457`)
- Total frames: `66697`
- FPS: `29.970030`
- CFR/VFR: `CFR-likely` via `mp4-stts`
- CFR evidence: stts_entries=`1` sample_deltas=`[1001]` timescale=`30000`; the MP4 timing table accounts for all `66697` frames.
- Video codec: `hevc` profile=`Main` level=`156` pix_fmt=`yuv420p` time_base=`1/30000`
- Video size: `1920x1080`
- Audio: `aac` sample_rate=`48000` channels=`2`
- Audio segment policy: segment_audio=`none` final_audio=`copy_source_once`

## Coverage

- Coverage basis: the production selection contract (`select_render_entries`) applied to `mask_timeline.json` plus `review/confirmed_events.json`; selected proposals=`206`, selected render frames=`4970`.
- Selection detail: auto=`21`, reviewed=`185`, partial=`7`, rejected=`90`, unreviewed skipped=`260`.
- Raw mask intervals: `86`
- Raw mask duration: `165.832s` (7.45%)
- Merge threshold: `1.000s` (`30` frames)
- Merged mask intervals: `50`
- Merged mask duration: `179.446s` (8.06%)
- Keyframes read: `2224` via `mp4-stss-stts`
- Keyframe-expanded render intervals: `43`
- Keyframe-expanded render duration: `231.231s` (10.39%)

### Merge-threshold Sensitivity

| merge gap | merged intervals | merged coverage | keyframe intervals | actual reencode coverage |
|---:|---:|---:|---:|---:|
| 0.00s | 86 | 7.45% | 43 | 10.39% |
| 0.25s | 72 | 7.52% | 43 | 10.39% |
| 0.50s | 58 | 7.75% | 43 | 10.39% |
| 1.00s | 50 | 8.06% | 43 | 10.39% |
| 2.00s | 40 | 8.71% | 40 | 10.53% |
| 3.00s | 36 | 9.16% | 36 | 10.75% |
| 5.00s | 27 | 10.69% | 27 | 11.83% |

The 0-1s thresholds all produce the same 10.39% GOP-expanded coverage, so the feasibility conclusion is not sensitive to the selected 1s merge threshold. Larger gaps reduce segment count only by reencoding more unmasked frames.

## Keyframe Distribution

- First keyframes: `00:00:00.000, 00:00:01.001, 00:00:02.002, 00:00:03.003, 00:00:04.004, 00:00:05.005, 00:00:06.006, 00:00:07.007`
- Last keyframes: `00:36:58.216, 00:36:59.217, 00:37:00.218, 00:37:01.219, 00:37:02.220, 00:37:03.221, 00:37:04.222, 00:37:05.223`

## Interval Samples

### Raw Mask Intervals

| # | start frame | end frame | start | end | duration |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 18 | 00:00:00.000 | 00:00:00.634 | 0.634s |
| 2 | 51 | 66 | 00:00:01.702 | 00:00:02.236 | 0.534s |
| 3 | 156 | 207 | 00:00:05.205 | 00:00:06.940 | 1.735s |
| 4 | 663 | 675 | 00:00:22.122 | 00:00:22.556 | 0.434s |
| 5 | 3441 | 3459 | 00:01:54.815 | 00:01:55.449 | 0.634s |
| 6 | 3711 | 3723 | 00:02:03.824 | 00:02:04.257 | 0.434s |
| 7 | 3834 | 3897 | 00:02:07.928 | 00:02:10.063 | 2.135s |
| 8 | 4152 | 4170 | 00:02:18.538 | 00:02:19.172 | 0.634s |
| 9 | 4488 | 4548 | 00:02:29.750 | 00:02:31.785 | 2.035s |
| 10 | 4830 | 4881 | 00:02:41.161 | 00:02:42.896 | 1.735s |
| 11 | 4947 | 4989 | 00:02:45.065 | 00:02:46.500 | 1.435s |
| 12 | 5256 | 5313 | 00:02:55.375 | 00:02:57.310 | 1.935s |
| ... | ... | ... | ... | ... | 74 more |

### After Gap Merge

| # | start frame | end frame | start | end | duration |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 18 | 00:00:00.000 | 00:00:00.634 | 0.634s |
| 2 | 51 | 66 | 00:00:01.702 | 00:00:02.236 | 0.534s |
| 3 | 156 | 207 | 00:00:05.205 | 00:00:06.940 | 1.735s |
| 4 | 663 | 675 | 00:00:22.122 | 00:00:22.556 | 0.434s |
| 5 | 3441 | 3459 | 00:01:54.815 | 00:01:55.449 | 0.634s |
| 6 | 3711 | 3723 | 00:02:03.824 | 00:02:04.257 | 0.434s |
| 7 | 3834 | 3897 | 00:02:07.928 | 00:02:10.063 | 2.135s |
| 8 | 4152 | 4170 | 00:02:18.538 | 00:02:19.172 | 0.634s |
| 9 | 4488 | 4548 | 00:02:29.750 | 00:02:31.785 | 2.035s |
| 10 | 4830 | 4881 | 00:02:41.161 | 00:02:42.896 | 1.735s |
| 11 | 4947 | 4989 | 00:02:45.065 | 00:02:46.500 | 1.435s |
| 12 | 5256 | 5313 | 00:02:55.375 | 00:02:57.310 | 1.935s |
| ... | ... | ... | ... | ... | 38 more |

### After Keyframe Expansion

| # | start frame | end frame | start | end | duration |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 89 | 00:00:00.000 | 00:00:03.003 | 3.003s |
| 2 | 150 | 209 | 00:00:05.005 | 00:00:07.007 | 2.002s |
| 3 | 660 | 689 | 00:00:22.022 | 00:00:23.023 | 1.001s |
| 4 | 3420 | 3479 | 00:01:54.114 | 00:01:56.116 | 2.002s |
| 5 | 3690 | 3749 | 00:02:03.123 | 00:02:05.125 | 2.002s |
| 6 | 3810 | 3899 | 00:02:07.127 | 00:02:10.130 | 3.003s |
| 7 | 4140 | 4199 | 00:02:18.138 | 00:02:20.140 | 2.002s |
| 8 | 4470 | 4559 | 00:02:29.149 | 00:02:32.152 | 3.003s |
| 9 | 4830 | 4889 | 00:02:41.161 | 00:02:43.163 | 2.002s |
| 10 | 4920 | 5009 | 00:02:44.164 | 00:02:47.167 | 3.003s |
| 11 | 5250 | 5339 | 00:02:55.175 | 00:02:58.178 | 3.003s |
| 12 | 5400 | 5639 | 00:03:00.180 | 00:03:08.188 | 8.008s |
| ... | ... | ... | ... | ... | 31 more |

## Segment Plan Summary

- Copy segments: `43`
- Reencode segments: `43`
- Planned final concat mode: video-only first, then remux/copy audio once from source.

## Feasibility Assessment

- Expected benefit: `promising` based on keyframe-expanded coverage.
- Stop condition: do not replace production rendering until frame count, duration, PTS/DTS monotonicity, concat boundaries, and player seeking all pass.
- MP4 direct concat is not assumed safe. Prototype should test TS intermediate with h264/hevc bitstream filters and final `-c copy` remux.

## Design Rules For Prototype

1. Copy segments must start at independently decodable keyframes.
2. Mask intervals inside a GOP must expand to surrounding keyframes.
3. Expanded non-mask frames are reencoded without mosaic.
4. Reencoded and copied segments must keep compatible resolution, FPS, pixel format, codec/profile/level, and monotonic timestamps.
5. Segment processing is video-only; audio is copied once from the original at final mux.
6. Existing full-video render remains the fallback.

## Required Validation Before Production

- Output frame count equals source frame count.
- Output duration differs by less than one frame.
- Audio/video duration drift is less than one frame.
- First/last frame PTS and PTS/DTS monotonicity pass.
- Around every concat point, inspect at least 30 frames for dropped/repeated/black/corrupt frames and timestamp jumps.
- ffprobe reports no timestamp or decode errors.
- VLC, Windows player, and browser seeking are manually checked.
- Compare full-render vs smart-render runtime, file size, frame count, duration, and A/V sync.
