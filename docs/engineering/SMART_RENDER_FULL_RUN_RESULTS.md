# Smart Render Full-Video Run

## Outcome

- Source duration: `2225.457s` (`00:37:05.457`).
- Full-render baseline: `1726.506s` (`00:28:46.506`).
- Smart-render output completed in `247.190s` (`00:04:07.190`).
- Speedup: `6.98x`.
- Render-time reduction: `85.7%` (`1479.3s` saved).
- Hybrid plan: `86` segments (`43` copied, `43` directly rendered).
- Output: `final_smart.mp4`, `7.159 GiB`.
- Existing full-render output: `final.mp4`, `3.234 GiB`.

## Validation

- Complete FFmpeg decode: pass, no errors.
- PTS monotonic: pass.
- DTS monotonic: pass.
- Audio/video drift: `0.008866s`, less than one frame.
- Output frames: `66694`.

The source contains `66697` video frames, but its AAC audio ends about `91ms`
before the video. The production final mux uses `-shortest`, so the correct muxed
target is `66694` frames. The existing full-render `final.mp4` also contains
exactly `66694` frames. The initial validator incorrectly compared against the
pre-mux video count; that rule has been corrected.

The one-time full qualification run took about `15m44s` including an additional
complete decode of the 37-minute HEVC output. Normal `confirm.py --smart-render`
now uses frame count, duration, A/V drift, packet timestamp, and stream checks
without decoding every frame. Use `--smart-render-full-validation` when another
complete qualification decode is required.
