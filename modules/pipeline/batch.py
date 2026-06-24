"""Batch overnight folder queue."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".flv", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser(description="Batch pipeline")
    p.add_argument("-i", "--input-dir", required=True)
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()
    return args, [a for a in extra if a != "--"]


def find_videos(root: str) -> list[str]:
    vids = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                vids.append(os.path.join(dirpath, fn))
    return sorted(vids)


def log_line(log_file: str, msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    args, extra = parse_args()
    in_root = os.path.abspath(args.input_dir)
    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)
    log_file = os.path.join(out_root, "batch_log.txt")
    videos = find_videos(in_root)
    if not videos:
        print(f"[error] no videos in {in_root}", file=sys.stderr)
        sys.exit(1)

    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")
    ok = fail = skip = 0
    for v in videos:
        rel = os.path.relpath(v, in_root)
        stem = os.path.splitext(os.path.basename(v))[0]
        marker = os.path.join(out_root, stem, "review_report.json")
        if os.path.isfile(marker) and not args.overwrite:
            log_line(log_file, f"SKIP {rel}")
            skip += 1
            continue
        if args.dry_run:
            log_line(log_file, f"DRY-RUN {rel}")
            continue
        cmd = [sys.executable, runner, "-i", v, "-o", out_root] + extra
        log_line(log_file, f"START {rel}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(runner))))
        if r.returncode == 0:
            log_line(log_file, f"OK {rel} ({time.time()-t0:.0f}s)")
            ok += 1
        else:
            log_line(log_file, f"FAIL {rel} exit={r.returncode}")
            fail += 1
    log_line(log_file, f"DONE ok={ok} fail={fail} skip={skip}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
