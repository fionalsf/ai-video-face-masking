#!/usr/bin/env python3
"""Background contact-sheet prefetcher for mask review UI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from mask_timeline import generate_mask_review_contact_sheet


STATUS_NAME = "preview_prefetch_status.json"
LOCK_NAME = "preview_prefetch.lock"


def parse_args():
    p = argparse.ArgumentParser(description="Prefetch lazy mask-review contact sheets")
    p.add_argument("--review-dir", required=True, help="Review directory containing pending_events.json")
    p.add_argument("--start-index", type=int, default=0, help="Start event index in pending_events.json")
    p.add_argument("--limit", type=int, default=0, help="Max events to generate; 0 means all remaining")
    p.add_argument("--max-samples", type=int, default=4, help="Frames per contact sheet")
    p.add_argument("--expand", type=float, default=0.20, help="Preview mosaic expansion")
    p.add_argument("--stale-lock-sec", type=int, default=21600, help="Ignore lock files older than this")
    return p.parse_args()


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def resolve_timeline_path(review_dir: str, pending_doc: dict) -> str:
    rel = pending_doc.get("timeline") or "mask_timeline.json"
    candidates = []
    if os.path.isabs(str(rel)):
        candidates.append(str(rel))
    else:
        candidates.append(os.path.join(review_dir, str(rel)))
        candidates.append(os.path.join(os.path.dirname(review_dir), str(rel)))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"mask_timeline.json not found for review dir: {review_dir}")


def write_status(review_dir: str, payload: dict) -> None:
    path = os.path.join(review_dir, STATUS_NAME)
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **payload,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def acquire_lock(review_dir: str, stale_lock_sec: int) -> int | None:
    lock_path = os.path.join(review_dir, LOCK_NAME)
    if os.path.exists(lock_path):
        age = time.time() - os.path.getmtime(lock_path)
        if age < stale_lock_sec:
            return None
        try:
            os.remove(lock_path)
        except OSError:
            return None
    try:
        return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


def main() -> int:
    args = parse_args()
    review_dir = os.path.abspath(args.review_dir)
    os.makedirs(review_dir, exist_ok=True)

    lock_fd = acquire_lock(review_dir, args.stale_lock_sec)
    if lock_fd is None:
        return 0

    lock_path = os.path.join(review_dir, LOCK_NAME)
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii", errors="ignore"))
        os.close(lock_fd)

        pending_path = os.path.join(review_dir, "pending_events.json")
        pending = load_json(pending_path)
        timeline_path = resolve_timeline_path(review_dir, pending)
        timeline = load_json(timeline_path)
        events = pending.get("events") or []
        start = max(0, min(int(args.start_index), len(events)))
        selected = events[start:]
        if args.limit and args.limit > 0:
            selected = selected[:args.limit]

        total = len(selected)
        done = 0
        skipped = 0
        write_status(review_dir, {
            "state": "running",
            "pid": os.getpid(),
            "start_index": start,
            "total": total,
            "done": done,
            "skipped": skipped,
        })

        for row in selected:
            event_id = str(row.get("event_id") or row.get("proposal_id") or "")
            if not event_id:
                skipped += 1
                continue
            out_path = os.path.join(review_dir, "event_contact_sheet", f"{event_id}.jpg")
            if os.path.isfile(out_path):
                skipped += 1
            else:
                generate_mask_review_contact_sheet(
                    timeline,
                    review_dir,
                    event_id,
                    expand=args.expand,
                    max_samples=args.max_samples,
                )
                done += 1
            write_status(review_dir, {
                "state": "running",
                "pid": os.getpid(),
                "start_index": start,
                "total": total,
                "done": done,
                "skipped": skipped,
                "current_event_id": event_id,
            })

        write_status(review_dir, {
            "state": "complete",
            "pid": os.getpid(),
            "start_index": start,
            "total": total,
            "done": done,
            "skipped": skipped,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return 0
    except Exception as exc:
        write_status(review_dir, {
            "state": "error",
            "pid": os.getpid(),
            "error": repr(exc),
        })
        print(f"[preview-prefetch] {exc!r}", file=sys.stderr)
        return 1
    finally:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
