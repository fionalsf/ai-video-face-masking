"""Bootstrap writer: emit project .py files as UTF-8 (no BOM)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


REVIEW_UI = r'''#!/usr/bin/env python3
"""Streamlit Review UI — continuous event review with Accept/Reject/Skip."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def parse_review_dir() -> str:
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        args = sys.argv[2:]
    else:
        args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--review-dir" and i + 1 < len(args):
            return os.path.abspath(args[i + 1])
    return os.environ.get("REVIEW_DIR", "")


REVIEW_DIR = parse_review_dir()


def load_pending(review_dir: str) -> dict:
    with open(os.path.join(review_dir, "pending_events.json"), encoding="utf-8") as f:
        return json.load(f)


def load_confirmed(review_dir: str) -> dict:
    path = os.path.join(review_dir, "confirmed_events.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "video": "",
        "review_dir": review_dir,
        "started_at": None,
        "updated_at": None,
        "total_review_events": 0,
        "decided_count": 0,
        "summary": {"confirmed_face": 0, "rejected_fp": 0, "skipped": 0},
        "events": [],
    }


def save_confirmed(review_dir: str, data: dict) -> None:
    path = os.path.join(review_dir, "confirmed_events.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_queue(all_events: list, confirmed: dict) -> tuple[list, set[str], dict[str, str]]:
    by_id = {e["event_id"]: e for e in confirmed.get("events", [])}
    decided_final = {
        eid for eid, e in by_id.items()
        if e.get("status") in ("confirmed_face", "rejected_fp")
    }
    status_map = {eid: e.get("status", "") for eid, e in by_id.items()}
    never = [
        e for e in all_events
        if e["event_id"] not in decided_final and status_map.get(e["event_id"]) != "skipped"
    ]
    skipped = [e for e in all_events if status_map.get(e["event_id"]) == "skipped"]
    return never + skipped, decided_final, status_map


def avg_decision_ms(confirmed: dict) -> float:
    times = [e.get("decision_ms", 0) for e in confirmed.get("events", []) if e.get("decision_ms")]
    return sum(times) / len(times) if times else 4000.0


def record_decision(review_dir: str, event: dict, status: str, t0: float) -> None:
    confirmed = load_confirmed(review_dir)
    pending = load_pending(review_dir)
    if confirmed.get("started_at") is None:
        confirmed["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    confirmed["video"] = pending.get("video", "")
    confirmed["review_dir"] = review_dir
    confirmed["total_review_events"] = len(pending.get("events", []))
    entry = {
        "event_id": event["event_id"],
        "track_id": event.get("track_id"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "start_timecode": event.get("start_timecode"),
        "end_timecode": event.get("end_timecode"),
        "peak_confidence": event.get("peak_confidence"),
        "avg_confidence": event.get("avg_confidence"),
        "status": status,
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decision_ms": int((time.time() - t0) * 1000),
        "operator": "local",
        "trajectory": event.get("trajectory", []),
    }
    confirmed["events"] = [
        e for e in confirmed.get("events", []) if e["event_id"] != event["event_id"]
    ]
    confirmed["events"].append(entry)
    confirmed["decided_count"] = len(confirmed["events"])
    summary = {"confirmed_face": 0, "rejected_fp": 0, "skipped": 0}
    for e in confirmed["events"]:
        s = e.get("status", "")
        if s in summary:
            summary[s] += 1
    confirmed["summary"] = summary
    confirmed["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_confirmed(review_dir, confirmed)


def render_keyboard_listener():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.addEventListener("keydown", (e) => {
            if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
            const k = e.key.toLowerCase();
            if (!["a", "r", "s"].includes(k)) return;
            const labels = doc.querySelectorAll("button p, button span, button div");
            for (const el of labels) {
                const t = (el.textContent || "").trim();
                if (k === "a" && t.startsWith("Accept")) { el.closest("button").click(); break; }
                if (k === "r" && t.startsWith("Reject")) { el.closest("button").click(); break; }
                if (k === "s" && t.startsWith("Skip")) { el.closest("button").click(); break; }
            }
        }, { capture: true });
        </script>
        """,
        height=0,
    )


def main():
    st.set_page_config(page_title="Face Event Review", layout="wide")

    if not REVIEW_DIR or not os.path.isdir(REVIEW_DIR):
        st.error('Usage: streamlit run review_ui.py -- --review-dir "output/NAME/review"')
        st.stop()

    pending_path = os.path.join(REVIEW_DIR, "pending_events.json")
    if not os.path.isfile(pending_path):
        st.error(f"Missing pending_events.json in {REVIEW_DIR}")
        st.stop()

    pending = load_pending(REVIEW_DIR)
    confirmed = load_confirmed(REVIEW_DIR)
    all_events = pending.get("events", [])
    queue, decided_final, status_map = build_queue(all_events, confirmed)

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "t0" not in st.session_state:
        st.session_state.t0 = time.time()
    if "jump_id" not in st.session_state:
        st.session_state.jump_id = None

    st.session_state.idx = min(st.session_state.idx, max(0, len(queue) - 1))

    video_name = os.path.basename(pending.get("video", ""))
    decided_count = len(decided_final)
    remaining = len(queue) - st.session_state.idx
    eta_sec = int(remaining * avg_decision_ms(confirmed) / 1000)

    with st.sidebar:
        st.header("Events")
        st.caption(f"{decided_count}/{len(all_events)} decided")
        if remaining > 0:
            st.caption(f"~{eta_sec}s remaining (est.)")
        for i, ev in enumerate(all_events):
            eid = ev["event_id"]
            stt = status_map.get(eid, "pending")
            if eid in decided_final:
                label = f"{eid} [{stt}]"
            elif stt == "skipped":
                label = f"{eid} [skip]"
            elif queue and st.session_state.idx < len(queue) and queue[st.session_state.idx]["event_id"] == eid:
                label = f"> {eid} [current]"
            else:
                label = f"{eid} [pending]"
            if st.button(label, key=f"jump_{eid}", use_container_width=True):
                for j, qe in enumerate(queue):
                    if qe["event_id"] == eid:
                        st.session_state.idx = j
                        st.session_state.t0 = time.time()
                        st.rerun()

    if not queue or st.session_state.idx >= len(queue):
        st.success("Review complete")
        st.json(confirmed.get("summary", {}))
        out_dir = str(Path(REVIEW_DIR).parent)
        st.code(f'python confirm.py --output-dir "{out_dir}"')
        st.stop()

    ev = queue[st.session_state.idx]
    render_keyboard_listener()

    st.title("Face Event Review")
    st.caption(video_name)
    c1, c2, c3 = st.columns(3)
    c1.metric("Progress", f"{decided_count + 1}/{len(all_events)}")
    c2.metric("Peak conf", f"{ev.get('peak_confidence', 0):.3f}")
    c3.metric("ETA", f"~{eta_sec}s")

    tc0 = ev.get("start_timecode") or f"{ev.get('start_time', 0):.1f}s"
    tc1 = ev.get("end_timecode") or f"{ev.get('end_time', 0):.1f}s"
    st.subheader(f"{ev['event_id']}  |  {tc0} ~ {tc1}  |  track {ev.get('track_id', '?')}")
    if ev.get("rule_hints"):
        st.warning("Rule hints: " + ", ".join(ev["rule_hints"]))

    cols = st.columns(3)
    for col, label in zip(cols, ["start", "mid", "end"]):
        rel = ev.get("previews", {}).get(label)
        with col:
            if rel:
                p = os.path.join(REVIEW_DIR, rel.replace("/", os.sep))
                if os.path.isfile(p):
                    st.image(p, caption=label, use_container_width=True)
                else:
                    st.caption(f"{label}: missing")
            else:
                st.caption(f"{label}: n/a")

    def advance(status: str):
        record_decision(REVIEW_DIR, ev, status, st.session_state.t0)
        st.session_state.idx += 1
        st.session_state.t0 = time.time()
        st.rerun()

    st.divider()
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Accept (A)", type="primary", use_container_width=True):
            advance("confirmed_face")
    with b2:
        if st.button("Reject (R)", use_container_width=True):
            advance("rejected_fp")
    with b3:
        if st.button("Skip (S)", use_container_width=True):
            advance("skipped")
    st.caption("Shortcuts: A = Accept, R = Reject, S = Skip")


if __name__ == "__main__":
    main()
'''


PIPELINE_PY = r'''#!/usr/bin/env python3
"""Single-video event pipeline: detect -> track -> tier -> auto mask -> review pack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from audit_log import write_audit_log
from event_builder import TIER_AUTO, TIER_LOW_CONF, TIER_REVIEW, build_events, events_by_tier
from export_events import export_review_pack, write_confirmed_events_template
from low_conf_log import write_low_conf_stats
from render import render_masked_output
from tracker import run_detect_track
from video_meta import get_video_meta, safe_video_stem


def parse_args():
    p = argparse.ArgumentParser(description="Event-based face masking pipeline (single video)")
    p.add_argument("-i", "--input", required=True, help="Input video")
    p.add_argument("-o", "--output-dir", default="output", help="Output root directory")
    p.add_argument("--interval", type=int, default=5, help="Detection stride (frames)")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--model", default="models/face.pt", help="YOLO-face weights")
    p.add_argument("--device", default="0", help="GPU id or cpu")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--event-gap", type=float, default=1.0, help="Event split gap (seconds)")
    p.add_argument("--expand", type=float, default=0.18)
    p.add_argument("--mosaic-block", type=int, default=22)
    p.add_argument("--encoder", default="auto")
    p.add_argument("--skip-render", action="store_true", help="Skip Auto-tier render (debug)")
    return p.parse_args()


def run_pipeline(args) -> int:
    video_path = os.path.abspath(args.input)
    if not os.path.isfile(video_path):
        print(f"[error] Video not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.model):
        print(f"[error] Model not found: {args.model}", file=sys.stderr)
        return 1

    stem = safe_video_stem(video_path)
    out_dir = os.path.join(os.path.abspath(args.output_dir), stem)
    review_dir = os.path.join(out_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    meta = get_video_meta(video_path)
    print(
        f"[info] {stem} | {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps"
        f" | {meta['frames']} frames"
    )

    detections = run_detect_track(
        video_path, args.model, meta,
        device=args.device, conf=args.conf, imgsz=args.imgsz, interval=args.interval,
    )
    events = build_events(
        detections, gap_sec=args.event_gap,
        frame_h=meta["height"], frame_w=meta["width"],
    )
    tiers = events_by_tier(events)
    auto_ev = tiers[TIER_AUTO]
    review_ev = tiers[TIER_REVIEW]
    low_ev = tiers[TIER_LOW_CONF]

    print(f"[Event] total={len(events)} Auto={len(auto_ev)} Review={len(review_ev)} LowConf={len(low_ev)}")

    face_events_path = os.path.join(out_dir, "face_events.json")
    with open(face_events_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "fps": meta["fps"],
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "events": [e.to_dict() for e in events],
        }, f, ensure_ascii=False, indent=2)

    write_low_conf_stats(out_dir, video_path, low_ev)
    write_audit_log(out_dir, video_path, auto_ev)

    draft_path = os.path.join(out_dir, "masked_draft.mp4")
    if not args.skip_render:
        auto_dicts = [e.to_dict() for e in auto_ev]
        if auto_dicts:
            print(f"[Auto] Rendering {len(auto_ev)} events -> masked_draft.mp4")
            render_masked_output(
                video_path, draft_path, auto_dicts, meta,
                expand=args.expand, mosaic_block=args.mosaic_block, encoder=args.encoder,
            )
        else:
            import shutil
            shutil.copy2(video_path, draft_path)
            print("[Auto] No Auto events; copied source as masked_draft.mp4")
    else:
        print("[Auto] --skip-render: skipped")

    if review_ev:
        export_review_pack(video_path, review_dir, review_ev)
        write_confirmed_events_template(review_dir, video_path)
        print(f"[Review] {len(review_ev)} events pending -> {review_dir}")
    else:
        print("[Review] No events need review")

    report = {
        "video": video_path,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "events_total": len(events),
        "auto_masked": len(auto_ev),
        "pending_review": len(review_ev),
        "low_conf_logged": len(low_ev),
        "output_dir": out_dir,
        "masked_draft": draft_path,
        "review_dir": review_dir if review_ev else None,
        "delivery_ready": len(review_ev) == 0,
        "morning_action": "none" if not review_ev else f"streamlit run review_ui.py -- --review-dir {review_dir}",
    }
    report_path = os.path.join(out_dir, "review_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report["delivery_ready"]:
        final_path = os.path.join(out_dir, "final.mp4")
        import shutil
        shutil.copy2(draft_path, final_path)
        print("[done] No Review events; final.mp4 ready")

    print(f"\n[done] Output: {out_dir}")
    print("  review_report.json | face_events.json | audit.log")
    if review_ev:
        print(f'  Morning: streamlit run review_ui.py -- --review-dir "{review_dir}"')
    return 0


def main():
    sys.exit(run_pipeline(parse_args()))


if __name__ == "__main__":
    main()
'''


BATCH_PY = r'''#!/usr/bin/env python3
"""Batch overnight processing: folder queue with resume."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".flv", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser(description="Batch Event pipeline (overnight folder queue)")
    p.add_argument("-i", "--input-dir", required=True)
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()
    extra = [a for a in extra if a != "--"]
    return args, extra


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
        print(f"[error] No videos under {in_root}", file=sys.stderr)
        sys.exit(1)

    pipeline = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.py")
    ok, fail, skip = 0, 0, 0

    for v in videos:
        rel = os.path.relpath(v, in_root)
        stem = os.path.splitext(os.path.basename(v))[0]
        out_sub = os.path.join(out_root, stem)
        done_marker = os.path.join(out_sub, "review_report.json")

        if os.path.isfile(done_marker) and not args.overwrite:
            log_line(log_file, f"SKIP {rel}")
            skip += 1
            continue

        if args.dry_run:
            log_line(log_file, f"DRY-RUN {rel}")
            continue

        cmd = [sys.executable, pipeline, "-i", v, "-o", out_root] + extra
        log_line(log_file, f"START {rel}")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=os.path.dirname(pipeline))
            elapsed = time.time() - t0
            if r.returncode == 0:
                log_line(log_file, f"OK {rel} ({elapsed:.0f}s)")
                ok += 1
            else:
                log_line(log_file, f"FAIL {rel} exit={r.returncode}")
                fail += 1
        except Exception as e:
            log_line(log_file, f"FAIL {rel} {e}")
            fail += 1

    log_line(log_file, f"DONE ok={ok} fail={fail} skip={skip}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
'''


def write(name: str, content: str) -> None:
    path = ROOT / name
    path.write_text(content, encoding="utf-8")
    print(f"wrote {name} ({len(content)} bytes, nulls={path.read_bytes().count(0)})")


if __name__ == "__main__":
    write("review_ui.py", REVIEW_UI)
    write("pipeline.py", PIPELINE_PY)
    write("batch.py", BATCH_PY)
