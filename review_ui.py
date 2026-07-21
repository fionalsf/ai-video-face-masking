#!/usr/bin/env python3
"""Streamlit Event Review UI v1 — fast local production review tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from event_quality import (
    LOW_QUALITY_WARNING,
    QUALITY_LOW,
    QUALITY_MEDIUM,
    QUALITY_HIGH,
    build_low_quality_reasons,
    load_event_quality,
    quality_by_event_id,
)
from mask_timeline import generate_mask_review_contact_sheet
from review_stats import REPORT_NAME, STATISTICS_NAME, load_review_report, update_review_report

STATUS_ACCEPTED = "accepted"
STATUS_ACCEPTED_RANGES = "accepted_ranges"
STATUS_ACCEPTED_FIRST_HALF = "accepted_first_half"
STATUS_ACCEPTED_SECOND_HALF = "accepted_second_half"
STATUS_REJECTED = "rejected"
STATUS_SKIPPED = "skipped"
ACCEPT_STATUSES = {
    STATUS_ACCEPTED, STATUS_ACCEPTED_RANGES,
    STATUS_ACCEPTED_FIRST_HALF, STATUS_ACCEPTED_SECOND_HALF,
}
FINAL_STATUSES = ACCEPT_STATUSES | {STATUS_REJECTED}

PREVIEW_MAX_HEIGHT_PX = 360
SIDEBAR_WINDOW = 8
PREFETCH_STATUS_NAME = "preview_prefetch_status.json"
PREFETCH_LOCK_NAME = "preview_prefetch.lock"
TIMING_NAME = "review_timing.json"


def parse_output_dir() -> str:
    argv = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == "--" else sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in ("--output-dir", "--review-dir") and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
    return os.environ.get("OUTPUT_DIR") or os.environ.get("REVIEW_DIR") or ""


OUTPUT_DIR = parse_output_dir()
CONFIRMED_NAME = "confirmed_events.json"


def confirmed_path(output_dir: str) -> str:
    return os.path.join(output_dir, CONFIRMED_NAME)


def decision_status(decision) -> str | None:
    if isinstance(decision, dict):
        return str(decision.get("status") or "") or None
    return str(decision) if decision else None


def decision_ranges(decision) -> list[list[float]]:
    if not isinstance(decision, dict) or decision_status(decision) != STATUS_ACCEPTED_RANGES:
        return []
    ranges = []
    for item in decision.get("ranges") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 and float(item[1]) > float(item[0]):
            ranges.append([round(float(item[0]), 3), round(float(item[1]), 3)])
    return ranges


def merge_time_ranges(ranges: list[list[float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(ranges, key=lambda item: float(item[0])):
        start, end = round(float(start), 3), round(float(end), 3)
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + 0.001:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def load_decisions(output_dir: str) -> dict[str, str | dict]:
    path = confirmed_path(output_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data and not isinstance(next(iter(data.values())), dict):
        meta_keys = {
            "events", "summary", "video", "review_dir", "started_at", "updated_at",
            "total_review_events", "decided_count", "schema", "decision_unit",
        }
        return {
            str(k): v if isinstance(v, dict) else str(v)
            for k, v in data.items()
            if str(k) not in meta_keys
            and (str(k).startswith("bevt_") or str(k).startswith("evt_") or str(k).startswith("mask_"))
        }
    if isinstance(data, dict) and "events" in data:
        out: dict[str, str] = {}
        for e in data["events"]:
            stt = e.get("status", "")
            if stt == "confirmed_face":
                out[e["event_id"]] = STATUS_ACCEPTED
            elif stt == "rejected_fp":
                out[e["event_id"]] = STATUS_REJECTED
            elif stt == "skipped":
                out[e["event_id"]] = STATUS_SKIPPED
        return out
    return {}


def save_decisions(
    output_dir: str,
    decisions: dict[str, str | dict],
    events: list[dict] | None = None,
    video: str = "",
    update_report: bool = False,
) -> None:
    meta = {
        "schema": "mask_review_decisions.v2",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decision_unit": "mask_timeline_proposal",
        "total_review_events": len(events) if events is not None else None,
        "decided_count": len(decisions),
    }
    payload = {
        **{k: v for k, v in meta.items() if v is not None},
        **dict(sorted(decisions.items())),
    }
    with open(confirmed_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if update_report and events is not None:
        update_review_report(output_dir, video, events, decisions)


def timing_path(output_dir: str) -> str:
    return os.path.join(output_dir, TIMING_NAME)


def load_review_timing(output_dir: str) -> dict:
    path = timing_path(output_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def start_review_timing(output_dir: str, decided_count: int, total: int) -> dict:
    now = time.time()
    timing = {
        "schema": "review_timing.v1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at_epoch": now,
        "starting_decided_count": int(decided_count),
        "current_decided_count": int(decided_count),
        "total_review_events": int(total),
        "elapsed_sec": 0.0,
        "finished_at": None,
    }
    with open(timing_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)
    return timing


def update_review_timing(output_dir: str, decided_count: int, total: int) -> dict:
    timing = load_review_timing(output_dir)
    if not timing or timing.get("finished_at"):
        return timing
    elapsed = max(0.0, time.time() - float(timing.get("started_at_epoch") or time.time()))
    timing["current_decided_count"] = int(decided_count)
    timing["total_review_events"] = int(total)
    timing["elapsed_sec"] = round(elapsed, 3)
    if decided_count >= total:
        timing["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(timing_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)
    return timing


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_events(output_dir: str) -> tuple[list[dict], str]:
    pending_path = os.path.join(output_dir, "pending_events.json")
    if os.path.isfile(pending_path):
        with open(pending_path, encoding="utf-8") as f:
            pending_doc = json.load(f)
        video = pending_doc.get("video", "")
        events: list[dict] = []
        for ev in pending_doc.get("events") or []:
            events.append({
                "event_id": ev["event_id"],
                "source_event_id": ev.get("source_event_id"),
                "source_tier": ev.get("source_tier"),
                "track_id": ev.get("track_id"),
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "start_timecode": ev.get("start_timecode"),
                "end_timecode": ev.get("end_timecode"),
                "duration_sec": ev.get("duration_sec"),
                "frame_count": ev.get("frame_count") or ev.get("detection_count"),
                "peak_confidence": ev.get("peak_confidence"),
                "avg_confidence": ev.get("avg_confidence"),
                "previews": ev.get("previews"),
                "contact_sheet": ev.get("contact_sheet"),
                "requires_explicit_accept": ev.get("requires_explicit_accept"),
            })
        return events, video

    behavior_path = os.path.join(output_dir, "behavior_events.json")
    if os.path.isfile(behavior_path):
        with open(behavior_path, encoding="utf-8") as f:
            behavior_doc = json.load(f)
        video = behavior_doc.get("video", "")
        events: list[dict] = []
        for ev in behavior_doc.get("events") or []:
            events.append({
                "event_id": ev["event_id"],
                "identity_id": ev.get("identity_id"),
                "track_id": ev.get("primary_track_id") or (ev.get("source_track_ids") or [0])[0],
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "start_timecode": ev.get("start_timecode"),
                "end_timecode": ev.get("end_timecode"),
                "duration_sec": ev.get("duration_sec"),
                "frame_count": ev.get("detection_count"),
                "peak_confidence": ev.get("peak_confidence"),
                "avg_confidence": ev.get("avg_confidence"),
                "event_quality_score": ev.get("event_quality_score"),
                "source_track_ids": ev.get("source_track_ids"),
                "cross_track_merge": ev.get("cross_track_merge", False),
            })
        return events, video

    final_path = os.path.join(output_dir, "final_events.json")
    if os.path.isfile(final_path):
        with open(final_path, encoding="utf-8") as f:
            final_doc = json.load(f)
        video = final_doc.get("video", "")
        events: list[dict] = []
        for ev in final_doc.get("events") or []:
            merged = {
                "event_id": ev["event_id"],
                "track_id": ev["track_id"],
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "start_timecode": ev.get("start_timecode"),
                "end_timecode": ev.get("end_timecode"),
                "duration_sec": ev.get("duration_sec"),
                "frame_count": ev.get("detection_count"),
                "peak_confidence": ev.get("peak_confidence"),
                "avg_confidence": ev.get("avg_confidence"),
                "event_quality_score": ev.get("event_quality_score"),
                "source_event_ids": ev.get("source_event_ids"),
                "behavior_merged": ev.get("behavior_merged", False),
            }
            events.append(merged)
        return events, video

    summary_path = os.path.join(output_dir, "event_summary.json")
    preview_path = os.path.join(output_dir, "event_preview.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing event_summary.json in {output_dir}")

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    preview_by_id: dict[str, dict] = {}
    video = summary.get("video", "")
    if os.path.isfile(preview_path):
        with open(preview_path, encoding="utf-8") as f:
            preview = json.load(f)
        video = preview.get("video") or video
        preview_by_id = {e["event_id"]: e for e in preview.get("events", [])}

    events: list[dict] = []
    for ev in summary.get("events", []):
        merged = dict(ev)
        extra = preview_by_id.get(ev["event_id"], {})
        if extra.get("avg_confidence") is not None:
            merged["avg_confidence"] = extra["avg_confidence"]
        merged.setdefault("start_timecode", extra.get("start_timecode"))
        merged.setdefault("end_timecode", extra.get("end_timecode"))
        events.append(merged)
    return events, video


def count_stats(events: list[dict], decisions: dict[str, str | dict]) -> dict[str, int | float]:
    total = len(events)
    accepted = sum(1 for e in events if decision_status(decisions.get(e["event_id"])) in ACCEPT_STATUSES)
    rejected = sum(1 for e in events if decision_status(decisions.get(e["event_id"])) == STATUS_REJECTED)
    skipped = sum(1 for e in events if decision_status(decisions.get(e["event_id"])) == STATUS_SKIPPED)
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "skipped": skipped,
        "remaining": total - accepted - rejected,
        "accept_pct": round(100.0 * accepted / total, 1) if total else 0.0,
        "reject_pct": round(100.0 * rejected / total, 1) if total else 0.0,
        "skip_pct": round(100.0 * skipped / total, 1) if total else 0.0,
    }


def next_pending_index(events: list[dict], decisions: dict[str, str | dict], start: int) -> int | None:
    for j in range(start, len(events)):
        if decision_status(decisions.get(events[j]["event_id"])) not in FINAL_STATUSES:
            return j
    for j in range(0, start):
        if decision_status(decisions.get(events[j]["event_id"])) not in FINAL_STATUSES:
            return j
    return None


def prev_pending_index(events: list[dict], decisions: dict[str, str | dict], start: int) -> int | None:
    for j in range(start, -1, -1):
        if decision_status(decisions.get(events[j]["event_id"])) not in FINAL_STATUSES:
            return j
    for j in range(len(events) - 1, start, -1):
        if decision_status(decisions.get(events[j]["event_id"])) not in FINAL_STATUSES:
            return j
    return None


def media_paths(output_dir: str, event_id: str, ev: dict | None = None) -> tuple[str | None, str | None]:
    if ev and ev.get("previews"):
        for key in ("mid", "start", "end"):
            rel = ev["previews"].get(key)
            if rel:
                path = os.path.join(output_dir, rel.replace("/", os.sep))
                if os.path.isfile(path):
                    return path, None
    if ev and ev.get("contact_sheet"):
        path = os.path.join(output_dir, str(ev["contact_sheet"]).replace("/", os.sep))
        if os.path.isfile(path):
            return path, None

    previews_dir = os.path.join(output_dir, "previews")
    for name in (f"{event_id}_mid.jpg", f"{event_id}_start.jpg", f"{event_id}_end.jpg"):
        path = os.path.join(previews_dir, name)
        if os.path.isfile(path):
            return path, None

    sheet = os.path.join(output_dir, "event_contact_sheet", f"{event_id}.jpg")
    gif = os.path.join(output_dir, "event_gifs", f"{event_id}.gif")
    contact = sheet if os.path.isfile(sheet) else None
    if contact is None:
        contact = ensure_lazy_contact_sheet(output_dir, event_id)
    if contact is None:
        preview = os.path.join(output_dir, "event_previews", f"{event_id}.jpg")
        contact = preview if os.path.isfile(preview) else None
    gif_path = gif if os.path.isfile(gif) else None
    return contact, gif_path


def _pending_doc_path(output_dir: str) -> str:
    return os.path.join(output_dir, "pending_events.json")


@st.cache_data(show_spinner=False)
def _load_json_cached(path: str, mtime: float) -> dict:
    del mtime
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _load_pending_doc(output_dir: str) -> dict:
    path = _pending_doc_path(output_dir)
    if not os.path.isfile(path):
        return {}
    return _load_json_cached(path, os.path.getmtime(path))


def _resolve_timeline_path(output_dir: str) -> str | None:
    pending = _load_pending_doc(output_dir)
    rel = pending.get("timeline") or "mask_timeline.json"
    candidates = []
    if os.path.isabs(str(rel)):
        candidates.append(str(rel))
    else:
        candidates.append(os.path.join(output_dir, str(rel)))
        candidates.append(os.path.join(os.path.dirname(output_dir), str(rel)))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_review_timeline(output_dir: str) -> dict | None:
    path = _resolve_timeline_path(output_dir)
    if not path:
        return None
    return _load_json_cached(path, os.path.getmtime(path))


def ensure_lazy_contact_sheet(output_dir: str, event_id: str) -> str | None:
    sheet = os.path.join(output_dir, "event_contact_sheet", f"{event_id}.jpg")
    if os.path.isfile(sheet):
        return sheet
    timeline = _load_review_timeline(output_dir)
    if not timeline:
        return None
    with st.spinner("Generating preview..."):
        return generate_mask_review_contact_sheet(timeline, output_dir, event_id)


def _prefetch_status_path(output_dir: str) -> str:
    return os.path.join(output_dir, PREFETCH_STATUS_NAME)


def load_prefetch_status(output_dir: str) -> dict:
    path = _prefetch_status_path(output_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _prefetch_lock_active(output_dir: str, stale_sec: int = 21600) -> bool:
    path = os.path.join(output_dir, PREFETCH_LOCK_NAME)
    if not os.path.isfile(path):
        return False
    try:
        return (time.time() - os.path.getmtime(path)) < stale_sec
    except OSError:
        return True


def start_background_preview_prefetch(output_dir: str, start_index: int) -> None:
    session_key = f"preview_prefetch_started::{output_dir}"
    if st.session_state.get(session_key):
        return
    if _prefetch_lock_active(output_dir):
        st.session_state[session_key] = True
        return

    pending = _load_pending_doc(output_dir)
    events = pending.get("events") or []
    if not events:
        return
    status = load_prefetch_status(output_dir)
    if status.get("state") == "complete":
        sheet_dir = os.path.join(output_dir, "event_contact_sheet")
        generated = len([name for name in os.listdir(sheet_dir)]) if os.path.isdir(sheet_dir) else 0
        if generated >= len(events):
            st.session_state[session_key] = True
            return

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefetch_review_previews.py")
    if not os.path.isfile(script):
        return
    cmd = [
        sys.executable,
        script,
        "--review-dir",
        output_dir,
        "--start-index",
        str(max(0, int(start_index))),
    ]
    try:
        kwargs = {
            "cwd": os.path.dirname(script),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(cmd, **kwargs)
        st.session_state[session_key] = True
    except OSError:
        return


def render_prefetch_status(output_dir: str) -> None:
    status = load_prefetch_status(output_dir)
    if not status:
        st.caption("Preview prefetch: starting")
        return
    state = status.get("state", "unknown")
    total = int(status.get("total") or 0)
    done = int(status.get("done") or 0)
    skipped = int(status.get("skipped") or 0)
    completed = min(total, done + skipped)
    if total > 0:
        st.caption(f"Preview prefetch: {state} {completed}/{total}")
    else:
        st.caption(f"Preview prefetch: {state}")


def status_icon(status) -> str:
    status = decision_status(status)
    return {
        STATUS_ACCEPTED: "✅",
        STATUS_ACCEPTED_RANGES: "✅",
        STATUS_ACCEPTED_FIRST_HALF: "◐",
        STATUS_ACCEPTED_SECOND_HALF: "◑",
        STATUS_REJECTED: "❌",
        STATUS_SKIPPED: "⏭",
    }.get(status or "", "⬜")


def render_keyboard_listener():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.addEventListener("keydown", (e) => {
            if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
            const k = e.key.toLowerCase();
            const targets = {
                a: "Accept", r: "Reject", s: "Skip",
                e: "Local adjust",
                arrowleft: "Prev", arrowright: "Next",
            };
            const needle = targets[k];
            if (!needle) return;
            for (const btn of doc.querySelectorAll("button")) {
                const t = (btn.innerText || "").trim();
                if (t.includes(needle)) { btn.click(); e.preventDefault(); break; }
            }
        }, { capture: true });
        </script>
        """,
        height=0,
    )


def inject_compact_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .stButton button {
            text-align: left; padding: 0.3rem 0.55rem; font-size: 0.8rem; line-height: 1.25;
        }
        [data-testid="stMetric"] { background:#161616; padding:4px 8px; border-radius:6px; }
        [data-testid="stMetric"] label { font-size:0.75rem; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size:1.1rem; }
        .quality-high { color:#22c55e; font-weight:700; }
        .quality-medium { color:#eab308; font-weight:700; }
        .quality-low { color:#ef4444; font-weight:700; }
        .quality-badge {
            display:inline-block; padding:0.2rem 0.65rem; border-radius:0.35rem;
            font-weight:700; font-size:0.9rem;
        }
        .quality-badge-high { background:#052e16; color:#22c55e; border:1px solid #22c55e; }
        .quality-badge-medium { background:#422006; color:#eab308; border:1px solid #eab308; }
        .quality-badge-low { background:#450a0a; color:#ef4444; border:1px solid #ef4444; }
        div[data-testid="stAlert"] { padding:0.35rem 0.65rem; margin:0.15rem 0; }
        [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap:0.5rem; }
        [data-testid="stMainBlockContainer"] [data-testid="stImage"] img {
            max-height: """ + str(PREVIEW_MAX_HEIGHT_PX) + """px;
            width: 100% !important;
            max-width: 100% !important;
            object-fit: contain;
            background: #111827;
            border-radius: 10px;
            border: 1px solid #374151;
            padding: 6px;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {
            font-size: 0.8rem;
            margin-top: 0.25rem;
            color: #9ca3af;
        }
        .event-meta-bar {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 0.65rem 0.9rem;
            margin: 0.15rem 0 0.35rem;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .event-meta-bar .meta-line { color: #e5e7eb; margin: 0; }
        .event-meta-bar .meta-sub { color: #9ca3af; font-size: 0.82rem; margin: 0.2rem 0 0; }
        #review_accept button[kind="primary"],
        #review_accept button[data-testid="stBaseButton-primary"] {
            background-color: #16a34a !important;
            border-color: #16a34a !important;
        }
        #review_reject button[kind="primary"],
        #review_reject button[data-testid="stBaseButton-primary"] {
            background-color: #dc2626 !important;
            border-color: #dc2626 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def quality_icon(quality: str | None) -> str:
    return {
        QUALITY_HIGH: "🟢",
        QUALITY_MEDIUM: "🟡",
        QUALITY_LOW: "🔴",
    }.get(quality or "", "⚪")


def quality_css_class(quality: str | None) -> str:
    return {
        QUALITY_HIGH: "quality-high",
        QUALITY_MEDIUM: "quality-medium",
        QUALITY_LOW: "quality-low",
    }.get(quality or "", "")


def quality_badge_html(quality: str | None) -> str:
    label = quality or "—"
    badge_cls = {
        QUALITY_HIGH: "quality-badge-high",
        QUALITY_MEDIUM: "quality-badge-medium",
        QUALITY_LOW: "quality-badge-low",
    }.get(quality or "", "quality-badge")
    return f'<span class="quality-badge {badge_cls}">{quality_icon(quality)} {label}</span>'


def low_risk_reasons(quality: dict) -> list[str]:
    reasons = quality.get("risk_reasons")
    if reasons:
        return reasons
    if quality.get("quality") != QUALITY_LOW:
        return []
    return build_low_quality_reasons(
        detection_count=int(quality.get("detection_count") or 0),
        duration_sec=float(quality.get("duration_sec") or 0),
        max_detection_gap_sec=float(quality.get("max_detection_gap_sec") or 0),
        interpolation_ratio=float(quality.get("interpolation_ratio") or 0),
    )


def merge_event_quality(events: list[dict], output_dir: str) -> dict[str, dict]:
    embedded = {
        e["event_id"]: e["event_quality_score"]
        for e in events
        if e.get("event_quality_score")
    }
    if embedded:
        return embedded
    return quality_by_event_id(load_event_quality(output_dir))


def render_event_metadata_html(ev: dict, quality: dict | None) -> str:
    avg = ev.get("avg_confidence")
    avg_txt = f"{avg:.4f}" if avg is not None else "—"
    q_html = quality_badge_html(quality.get("quality")) if quality else ""
    hint = ""
    if quality and quality.get("quality") == QUALITY_LOW:
        reasons = low_risk_reasons(quality)
        extra = " · ".join(reasons) if reasons else (quality.get("warning") or LOW_QUALITY_WARNING)
        hint = f'<p class="meta-sub quality-low">🔴 建议 Reject — {extra}</p>'
    elif quality and quality.get("quality") == QUALITY_MEDIUM:
        hint = f'<p class="meta-sub quality-medium">🟡 {QUALITY_MEDIUM}</p>'
    elif quality and quality.get("quality") == QUALITY_HIGH:
        hint = f'<p class="meta-sub quality-high">🟢 {QUALITY_HIGH}</p>'
    return (
        '<div class="event-meta-bar">'
        f'<p class="meta-line"><b>{ev["event_id"]}</b> · Track {ev.get("track_id", "—")} · '
        f'{ev.get("duration_sec", 0):.2f}s · {ev.get("frame_count", "—")} frames · '
        f'Peak {ev.get("peak_confidence", 0):.3f} · Avg {avg_txt} · {q_html}</p>'
        f'<p class="meta-sub">{ev.get("start_timecode") or ev.get("start_time")} → '
        f'{ev.get("end_timecode") or ev.get("end_time")}</p>'
        f"{hint}"
        "</div>"
    )


def render_event_panel(
    output_dir: str,
    ev: dict,
    quality: dict | None = None,
) -> None:
    eid = ev["event_id"]
    contact, gif_path = media_paths(output_dir, eid, ev)

    st.markdown(render_event_metadata_html(ev, quality), unsafe_allow_html=True)

    if gif_path and contact:
        gif_col, sheet_col = st.columns(2, gap="medium")
        with gif_col:
            st.image(gif_path, caption=f"{eid} — GIF preview", use_container_width=True)
        with sheet_col:
            st.image(contact, caption=f"{eid} — contact sheet", use_container_width=True)
    elif gif_path:
        st.image(gif_path, caption=f"{eid} — GIF preview", use_container_width=True)
    elif contact:
        st.image(contact, caption=f"{eid} — contact sheet", use_container_width=True)
    else:
        st.warning("No preview media found.")

    if quality:
        with st.expander("Quality metrics", expanded=False):
            st.caption(
                f"检测 {quality.get('detection_count', '—')} · "
                f"跨度 {quality.get('duration_sec', 0):.2f}s · "
                f"密度 {quality.get('detection_density', 0):.3f}/s · "
                f"插值比 {quality.get('interpolation_ratio', 0):.2%}"
            )
            st.caption(
                f"最大间隔 {quality.get('max_detection_gap_sec', 0):.2f}s · "
                f"平均间隔 {quality.get('avg_detection_gap_sec', 0):.2f}s · "
                f"最长无检测 {quality.get('longest_interpolation_gap_sec', 0):.2f}s"
            )


def render_decision_banner(current) -> None:
    current_status = decision_status(current)
    if not current_status:
        st.markdown(
            '<p style="margin:0.1rem 0 0.25rem;font-size:0.88rem;color:#f59e0b;">'
            "Decision: <b>unreviewed</b> - click Accept or Reject to save</p>",
            unsafe_allow_html=True,
        )
        return
    colors = {
        STATUS_ACCEPTED: "#22c55e",
        STATUS_ACCEPTED_RANGES: "#22c55e",
        STATUS_ACCEPTED_FIRST_HALF: "#22c55e",
        STATUS_ACCEPTED_SECOND_HALF: "#22c55e",
        STATUS_REJECTED: "#ef4444",
        STATUS_SKIPPED: "#38bdf8",
    }
    color = colors.get(current_status, "#94a3b8")
    label = current_status
    if current_status == STATUS_ACCEPTED_RANGES:
        label = f"accepted_ranges ({len(decision_ranges(current))})"
    st.markdown(
        f'<p style="margin:0.1rem 0 0.25rem;font-size:0.88rem;color:{color};">'
        f"Decision: <b>{label}</b> — click buttons above to change</p>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Event Review", layout="wide", initial_sidebar_state="expanded")
    inject_compact_css()

    if not OUTPUT_DIR or not os.path.isdir(OUTPUT_DIR):
        st.error('Usage: streamlit run review_ui.py -- --output-dir "output/detection/test456"')
        st.stop()

    try:
        all_events, video_path = load_events(OUTPUT_DIR)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if not all_events:
        st.warning("No events in event_summary.json")
        st.stop()

    quality_map = merge_event_quality(all_events, OUTPUT_DIR)

    if "decisions" not in st.session_state:
        st.session_state.decisions = load_decisions(OUTPUT_DIR)
    if "view_idx" not in st.session_state:
        st.session_state.view_idx = 0
    if "auto_advance" not in st.session_state:
        st.session_state.auto_advance = True
    if "report_bootstrapped" not in st.session_state:
        st.session_state.report_bootstrapped = True

    decisions = st.session_state.decisions
    stats = count_stats(all_events, decisions)
    st.session_state.view_idx = min(st.session_state.view_idx, len(all_events) - 1)
    start_background_preview_prefetch(OUTPUT_DIR, st.session_state.view_idx + 1)

    video_name = os.path.basename(video_path) if video_path else Path(OUTPUT_DIR).name

    st.markdown(f"### Event Review · `{video_name}`")
    if stats["remaining"] == 0:
        st.caption("All events decided — sidebar click to review or change.")

    with st.expander("Progress & analysis", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", stats["total"])
        c2.metric("Accepted", stats["accepted"])
        c3.metric("Rejected", stats["rejected"])
        c4.metric("Skipped", stats["skipped"])
        c5.metric("Remaining", stats["remaining"])
        st.caption(
            f"Accept {stats['accept_pct']}% · Reject {stats['reject_pct']}% · "
            f"Skip {stats['skip_pct']}% · {'done' if not stats['remaining'] else 'need A/R'}"
        )
        timing = load_review_timing(OUTPUT_DIR)
        final_decided = int(stats["total"] - stats["remaining"])
        if timing:
            elapsed = float(timing.get("elapsed_sec") or 0.0)
            if not timing.get("finished_at"):
                elapsed = max(
                    elapsed,
                    time.time() - float(timing.get("started_at_epoch") or time.time()),
                )
            reviewed_in_session = max(
                0,
                final_decided - int(timing.get("starting_decided_count") or 0),
            )
            avg_sec = elapsed / reviewed_in_session if reviewed_in_session else 0.0
            st.info(
                f"Timing: {format_elapsed(elapsed)} · session decisions {reviewed_in_session} · "
                f"average {avg_sec:.1f}s/event"
            )
        if st.button("⏱ Start timing from current progress", use_container_width=False):
            start_review_timing(OUTPUT_DIR, final_decided, len(all_events))
            st.rerun()

        report = load_review_report(OUTPUT_DIR)
        if report:
            a = report.get("analysis", {})
            st.caption(
                f"Accepted avg peak {a.get('accepted_avg_peak_confidence', '—')} · "
                f"Rejected avg peak {a.get('rejected_avg_peak_confidence', '—')} · "
                f"Accepted avg dur {a.get('accepted_avg_duration_sec', '—')}s"
            )
            if report.get("review_finished_at"):
                st.caption(f"Review finished at {report['review_finished_at']}")
        if st.button("Refresh saved report", use_container_width=False):
            update_review_report(OUTPUT_DIR, video_path, all_events, decisions)
            st.rerun()

    with st.sidebar:
        st.header("Event List")
        st.caption("Fast window view; use jump for far events")
        render_prefetch_status(OUTPUT_DIR)
        st.session_state.auto_advance = st.checkbox(
            "Auto-advance after decision",
            value=st.session_state.auto_advance,
            help="When off, stay on current event after Accept/Reject/Skip (useful when correcting).",
        )
        nav_a, nav_b = st.columns(2)
        with nav_a:
            if st.button("Prev pending", use_container_width=True):
                target = prev_pending_index(all_events, decisions, st.session_state.view_idx - 1)
                if target is not None:
                    st.session_state.view_idx = target
                    st.rerun()
        with nav_b:
            if st.button("Next pending", use_container_width=True):
                target = next_pending_index(all_events, decisions, st.session_state.view_idx + 1)
                if target is not None:
                    st.session_state.view_idx = target
                    st.rerun()

        jump_value = st.number_input(
            "Jump to #",
            min_value=1,
            max_value=len(all_events),
            value=st.session_state.view_idx + 1,
            step=1,
        )
        if st.button("Go", use_container_width=True):
            st.session_state.view_idx = int(jump_value) - 1
            st.rerun()

        start_i = max(0, st.session_state.view_idx - SIDEBAR_WINDOW)
        end_i = min(len(all_events), st.session_state.view_idx + SIDEBAR_WINDOW + 1)
        st.caption(f"Showing {start_i + 1}-{end_i} of {len(all_events)}")
        for i in range(start_i, end_i):
            ev = all_events[i]
            eid = ev["event_id"]
            stt = decisions.get(eid)
            is_current = st.session_state.view_idx == i
            mark = "▶ " if is_current else ""
            q = quality_map.get(eid, {})
            q_mark = quality_icon(q.get("quality"))
            label = (
                f"{mark}{status_icon(stt)} {q_mark} {eid} | T{ev.get('track_id', '?')}\n"
                f"{ev.get('duration_sec', 0):.2f}s | peak {ev.get('peak_confidence', 0):.2f}"
            )
            if st.button(label, key=f"jump_{eid}", use_container_width=True):
                st.session_state.view_idx = i
                st.rerun()

    ev = all_events[st.session_state.view_idx]
    eid = ev["event_id"]
    ev_quality = quality_map.get(eid)
    render_keyboard_listener()

    nav_l, nav_m, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("◀ Prev", disabled=st.session_state.view_idx <= 0):
            st.session_state.view_idx -= 1
            st.rerun()
    with nav_m:
        st.markdown(
            f"**{eid}** · track {ev.get('track_id', '?')} · "
            f"{st.session_state.view_idx + 1}/{len(all_events)}"
        )
    with nav_r:
        if st.button("Next ▶", disabled=st.session_state.view_idx >= len(all_events) - 1):
            st.session_state.view_idx += 1
            st.rerun()

    suggest_reject = bool(ev_quality and ev_quality.get("suggest_reject"))
    current = decisions.get(eid)
    current_status = decision_status(current)
    btn1, btn2, btn3, btn4, btn5 = st.columns([1.2, 1.2, 1, 1, 1])

    def decide(decision, *, force_advance: bool = False):
        was_final = decision_status(decisions.get(eid)) in FINAL_STATUSES
        decisions[eid] = decision
        st.session_state.decisions = decisions
        save_decisions(OUTPUT_DIR, decisions, all_events, video_path)
        updated_stats = count_stats(all_events, decisions)
        update_review_timing(
            OUTPUT_DIR,
            int(updated_stats["total"] - updated_stats["remaining"]),
            len(all_events),
        )
        if st.session_state.auto_advance and (force_advance or not was_final):
            nxt = next_pending_index(all_events, decisions, st.session_state.view_idx + 1)
            if nxt is not None:
                st.session_state.view_idx = nxt
        if decision_status(decision) != STATUS_ACCEPTED_RANGES:
            st.session_state.pop("range_editor_eid", None)
        st.rerun()

    def clear_decision():
        decisions.pop(eid, None)
        st.session_state.decisions = decisions
        save_decisions(OUTPUT_DIR, decisions, all_events, video_path)
        updated_stats = count_stats(all_events, decisions)
        update_review_timing(
            OUTPUT_DIR,
            int(updated_stats["total"] - updated_stats["remaining"]),
            len(all_events),
        )
        st.rerun()

    with btn1:
        accept_primary = current_status == STATUS_ACCEPTED
        if st.button(
            "✅ Accept all (A)",
            type="primary" if accept_primary else "secondary",
            use_container_width=True,
            key="review_accept",
        ):
            decide(STATUS_ACCEPTED)
    with btn2:
        if st.button(
            "✂ Local adjust (E)",
            type="primary" if current_status == STATUS_ACCEPTED_RANGES else "secondary",
            use_container_width=True,
            key="review_local_adjust",
        ):
            st.session_state.range_editor_eid = eid
    with btn3:
        if st.button(
            "❌ Reject (R)",
            type="primary" if current_status == STATUS_REJECTED or (not current_status and suggest_reject) else "secondary",
            use_container_width=True,
            key="review_reject",
        ):
            decide(STATUS_REJECTED)
    with btn4:
        if st.button(
            "⏭ Skip (S)",
            type="primary" if current_status == STATUS_SKIPPED else "secondary",
            use_container_width=True,
            key="review_skip",
        ):
            decide(STATUS_SKIPPED)
    with btn5:
        if st.button(
            "↩ Clear",
            use_container_width=True,
            disabled=not decisions.get(eid),
            key="review_clear",
        ):
            clear_decision()

    st.caption(f"Most events: A/R/S. Use Local adjust only for mixed proposals. Auto-save → `{CONFIRMED_NAME}`")

    if st.session_state.get("range_editor_eid") == eid:
        start_time = float(ev.get("start_time") or 0.0)
        end_time = float(ev.get("end_time") or start_time)
        duration = max(0.001, end_time - start_time)
        existing_ranges = decision_ranges(current)
        default_range = (
            tuple(existing_ranges[0])
            if existing_ranges
            else (round(start_time, 3), round(end_time, 3))
        )
        step = max(0.001, round(duration / 200.0, 3))
        selected_range = st.slider(
            "Drag both ends to keep only the real-face range",
            min_value=round(start_time, 3),
            max_value=round(end_time, 3),
            value=default_range,
            step=step,
            key=f"review_range_slider::{eid}",
        )
        selected_range = [round(float(selected_range[0]), 3), round(float(selected_range[1]), 3)]
        st.info(
            f"Selected {selected_range[0]:.3f}s–{selected_range[1]:.3f}s of source video "
            f"({selected_range[1] - selected_range[0]:.3f}s)"
        )
        if existing_ranges:
            st.success(
                "Saved ranges: "
                + "; ".join(f"{a:.3f}s–{b:.3f}s" for a, b in existing_ranges)
            )

        range_a, range_b, range_c, range_d = st.columns([1.4, 1.2, 1.2, 1])
        with range_a:
            if st.button("✅ Save range & next", type="primary", use_container_width=True):
                decide(
                    {"status": STATUS_ACCEPTED_RANGES, "ranges": [selected_range]},
                    force_advance=True,
                )
        with range_b:
            if st.button("＋ Add another range", use_container_width=True):
                merged = merge_time_ranges([*existing_ranges, selected_range])
                decisions[eid] = {"status": STATUS_ACCEPTED_RANGES, "ranges": merged}
                st.session_state.decisions = decisions
                save_decisions(OUTPUT_DIR, decisions, all_events, video_path)
                updated_stats = count_stats(all_events, decisions)
                update_review_timing(
                    OUTPUT_DIR,
                    int(updated_stats["total"] - updated_stats["remaining"]),
                    len(all_events),
                )
                st.rerun()
        with range_c:
            if st.button(
                "✅ Finish & next",
                use_container_width=True,
                disabled=not existing_ranges,
            ):
                decide(current, force_advance=True)
        with range_d:
            if st.button("Cancel", use_container_width=True):
                st.session_state.pop("range_editor_eid", None)
                st.rerun()

    render_decision_banner(current)
    render_event_panel(OUTPUT_DIR, ev, ev_quality)


if __name__ == "__main__":
    main()
