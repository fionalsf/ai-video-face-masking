"""Streamlit event-level review UI."""

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


def load_pending(review_dir: str) -> dict:
    with open(os.path.join(review_dir, "pending_events.json"), encoding="utf-8") as f:
        return json.load(f)


def load_confirmed(review_dir: str) -> dict:
    path = os.path.join(review_dir, "confirmed_events.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "video": "", "review_dir": review_dir, "started_at": None, "updated_at": None,
        "total_review_events": 0, "decided_count": 0,
        "summary": {"confirmed_face": 0, "rejected_fp": 0, "skipped": 0}, "events": [],
    }


def save_confirmed(review_dir: str, data: dict) -> None:
    with open(os.path.join(review_dir, "confirmed_events.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_queue(all_events: list, confirmed: dict):
    by_id = {e["event_id"]: e for e in confirmed.get("events", [])}
    decided_final = {eid for eid, e in by_id.items() if e.get("status") in ("confirmed_face", "rejected_fp")}
    status_map = {eid: e.get("status", "") for eid, e in by_id.items()}
    never = [e for e in all_events if e["event_id"] not in decided_final and status_map.get(e["event_id"]) != "skipped"]
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
        "event_id": event["event_id"], "track_id": event.get("track_id"),
        "start_time": event.get("start_time"), "end_time": event.get("end_time"),
        "start_timecode": event.get("start_timecode"), "end_timecode": event.get("end_timecode"),
        "peak_confidence": event.get("peak_confidence"), "avg_confidence": event.get("avg_confidence"),
        "status": status, "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decision_ms": int((time.time() - t0) * 1000), "operator": "local",
        "trajectory": event.get("trajectory", []),
    }
    confirmed["events"] = [e for e in confirmed.get("events", []) if e["event_id"] != event["event_id"]]
    confirmed["events"].append(entry)
    confirmed["decided_count"] = len(confirmed["events"])
    summary = {"confirmed_face": 0, "rejected_fp": 0, "skipped": 0}
    for e in confirmed["events"]:
        if e.get("status") in summary:
            summary[e["status"]] += 1
    confirmed["summary"] = summary
    confirmed["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_confirmed(review_dir, confirmed)


def main(review_dir: str | None = None):
    review_dir = review_dir or parse_review_dir()
    st.set_page_config(page_title="Face Event Review", layout="wide")
    if not review_dir or not os.path.isdir(review_dir):
        st.error('streamlit run review_ui.py -- --review-dir "output/NAME/review"')
        st.stop()
    if not os.path.isfile(os.path.join(review_dir, "pending_events.json")):
        st.error("Missing pending_events.json")
        st.stop()

    pending = load_pending(review_dir)
    confirmed = load_confirmed(review_dir)
    all_events = pending.get("events", [])
    queue, decided_final, status_map = build_queue(all_events, confirmed)
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "t0" not in st.session_state:
        st.session_state.t0 = time.time()
    if queue:
        st.session_state.idx = min(st.session_state.idx, len(queue) - 1)

    decided_count = len(decided_final)
    remaining = max(0, len(queue) - st.session_state.idx)
    eta_sec = int(remaining * avg_decision_ms(confirmed) / 1000)

    with st.sidebar:
        st.header("Events")
        st.caption(f"{decided_count}/{len(all_events)} decided")
        if remaining > 0:
            st.caption(f"~{eta_sec}s remaining")
        for ev in all_events:
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
        st.code(f'python confirm.py --output-dir "{Path(review_dir).parent}"')
        st.stop()

    ev = queue[st.session_state.idx]
    components.html("""<script>
    const doc=window.parent.document;
    doc.addEventListener("keydown",(e)=>{
      if(e.target&&(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA"))return;
      const k=e.key.toLowerCase(); if(!["a","r","s"].includes(k))return;
      for(const el of doc.querySelectorAll("button p,button span,button div")){
        const t=(el.textContent||"").trim();
        if(k==="a"&&t.startsWith("Accept")){el.closest("button").click();break;}
        if(k==="r"&&t.startsWith("Reject")){el.closest("button").click();break;}
        if(k==="s"&&t.startsWith("Skip")){el.closest("button").click();break;}
      }
    },{capture:true});
    </script>""", height=0)

    st.title("Face Event Review")
    st.caption(os.path.basename(pending.get("video", "")))
    c1, c2, c3 = st.columns(3)
    c1.metric("Progress", f"{decided_count + 1}/{len(all_events)}")
    c2.metric("Peak conf", f"{ev.get('peak_confidence', 0):.3f}")
    c3.metric("ETA", f"~{eta_sec}s")
    tc0 = ev.get("start_timecode") or f"{ev.get('start_time', 0):.1f}s"
    tc1 = ev.get("end_timecode") or f"{ev.get('end_time', 0):.1f}s"
    st.subheader(f"{ev['event_id']} | {tc0} ~ {tc1} | track {ev.get('track_id', '?')}")
    if ev.get("rule_hints"):
        st.warning("Rule hints: " + ", ".join(ev["rule_hints"]))
    for col, label in zip(st.columns(3), ["start", "mid", "end"]):
        rel = ev.get("previews", {}).get(label)
        with col:
            if rel:
                p = os.path.join(review_dir, rel.replace("/", os.sep))
                st.image(p, caption=label, use_container_width=True) if os.path.isfile(p) else st.caption(f"{label}: missing")
            else:
                st.caption(f"{label}: n/a")

    def advance(status: str):
        record_decision(review_dir, ev, status, st.session_state.t0)
        st.session_state.idx += 1
        st.session_state.t0 = time.time()
        st.rerun()

    st.divider()
    b1, b2, b3 = st.columns(3)
    if b1.button("Accept (A)", type="primary", use_container_width=True):
        advance("confirmed_face")
    if b2.button("Reject (R)", use_container_width=True):
        advance("rejected_fp")
    if b3.button("Skip (S)", use_container_width=True):
        advance("skipped")
    st.caption("Shortcuts: A / R / S")
