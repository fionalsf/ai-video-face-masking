#!/usr/bin/env python3
"""Streamlit Timeline Debug UI — verify timeline.json before Render."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import streamlit as st

from timeline_generator import TIMELINE_NAME, generate_timeline, load_decisions


def parse_output_dir() -> str:
    argv = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == "--" else sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--output-dir" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
    return os.environ.get("OUTPUT_DIR", "")


OUTPUT_DIR = parse_output_dir()


def load_timeline(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, TIMELINE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="Timeline Debug", layout="wide")

    if not OUTPUT_DIR or not os.path.isdir(OUTPUT_DIR):
        st.error('Usage: streamlit run timeline_debug_ui.py -- --output-dir "output/detection/test456"')
        st.stop()

    st.title("Timeline Debug")
    st.caption(OUTPUT_DIR)

    col_a, _ = st.columns([1, 3])
    with col_a:
        if st.button("Generate / Refresh timeline.json", type="primary", use_container_width=True):
            if not load_decisions(OUTPUT_DIR):
                st.error("confirmed_events.json missing — complete Review first.")
            else:
                generate_timeline(OUTPUT_DIR)
                st.success("timeline.json updated")
                st.rerun()

    timeline = load_timeline(OUTPUT_DIR)
    if timeline is None:
        st.warning("No timeline.json — click Generate / Refresh after Review.")
        st.stop()

    entries = timeline.get("entries", [])
    accepted_events = timeline.get("accepted_events", [])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accepted Events", timeline.get("accepted_event_count", 0))
    m2.metric("Timeline Entries", timeline.get("timeline_entry_count", len(entries)))
    m3.metric("FPS", timeline.get("fps", "—"))
    m4.metric("Generated", timeline.get("generated_at", "—"))

    tab_overview, tab_event, tab_time, tab_frame = st.tabs(
        ["Overview", "By Event", "By Time", "By Frame"]
    )

    with tab_overview:
        st.subheader("Accepted Events Summary")
        st.dataframe(accepted_events, use_container_width=True, hide_index=True)
        st.subheader("Sample Timeline Entries (first 200)")
        st.dataframe(entries[:200], use_container_width=True, hide_index=True)
        with st.expander("Raw timeline.json"):
            st.json(timeline)

    with tab_event:
        event_ids = sorted({e["event_id"] for e in entries})
        choice = st.selectbox("Event ID", event_ids if event_ids else ["—"])
        if event_ids:
            ev_entries = [e for e in entries if e["event_id"] == choice]
            meta = next((e for e in accepted_events if e["event_id"] == choice), {})
            st.write(meta)
            st.dataframe(ev_entries, use_container_width=True, hide_index=True)
            st.line_chart({e["frame"]: e.get("confidence") or 0 for e in ev_entries})

    with tab_time:
        if not entries:
            st.info("No entries.")
        else:
            t_min = min(e["timestamp"] for e in entries)
            t_max = max(e["timestamp"] for e in entries)
            t_range = st.slider(
                "Time window (sec)",
                min_value=float(t_min),
                max_value=float(t_max),
                value=(float(t_min), float(min(t_max, t_min + 5))),
            )
            window = [e for e in entries if t_range[0] <= e["timestamp"] <= t_range[1]]
            st.caption(f"{len(window)} entries in window")
            st.dataframe(window, use_container_width=True, hide_index=True)

    with tab_frame:
        by_frame: dict[int, list] = defaultdict(list)
        for e in entries:
            by_frame[e["frame"]].append(e)
        frames = sorted(by_frame)
        if not frames:
            st.info("No entries.")
        else:
            f_idx = st.slider("Frame", min_value=frames[0], max_value=frames[-1], value=frames[0])
            hits = by_frame.get(f_idx, [])
            st.caption(f"{len(hits)} bbox(es) at frame {f_idx}")
            st.dataframe(hits, use_container_width=True, hide_index=True)
            if len(frames) > 1:
                counts = {f: len(by_frame[f]) for f in frames}
                st.bar_chart(counts)


if __name__ == "__main__":
    main()