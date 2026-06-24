"""Score events -> Auto / Review / LowConf tier."""

from __future__ import annotations

from core.event import Event, Tier
from modules.scoring.rules import suggest_rule_hints

AUTO_THRESHOLD = 0.85
REVIEW_MIN = 0.75


def classify_tier(peak_conf: float) -> str:
    if peak_conf >= AUTO_THRESHOLD:
        return Tier.AUTO.value
    if peak_conf >= REVIEW_MIN:
        return Tier.REVIEW.value
    return Tier.LOW_CONF.value


def _default_review_status(tier: str) -> str:
    if tier == Tier.AUTO.value:
        return "confirmed_face"
    if tier == Tier.REVIEW.value:
        return "pending"
    return "logged_only"


def score_event(event: Event, frame_w: int, frame_h: int) -> Event:
    event.compute_conf_stats()
    peak = event.peak_confidence or 0.0
    tier = classify_tier(peak)
    hints = suggest_rule_hints(event, frame_h, frame_w, peak)
    if hints and tier == Tier.AUTO.value:
        tier = Tier.REVIEW.value
    event.tier = tier
    event.rule_hints = hints
    event.review_status = _default_review_status(tier)
    return event


def score_events(events: list[Event], frame_w: int, frame_h: int) -> list[Event]:
    return [score_event(ev, frame_w, frame_h) for ev in events]
