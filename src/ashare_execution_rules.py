
from __future__ import annotations
import pandas as pd


def preferred_ashare_execution(signal_date, config):
    """
    A-share execution planner.

    For all-A-share post-close fixed-price trading becoming available from
    the configured rule date, the default here is NEXT-DAY post-close fixed price.
    This is intentionally conservative because a daily-basic vendor feed may update
    after the 15:05 window starts; same-day post-close execution requires a validated
    real-time pre-close/close data stack.
    """
    d = pd.Timestamp(signal_date)
    ash = config["ashare_pro"]
    rule_date = pd.Timestamp(ash["post_close_fixed_price_from"])
    if d >= rule_date:
        return ash.get("preferred_execution_after_2026_07_06", "next_day_post_close_fixed")
    return ash.get("legacy_execution", "next_open")


def execution_notes(mode):
    notes = {
        "next_open": "Signal after close; execute next trading day at/near open. Exposed to overnight gap.",
        "next_day_post_close_fixed": "Signal after close; execute next trading day's 15:05-15:30 fixed-price session at that day's close, subject to venue/broker eligibility and tradability.",
        "same_day_post_close_fixed": "Only use with validated real-time close inputs and operational ability to calculate/submit before the post-close window; do not use delayed vendor EOD data."
    }
    return notes.get(mode, "")
