"""A deliberately designed behavior baseline, never presented as measured imitation."""

from __future__ import annotations

import re

from .models import BehaviorCue, BehaviorPlan, Language, Mode, SpeechStyle
from .retrieval import classify_situation


def plan_behavior(text: str, question: str, *, mode: Mode, language: Language) -> BehaviorPlan:
    """Build an estimated timeline; the client anchors/scales it to actual playback.

    This is a deterministic baseline controller. No cue is an asserted observation
    of the subject. Even if source prose describes emotions, it does not prove a
    particular facial movement.
    """
    units = len(re.findall(r"\w+", text)) if language == "en" else len(re.sub(r"\s+", "", text)) / 2.4
    duration = max(1800, min(120_000, int(units / 2.6 * 1000) + 420))
    speech = SpeechStyle(rate=1.0, pitch=1.0, pause_ms=260)
    if mode != "behavior":
        return BehaviorPlan(cues=[], speech=speech, duration_ms=duration)

    situation = classify_situation(question)
    face = "smile" if situation == "praise" else "thoughtful" if situation in {"reflection", "challenge"} else "brow_raise"
    gesture = "shrug" if situation == "reflection" else "emphasis" if situation == "challenge" else "open_palm"
    cues = [
        BehaviorCue(start_ms=0, duration_ms=min(900, duration), channel="face", name=face, intensity=0.30),
        BehaviorCue(start_ms=0, duration_ms=min(550, duration), channel="gaze", name="aversion" if situation == "reflection" else "center", intensity=0.20),
        BehaviorCue(start_ms=350, duration_ms=min(850, duration - 350), channel="head", name="tilt" if situation == "reflection" else "nod", intensity=0.25),
    ]
    if duration >= 2800:
        start = min(1200, duration // 3)
        cues.append(BehaviorCue(start_ms=start, duration_ms=min(1600, duration - start), channel="gesture", name=gesture, intensity=0.40))
    if duration > 7000:
        start = duration // 2
        cues.append(BehaviorCue(start_ms=start, duration_ms=1100, channel="head", name="nod", intensity=0.18))
    return BehaviorPlan(cues=cues, speech=speech, duration_ms=duration)
