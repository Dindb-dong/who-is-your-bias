"""Conservative descriptors of observable speech and image geometry.

These functions do not infer a person's emotion, intention, or identity. A
missing measurement stays missing and an unreviewed camera stays unreviewed.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from tools.media.common import MediaError, finite_number
from tools.motion.mining import geodesic
from tools.references.workflow import asr_words, publisher_words, tokens


def speech_units(text: str, language: str) -> tuple[int, str]:
    """Name the unit; do not silently call English words Korean syllables."""
    if language == "en":
        return len(tokens(text)), "orthographic_words"
    if language == "ko":
        # Hangul blocks are orthographic syllables, not phonetic syllables. A
        # Latin/numeric mixture requires a separately declared transcription.
        if re.search(r"[A-Za-z0-9]", text):
            raise MediaError("Korean mixed-script speech needs a reviewed spoken transcription")
        return len(re.findall(r"[가-힣]", text)), "hangul_syllable_blocks"
    raise MediaError("Declare a supported speech-rate unit for this language")


def aligned_exchange(publisher: dict, asr: dict, alignment: dict, *, question_cues: list[str], answer_cues: list[str], questioner: str, respondent: str, language: str) -> dict:
    """Measure only ASR words fully matched to selected publisher speaker cues.

    Full-answer onsets/offsets require the original boundary tokens. A truncated
    ASR window cannot become a complete answer. Subtitle cue times are never
    subtracted from media word times.
    """
    if not question_cues or not answer_cues or set(question_cues) & set(answer_cues):
        raise MediaError("An exchange needs disjoint original question and answer cues")
    cues = {cue["id"]: cue for cue in publisher["segments"]}
    for selected, speaker in ((question_cues, questioner), (answer_cues, respondent)):
        if len(selected) != len(set(selected)) or any(cue not in cues for cue in selected):
            raise MediaError("Exchange cue IDs are missing or duplicated")
        if any(cues[cue].get("speaker") != speaker for cue in selected):
            raise MediaError("Selected exchange cues disagree with publisher speaker labels")
    if publisher["source_id"] != asr["source_id"] or alignment["source_id"] != asr["source_id"]:
        raise MediaError("Speech sources differ")
    if alignment["media_timeline"] != asr["source_timeline"] or alignment["publisher_timeline"] != publisher["source_timeline"]:
        raise MediaError("Alignment clock differs from its source")
    words = asr_words(asr)
    pub_words = publisher_words(publisher)
    matches = alignment["matches"]
    by_word: dict[int, list[dict]] = {}
    for match in matches:
        by_word.setdefault(match["asr_word_index"], []).append(match)

    def selected_words(ids: list[str], speaker: str) -> tuple[list[dict], dict[int, dict], list[int]]:
        indices = [index for index, word in enumerate(pub_words) if word["cue_id"] in ids]
        if not indices:
            raise MediaError("Exchange has no caption text tokens")
        result, mapped = [], {}
        for index, word in enumerate(words):
            found = by_word.get(index, [])
            # An ASR word may tokenize into several components. Reject partial
            # matches rather than count the whole word as verified.
            if len(found) != len(word["tokens"]) or not found:
                continue
            if any(match["cue_id"] not in ids or match["speaker"] != speaker for match in found):
                continue
            result.append(dict(word, index=index))
            for match in found:
                mapped[match["publisher_token_index"]] = word
        return result, mapped, indices

    question, question_map, q_indices = selected_words(question_cues, questioner)
    answer, answer_map, a_indices = selected_words(answer_cues, respondent)
    question_end = question_map[q_indices[-1]]["source_end"] if q_indices[-1] in question_map else None
    answer_start = answer_map[a_indices[0]]["source_start"] if a_indices[0] in answer_map else None
    answer_end = answer_map[a_indices[-1]]["source_end"] if a_indices[-1] in answer_map else None
    rate = {"value": None, "unit": None, "unit_count": None, "window_start_seconds": None, "window_end_seconds": None, "window_duration_seconds": None, "status": "not_observed", "scope": "matched_source_window_including_internal_pauses"}
    rate_windows = []
    blocks: list[list[dict]] = []
    for word in answer:
        if not blocks or word["index"] != blocks[-1][-1]["index"] + 1:
            blocks.append([])
        blocks[-1].append(word)
    for block in blocks:
        duration = block[-1]["source_end"] - block[0]["source_start"]
        if len(block) < 3 or duration < 0.25:
            continue
        count, unit = speech_units(" ".join(word["word"] for word in block), language)
        rate_windows.append({"value": round(count / duration, 6), "unit": unit + "_per_second", "unit_count": count, "window_start_seconds": round(block[0]["source_start"], 6), "window_end_seconds": round(block[-1]["source_end"], 6), "window_duration_seconds": round(duration, 6), "status": "asr_candidate", "scope": "contiguous_matched_window_including_internal_pauses"})
    if len(answer) >= 3:
        lo, hi = answer[0]["source_start"], answer[-1]["source_end"]
        # Do not span an intervening question, unmatched hallucinated run or
        # dropped context and report it as the person's speech rate.
        selected_indices = {word["index"] for word in answer}
        intervening = [i for i, word in enumerate(words) if lo < word["source_end"] and word["source_start"] < hi]
        complete_window = set(intervening) <= selected_indices
        duration = hi - lo
        if complete_window and duration >= 0.25:
            count, unit = speech_units(" ".join(word["word"] for word in answer), language)
            rate.update(value=round(count / duration, 6), unit=unit + "_per_second", unit_count=count, window_start_seconds=round(lo, 6), window_end_seconds=round(hi, 6), window_duration_seconds=round(duration, 6), status="asr_candidate")
        else:
            rate["status"] = "intervening_unmatched_or_other_speaker_words" if not complete_window else "window_too_short"
    return {
        "source_timeline": asr["source_timeline"],
        "publisher_timeline": publisher["source_timeline"],
        "question_cue_ids": question_cues,
        "answer_cue_ids": answer_cues,
        "question_end_seconds_candidate": question_end,
        "answer_start_seconds_candidate": answer_start,
        "answer_end_seconds_candidate": answer_end,
        "answer_boundary_coverage": {"first_token": answer_start is not None, "last_token": answer_end is not None},
        "matched_question_words": len(question),
        "matched_answer_words": len(answer),
        "matched_answer_token_fraction": round(len(answer_map) / len(a_indices), 6),
        "answer_window_seconds": [answer[0]["source_start"], answer[-1]["source_end"]] if answer else None,
        "question_window_seconds": [question[0]["source_start"], question[-1]["source_end"]] if question else None,
        "speech_rate": rate,
        "speech_rate_windows": rate_windows,
        "pre_answer_pause": {"seconds": None, "status": "unreviewed_continuity_and_acoustic_boundaries", "candidate_gap_seconds": round(answer_start - question_end, 6) if question_end is not None and answer_start is not None else None},
        "speaker_basis": "publisher_text_labels_plus_exact_asr_token_matches",
        "audio_listening_review_completed": False,
        "limitations": ["Word timestamps are ASR estimates, not verified acoustic boundaries.", "The original recording may contain edits; a candidate turn gap is not a validated pre-answer pause.", "Speech rate includes pauses within the matched window and is not articulation rate.", "Speech rate is not computed across unmatched or other-speaker words."],
    }


def merge_speech_runs(runs: list[dict]) -> dict:
    """Keep partial windows separate; never add their durations or bridge gaps."""
    if not runs:
        return {"status": "no_covering_audio", "runs": [], "pre_answer_pause": {"seconds": None, "status": "no_covering_audio"}}
    timelines = {run["source_timeline"] for run in runs}
    if len(timelines) != 1:
        raise MediaError("An exchange cannot merge different source clocks")
    return {"status": "measured_asr_candidates", "source_timeline": next(iter(timelines)), "runs": runs, "pre_answer_pause": {"seconds": None, "status": "requires_recording_continuity_and_boundary_review"}}


def _spans(samples: list[tuple[float, bool]], *, step: float, minimum_duration: float, exclusions: list[tuple[float, float]]) -> list[dict]:
    spans, start, previous = [], None, None

    def finish() -> None:
        nonlocal start
        if start is not None and previous is not None and previous + step - start >= minimum_duration:
            spans.append({"start_seconds": round(start, 6), "end_seconds": round(previous + step, 6)})
        start = None

    for now, active in samples:
        if any(lo <= now < hi for lo, hi in exclusions):
            active = False
        if previous is not None and now - previous > step * 1.6:
            finish()
        if active and start is None:
            start = now
        elif not active:
            finish()
        previous = now
    finish()
    return spans


def phase_for_interval(start: float, end: float, speech_runs: list[dict]) -> str:
    """Conversation phase candidate only; it does not identify an image track."""
    phases = set()
    for run in speech_runs:
        for field, label in (("question_window_seconds", "listening_candidate"), ("answer_window_seconds", "speaking_candidate")):
            interval = run.get(field)
            if interval and interval[0] <= start and end <= interval[1]:
                phases.add(label)
    return next(iter(phases)) if len(phases) == 1 else "unknown"


def visual_features(rows: list[dict], geometry: dict, *, start: float, end: float, speech_runs: list[dict], exclusions: list[tuple[float, float]], smile_threshold: float = 0.5, minimum_smile_seconds: float = 0.25, visibility_threshold: float = 0.7, sample_fps: float = 8) -> dict:
    """Summarize one observed window without turning camera motion into a nod."""
    start = finite_number(start, "window start")
    end = finite_number(end, "window end", start + 0.001)
    step = 1 / finite_number(sample_fps, "sample rate", 0.5, 30)
    smile_threshold = finite_number(smile_threshold, "smile threshold", 0, 1)
    minimum_smile_seconds = finite_number(minimum_smile_seconds, "minimum smile duration", 0.01, 30)
    visibility_threshold = finite_number(visibility_threshold, "visibility threshold", 0, 1)
    if any(lo < start or hi > end or hi <= lo for lo, hi in exclusions):
        raise MediaError("Visual exclusions must be valid intervals within the window")
    selected = [row for row in rows if start <= row["source_timestamp_seconds"] < end]
    if not selected:
        raise MediaError("No sampled geometry lies inside the declared window")
    times = [row["source_timestamp_seconds"] for row in selected]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise MediaError("Visual samples must be strictly time-ordered")
    usable = [row for row in selected if not any(lo <= row["source_timestamp_seconds"] < hi for lo, hi in exclusions)]
    smiles = []
    smile_values = []
    hand_states: dict[str, Counter] = {"left": Counter(), "right": Counter()}
    hands: dict[str, list[tuple[float, list[float]]]] = {"left": [], "right": []}
    for row in selected:
        now = row["source_timestamp_seconds"]
        excluded = any(lo <= now < hi for lo, hi in exclusions)
        face = row["faces"][0] if not excluded and len(row["faces"]) == 1 else None
        values = face.get("blendshapes", {}) if face else {}
        score = min(values["mouthSmileLeft"], values["mouthSmileRight"]) if "mouthSmileLeft" in values and "mouthSmileRight" in values else None
        if score is not None:
            score = finite_number(score, "smile geometry score", 0, 1)
            smile_values.append(score)
        smiles.append((now, score is not None and score >= smile_threshold))
        pose = row["poses"][0] if not excluded and len(row["poses"]) == 1 else None
        landmarks = pose.get("landmarks", []) if pose else []
        for side, shoulder, elbow, wrist in (("left", 11, 13, 15), ("right", 12, 14, 16)):
            if excluded:
                hand_states[side]["excluded_camera_interval"] += 1
                continue
            if len(landmarks) <= wrist:
                hand_states[side]["no_single_pose"] += 1
                continue
            points = [landmarks[index] for index in (shoulder, elbow, wrist)]
            if any(not (0 <= p["x"] <= 1 and 0 <= p["y"] <= 1) for p in points):
                hand_states[side]["off_screen"] += 1
                continue
            if any(p.get("visibility", 0) < visibility_threshold or p.get("presence", 0) < visibility_threshold for p in points):
                hand_states[side]["occluded_or_low_confidence"] += 1
                continue
            hand_states[side]["observed"] += 1
            # Wrist relative to shoulder reduces common camera translation;
            # zoom/perspective remain uncorrected, hence a geometry candidate.
            hands[side].append((now, [points[2][axis] - points[0][axis] for axis in ("x", "y")]))
    intervals = _spans(smiles, step=step, minimum_duration=minimum_smile_seconds, exclusions=exclusions)
    for interval in intervals:
        interval["end_seconds"] = min(interval["end_seconds"], end)
        interval["phase"] = phase_for_interval(interval["start_seconds"], interval["end_seconds"], speech_runs)
        interval["status"] = "geometry_candidate"
    # A final partial frame interval cannot satisfy the minimum duration by
    # extending beyond the actual prepared recording.
    intervals = [interval for interval in intervals if interval["end_seconds"] - interval["start_seconds"] + 1e-9 >= minimum_smile_seconds]
    head = []
    offset = geometry["source"]["source_start_seconds"]
    for frame in geometry["frames"]:
        now = offset + frame["time_ms"] / 1000
        if start <= now < end and not any(lo <= now < hi for lo, hi in exclusions) and "head" in frame.get("bones", {}):
            head.append((now, frame["bones"]["head"]))
    head_pairs = [(left, right) for left, right in zip(head, head[1:]) if 0 < right[0] - left[0] <= step * 1.6 and not any(left[0] < hi and lo < right[0] for lo, hi in exclusions)]
    head_motion = {
        "status": "relative_geometry_candidate" if head_pairs else "unobservable",
        "observed_frames": len(head),
        "rotation_path_radians": round(sum(geodesic(a[1], b[1]) for a, b in head_pairs), 6) if head_pairs else None,
        "semantic_gesture": None,
        "camera_motion_reviewed": False,
        "reason": "Head rotation candidates require camera/cut review before a nod or tilt can be assigned.",
    }
    hand_results = {}
    for side, points in hands.items():
        pairs = [(a, b) for a, b in zip(points, points[1:]) if 0 < b[0] - a[0] <= step * 1.6 and not any(a[0] < hi and lo < b[0] for lo, hi in exclusions)]
        hand_results[side] = {
            "status": "relative_geometry_candidate" if pairs else "unobservable",
            "frame_states": dict(sorted(hand_states[side].items())),
            "observed_frame_fraction": round(hand_states[side]["observed"] / len(selected), 6),
            "wrist_path_in_normalized_image_units": round(sum(math.dist(a[1], b[1]) for a, b in pairs), 6) if pairs else None,
            "semantic_gesture": None,
        }
    return {
        "source_timeline": geometry["source"]["source_timeline"],
        "window_start_seconds": start,
        "window_end_seconds": end,
        "phase": phase_for_interval(start, end, speech_runs),
        "sampled_frames": len(selected),
        "excluded_frames": len(selected) - len(usable),
        "visible_smile": {"status": "geometry_candidates" if smile_values else "unobservable", "threshold": smile_threshold, "minimum_duration_seconds": minimum_smile_seconds, "observed_frames": len(smile_values), "mean_bilateral_score": round(statistics.mean(smile_values), 6) if smile_values else None, "intervals": intervals, "human_visual_review_completed": False},
        "audible_laughter": {"status": "not_listened", "intervals": None, "reason": "Visible smile scores and publisher stage directions cannot identify audible laughter by the respondent."},
        "head_movement": head_motion,
        "hands": hand_results,
        "exclusions": [{"start_seconds": lo, "end_seconds": hi, "reason": "camera_cut_or_unreviewable_camera_motion"} for lo, hi in exclusions],
        "speaker_association_reviewed": False,
        "track_continuity_reviewed": False,
        "camera_motion_reviewed": False,
    }
