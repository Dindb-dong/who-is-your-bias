"""Bounded transparent text/signal diagnostics, never a speaker-identity score."""

from __future__ import annotations

import array
import io
import math
import re
import sys
import unicodedata
import wave
from collections import Counter
from typing import Sequence


class EvaluationError(ValueError):
    pass


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or len(text) > 12000 or "\x00" in text:
        raise EvaluationError("Text must contain at most 12000 characters and no NUL")
    return unicodedata.normalize("NFC", text).casefold().replace("’", "'")


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalize_text(text), re.UNICODE)


def characters(text: str) -> list[str]:
    return [character for character in normalize_text(text) if unicodedata.category(character)[0] in {"L", "N"}]


def edit_counts(expected: Sequence[str], observed: Sequence[str]) -> dict:
    """Levenshtein alignment with bounded work and linear memory.

    Tie order is diagonal, deletion, insertion. Counts can differ under another
    minimum-cost alignment, while the total edit distance remains unchanged.
    """
    if not expected:
        raise EvaluationError("Expected text has no scoreable tokens")
    if len(expected) * len(observed) > 4_000_000 or max(len(expected), len(observed)) > 10000:
        raise EvaluationError("Text comparison exceeds the bounded alignment budget")
    # Each cell is distance, substitutions, deletions, insertions.
    previous = [(index, 0, 0, index) for index in range(len(observed) + 1)]
    for i, token in enumerate(expected, 1):
        current = [(i, 0, i, 0)]
        for j, other in enumerate(observed, 1):
            if token == other:
                diagonal = previous[j - 1]
            else:
                d, s, omitted, inserted = previous[j - 1]
                diagonal = (d + 1, s + 1, omitted, inserted)
            d, s, omitted, inserted = previous[j]
            deletion = (d + 1, s, omitted + 1, inserted)
            d, s, omitted, inserted = current[-1]
            insertion = (d + 1, s, omitted, inserted + 1)
            current.append(min((diagonal, deletion, insertion), key=lambda cell: cell[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {"reference_units": len(expected), "hypothesis_units": len(observed),
        "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "distance": distance, "error_rate": round(distance / len(expected), 6)}


def lexical_metrics(expected: str, observed: str, language: str) -> dict:
    if language not in {"en", "ko"}:
        raise EvaluationError("This evaluation declares English and Korean only")
    target, recognized = words(expected), words(observed)
    word_error = edit_counts(target, recognized)
    char_error = edit_counts(characters(expected), characters(observed))
    n = min(3, len(target))
    counts = Counter(tuple(target[i:i + n]) for i in range(len(target) - n + 1))
    observed_counts = Counter(tuple(recognized[i:i + n]) for i in range(len(recognized) - n + 1))
    excess = sum(max(0, count - max(1, counts[gram])) for gram, count in observed_counts.items())
    suffix_units = characters(expected) if language == "ko" else target
    observed_units = characters(observed) if language == "ko" else recognized
    suffix = suffix_units[-min(4 if language == "en" else 8, len(suffix_units)):]
    suffix_found = any(observed_units[i:i + len(suffix)] == suffix for i in range(len(observed_units) - len(suffix) + 1))
    return {"normalization": "NFC, casefold, punctuation removed; no semantic or numeric rewriting",
        "primary_metric": "cer" if language == "ko" else "wer", "wer": word_error, "cer": char_error,
        "screens": {"missing_expected_suffix": not suffix_found, "excess_repeated_ngrams": excess,
            "ngram_size": n, "substantial_deletions": word_error["deletions"] / len(target) > 0.2},
        "limitations": "ASR is an independent recognizer, not ground truth. Screens may reflect ASR errors; they do not prove audio truncation, EOS, naturalness, speaker identity, or likeness. Korean spacing and digit-vs-word formatting affect WER."}


def wav_metrics(raw: bytes, *, max_seconds: int = 180) -> dict:
    if not isinstance(raw, bytes) or not 44 <= len(raw) <= 20_000_000:
        raise EvaluationError("WAV must contain 44 bytes to 20 MB")
    try:
        with wave.open(io.BytesIO(raw), "rb") as stream:
            channels, rate, width, frames = stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getnframes()
            if channels not in {1, 2} or width != 2 or stream.getcomptype() != "NONE" or not 8000 <= rate <= 96000:
                raise EvaluationError("Expected PCM16 mono/stereo WAV at 8–96 kHz")
            if not 0 < frames <= max_seconds * rate:
                raise EvaluationError("WAV duration is empty or exceeds the evaluation limit")
            pcm = stream.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise EvaluationError("WAV frame data is truncated")
    except (wave.Error, EOFError) as exc:
        raise EvaluationError("Invalid WAV container") from exc
    samples = array.array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    normalized_square = sum((sample / 32768) ** 2 for sample in samples)
    rms = math.sqrt(normalized_square / len(samples))
    peak = max(abs(sample) for sample in samples) / 32768
    tail = samples[-min(len(samples), int(0.1 * rate) * channels):]
    tail_rms = math.sqrt(sum((sample / 32768) ** 2 for sample in tail) / len(tail))
    return {"sample_rate": rate, "channels": channels, "frames": frames,
        "duration_seconds": round(frames / rate, 6), "bytes": len(raw),
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 3),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 3),
        "clipped_sample_fraction": round(sum(abs(sample) >= 32767 for sample in samples) / len(samples), 8),
        "near_silent_sample_fraction": round(sum(abs(sample) <= 32 for sample in samples) / len(samples), 8),
        "tail_100ms_rms_dbfs": round(20 * math.log10(max(tail_rms, 1e-12)), 3),
        "signal_present": peak > 1e-4,
        "note": "Signal statistics cannot establish quality or completeness; tail energy is descriptive, not a truncation detector."}
