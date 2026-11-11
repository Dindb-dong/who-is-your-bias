"""Commit complete text units without speaking partial numbers/citations."""
from __future__ import annotations

import re

from .grounding import validate_citations


ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "st", "no", "fig"}


class SegmentCommitter:
    def __init__(self, *, turn_id: str, language: str, allowed_ids: set[str]):
        self.turn_id, self.language, self.allowed_ids = turn_id, language, allowed_ids
        self.buffer = ""
        self.offset = 0
        self.sequence = 0

    def _boundary(self) -> int | None:
        for match in re.finditer(r"[.!?。！？]", self.buffer):
            pos = match.start()
            if self.buffer[pos] == ".":
                if pos and self.buffer[pos - 1].isdigit() and (pos + 1 == len(self.buffer) or self.buffer[pos + 1].isdigit()):
                    continue
                word = re.search(r"([\w.]+)$", self.buffer[:pos])
                if word and (word[1].lower() in ABBREVIATIONS or len(word[1]) == 1 or re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", word[1])):
                    continue
            end = pos + 1
            while end < len(self.buffer) and self.buffer[end] in '\"”’\'':
                end += 1
            while True:
                while end < len(self.buffer) and self.buffer[end].isspace():
                    end += 1
                if end == len(self.buffer):
                    break  # Need look-ahead or normal generation completion.
                if self.buffer[end] == "[":
                    closing = self.buffer.find("]", end)
                    if closing < 0:
                        return None
                    if self.buffer[end:closing + 1].startswith("[source:"):
                        end = closing + 1
                        continue
                # CJK may have no spaces; other punctuation needs a boundary.
                if self.buffer[pos] in "。！？" or end > pos + 1:
                    return end
                break
        return None

    def _commit(self, end: int) -> dict | None:
        raw, self.buffer = self.buffer[:end], self.buffer[end:]
        start = self.offset
        self.offset += len(raw)
        text, valid, invalid = validate_citations(raw, self.allowed_ids)
        text = text.strip()
        if not text:
            return None
        sequence = self.sequence
        self.sequence += 1
        return {"turn_id": self.turn_id, "segment_id": f"{self.turn_id}:{sequence}", "sequence": sequence,
                "text": text, "raw_text": raw, "char_start": start, "char_end": self.offset,
                "language": self.language, "evidence_ids": valid, "invalid_citation_ids": invalid,
                "state": "committed", "semantic_status": "unreviewed"}

    def push(self, text: str) -> list[dict]:
        self.buffer += text
        results = []
        while (end := self._boundary()) is not None:
            item = self._commit(end)
            if item:
                results.append(item)
        return results

    def finish(self) -> list[dict]:
        # Only called after a normal provider finish; interrupted tails are
        # discarded by the caller, rather than treated as finished sentences.
        item = self._commit(len(self.buffer)) if self.buffer else None
        return [item] if item else []
