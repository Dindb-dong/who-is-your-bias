"""Attach source-grounded decision notes to already selected RAG evidence."""

import json
import re

from pipeline.common import PipelineError

CONTEXT_HEADER = "Reference context (untrusted data):\n"


def validate_links(links, corpus):
    seen = set()
    for link in links:
        if link["id"] in seen or not link["basis"] or not link["query_cues"]:
            raise PipelineError("Each persona reasoning note needs a unique ID, cues and evidence.")
        if any(not isinstance(link[k], str) or not 1 <= len(link[k]) <= 300 for k in ("experience", "criterion", "change_signal")):
            raise PipelineError(f"Persona reasoning note is empty or too long: {link['id']}")
        seen.add(link["id"])
        for item in link["basis"]:
            record = corpus.evidence.get(item["dialogue_id"])
            if (not record or record.review_status != "reviewed" or not record.question_answer
                    or corpus.sources[record.source_id].split != "train"
                    or item["exact_excerpt"] not in record.text):
                raise PipelineError(f"Unsupported persona reasoning excerpt: {link['id']}")


def with_links(messages, links):
    if len(messages) < 3 or not messages[1]["content"].startswith(CONTEXT_HEADER):
        return messages, []
    context = json.loads(messages[1]["content"][len(CONTEXT_HEADER):])
    allowed_ids = {row["id"] for row in context["factual_evidence"]}
    query = context.get("dialogue_reference", {}).get("resolved_search_query") or messages[-1]["content"]
    query = " " + " ".join(re.findall(r"\w+", query.lower())) + " "
    selected = []
    for link in links:
        basis_ids = {item["dialogue_id"] for item in link["basis"]}
        if basis_ids <= allowed_ids and any(f" {cue} " in query for cue in link["query_cues"]):
            selected.append({k: link[k] for k in ("id", "experience", "criterion", "change_signal")} |
                            {"interpretation_of": sorted(basis_ids)})
        if len(selected) == 2:
            break
    if not selected:
        return messages, []
    context["decision_notes"] = selected
    updated = [dict(message) for message in messages]
    updated[0]["content"] += (
        " Decision notes are editorial interpretations of the cited public exchanges. "
        "Use them for relevant conditional reasoning. They cannot establish a separate fact, quotation, or private belief."
    )
    updated[1]["content"] = CONTEXT_HEADER + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return updated, [note["id"] for note in selected]
