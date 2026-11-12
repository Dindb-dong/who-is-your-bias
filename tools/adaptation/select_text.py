"""Freeze complete, event-balanced dialogue within a pinned tokenizer budget."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from services.language.adapters import immutable_revision, json_digest, tokenizer_contract
from tools.persona_data.pipeline import write_json, write_jsonl
from .datasets import file_hash, validate_dataset
from .train import encode_example


def select(dataset: Path, output: Path, *, model: str, revision: str,
           max_length: int = 768, min_answer_words: int = 24,
           max_per_event: int = 16, seed: int = 42) -> dict:
    if output.exists() or not immutable_revision(revision):
        raise ValueError("Choose a new selection and immutable tokenizer revision")
    if max_length < 32 or min_answer_words < 1 or max_per_event < 1:
        raise ValueError("Selection limits must be positive and fit a chat sequence")
    from transformers import AutoTokenizer

    manifest, rows = validate_dataset(dataset, "text")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
                                             local_files_only=True, trust_remote_code=False)
    groups, exclusions, token_counts = defaultdict(list), [], {}
    for row in rows:
        if len(row["messages"][-1]["content"].split()) < min_answer_words:
            exclusions.append({"id": row["id"], "reason": "answer_below_minimum_words"})
            continue
        encoded = encode_example(tokenizer, row["messages"], max_length)
        if encoded is None:
            exclusions.append({"id": row["id"], "reason": "complete_exchange_exceeds_context"})
            continue
        token_counts[row["id"]] = len(encoded["input_ids"])
        groups[row["event_family_id"]].append(row)
    rng, selected = random.Random(seed), []
    for event in sorted(groups):
        group = sorted(groups[event], key=lambda row: row["id"])
        rng.shuffle(group)
        selected.extend(group[:max_per_event])
        exclusions.extend({"id": row["id"], "reason": "event_balance_cap"} for row in group[max_per_event:])
    selected.sort(key=lambda row: row["id"])
    if not selected:
        raise ValueError("No complete source exchanges satisfy the frozen selection")
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", selected)
    selection = {"schema_version": "bounded-text-selection.v1", "input_dataset_sha256": manifest["dataset_sha256"],
                 "model": model, "tokenizer_revision": revision,
                 "tokenizer_contract_sha256": json_digest(tokenizer_contract(tokenizer)),
                 "max_length": max_length, "min_answer_words": min_answer_words,
                 "max_per_event": max_per_event, "seed": seed,
                 "input_samples": len(rows), "accepted_samples": len(selected),
                 "accepted_ids": [row["id"] for row in selected],
                 "accepted_token_counts": {row["id"]: token_counts[row["id"]] for row in selected},
                 "exclusions": exclusions, "truncated_examples": 0,
                 "human_fidelity_review": "ungraded"}
    manifest = {**manifest, "dataset_sha256": file_hash(output / "train.jsonl"),
                "samples": len(selected), "max_per_event": max_per_event, "seed": seed,
                "event_counts": dict(Counter(row["event_family_id"] for row in selected)),
                "selection_sha256": json_digest(selection), "source_selection": selection}
    write_json(output / "manifest.json", manifest)
    write_json(output / "selection.json", selection)
    validate_dataset(output, "text")
    return {"samples": len(selected), "event_counts": manifest["event_counts"],
            "dataset_sha256": manifest["dataset_sha256"], "selection_sha256": manifest["selection_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--min-answer-words", type=int, default=24)
    parser.add_argument("--max-per-event", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(select(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
