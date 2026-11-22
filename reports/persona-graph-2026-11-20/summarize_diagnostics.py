"""Recompute descriptive transport/citation diagnostics; these are not quality ratings."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    results = json.loads((ROOT / "results.json").read_text())
    assert results["manifest_sha256"] == manifest["manifest_sha256"]
    assert len(results["attempts"]) == len(manifest["schedule"]), "Complete the frozen schedule first"
    rows = []
    for attempt in results["attempts"].values():
        prompt = manifest["prompts"][attempt["prompt_key"]]
        cited = sorted(set(re.findall(r"\[source:([^\]]+)\]", attempt["raw_text"])))
        usage = attempt.get("usage") or {}
        rows.append({
            "attempt_id": attempt["attempt_id"], "condition": attempt["condition"],
            "status": attempt["status"], "elapsed_seconds": attempt["elapsed_seconds"],
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "exact_prompt_count_matches": (usage.get("prompt_tokens") == prompt["budget"].get("input_units"))
                                          if attempt["status"] == "ok" else None,
            "cited_evidence_ids": cited,
            "unauthorized_citation_ids": sorted(set(cited) - set(prompt["supplied_ids"])),
        })
    summaries = {}
    for condition in ("baseline", "flat", "graph"):
        subset = [row for row in rows if row["condition"] == condition]
        generated = [row for row in subset if row["status"] == "ok"]
        times = sorted(row["elapsed_seconds"] for row in generated)
        summaries[condition] = {
            "statuses": dict(Counter(row["status"] for row in subset)),
            "generation_request_seconds_median": statistics.median(times) if times else None,
            "generation_request_seconds_p90_nearest_rank": times[math.ceil(.9 * len(times)) - 1] if times else None,
            "mean_prompt_tokens": statistics.fmean(row["prompt_tokens"] for row in generated) if generated else None,
            "mean_completion_tokens": statistics.fmean(row["completion_tokens"] for row in generated) if generated else None,
            "attempts_with_unauthorized_citations": sum(bool(row["unauthorized_citation_ids"]) for row in subset),
            "exact_prompt_count_mismatches": sum(row["exact_prompt_count_matches"] is False for row in subset),
        }
    output = {
        "manifest_sha256": manifest["manifest_sha256"],
        "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest(),
        "scope": "Recorded sequential CPU generation requests; not team-Mac or end-to-end avatar latency. Citation membership does not establish semantic entailment.",
        "conditions": summaries, "attempts": rows,
    }
    (ROOT / "diagnostics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
