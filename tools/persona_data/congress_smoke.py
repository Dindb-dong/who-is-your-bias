"""Development API source-availability smoke for the combined congressional corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from bias.api import create_app
from bias.config import Settings, repository_root


# These developer-authored prompts were inspected while implementing the
# adapter. They are reproducible availability checks, never a held-out score.
AVAILABILITY = (
    ("senate-launch-2014", "In 2014, why did you prefer fixed-price competition for government space launches over cost-plus?"),
    ("house-space-cooperation-2004", "In 2004, why were small space satellites intended for the Shuttle?"),
    ("senate-future-launch-2004", "In 2004, what is a meaningful prize to engage the private sector in space?"),
    ("house-commercial-space-2005", "In 2005, what was the Falcon I price for access to space compared to Pegasus?"),
    ("senate-advanced-energy-2007", "In 2007, why did Tesla need lithium-ion cell and pack manufacturing in the United States?"),
    ("house-commercial-crew-2011", "In 2011, why did government space contracts need well-defined terms and costs?"),
)
PARAPHRASES = (
    ("senate-launch-2014", "In 2014, how would fixed-price competition for the basic launch vehicle relate to cost-plus?"),
    ("house-space-cooperation-2004", "In March 2004, did satellites intended for the Shuttle have to be done on the Shuttle?"),
    AVAILABILITY[2],
    ("house-commercial-space-2005", "In 2005, what price did the Falcon I have compared to the Pegasus launch vehicle?"),
    AVAILABILITY[4],
    ("house-commercial-crew-2011", "In 2011, why were SpaceX launch vehicles sourced 100 percent in the United States?"),
)


def run(corpus: Path, snapshot: Path) -> dict:
    # Explicit construction ignores ambient provider credentials and models.
    settings = Settings(corpus_dir=corpus, persona_snapshot=snapshot, retrieval_backend="lexical")
    root = repository_root()
    report = {"schema_version": "congress-api-availability.v1", "exposure": "development",
              "started_at": datetime.now(timezone.utc).isoformat(), "transport": "in_process_asgi_testclient",
              "llm_inference": False, "tts_inference": False, "human_review": "ungraded", "efficacy_claim": False,
              "checks": [], "paraphrase_diagnostics": [], "current_checks": [],
              "implementation_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [Path(__file__).resolve(), root / "tools/persona_data/congress.py", root / "tools/persona_data/pipeline.py",
                               *sorted((root / "src/bias").glob("*.py"))]},
              "limitations": ["Developer-inspected source-availability questions, not a fresh ranking benchmark.",
                  "Source previews do not establish answer entailment, neural generation, speech or target-device latency.",
                  "Topic-word anchors were added after observing lexical abstention; original paraphrases remain separate diagnostics."]}
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        report.update(snapshot=health["persona_snapshot"], corpus=health["corpus"], retrieval=health["retrieval"])
        for field, cases in (("checks", AVAILABILITY), ("paraphrase_diagnostics", PARAPHRASES)):
            for source, question in cases:
                response = client.post("/api/chat", json={"message": question, "language": "en"})
                response.raise_for_status()
                turn = response.json()
                ids = turn["grounding"]["retrieved_ids"]
                cited = [citation["evidence_id"] for citation in turn["citations"]]
                prefix = "qa26-" + source + "-"
                report[field].append({"expected_source": source, "question": question,
                    "provider": turn["provider"], "temporal_mode": turn["context"]["temporal"]["mode"],
                    "grounding_status": turn["grounding"]["status"], "retrieved_ids": ids, "citation_ids": cited,
                    "expected_source_retrieved": any(identifier.startswith(prefix) for identifier in ids),
                    "expected_source_cited": any(identifier.startswith(prefix) for identifier in cited),
                    "response_text_sha256": hashlib.sha256(turn["text"].encode()).hexdigest()})
        session = None
        for question in ("현재 2030년 화성 탐사 계획은 뭐야?", "그건 왜?"):
            response = client.post("/api/chat", json={"message": question, "language": "ko", "session_id": session})
            response.raise_for_status()
            turn = response.json()
            session = turn["session_id"]
            report["current_checks"].append({"question": question,
                "temporal_mode": turn["context"]["temporal"]["mode"], "grounding_status": turn["grounding"]["status"],
                "retrieved_ids": turn["grounding"]["retrieved_ids"], "citation_ids": [c["evidence_id"] for c in turn["citations"]],
                "response_text_sha256": hashlib.sha256(turn["text"].encode()).hexdigest()})
    report["source_availability_passed"] = all(row["expected_source_retrieved"] and row["expected_source_cited"]
        and row["temporal_mode"] == "historical" for row in report["checks"])
    report["current_abstention_passed"] = all(row["temporal_mode"] == "current" and row["grounding_status"] == "insufficient"
        and not row["retrieved_ids"] and not row["citation_ids"] for row in report["current_checks"])
    report["paraphrase_sources_retrieved"] = sum(row["expected_source_retrieved"] for row in report["paraphrase_diagnostics"])
    report["passed"] = report["source_availability_passed"] and report["current_abstention_passed"]
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new report path")
    report = run(args.corpus, args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        out.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "source_availability_passed", "current_abstention_passed", "paraphrase_sources_retrieved")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
