# Reviewed live Q&A references

These records supplement the persona interview corpus at inference time. They
do not train model weights, become Jensen Huang's personal memories, or establish
that the avatar speaks for NVIDIA. `src/bias/qa_knowledge.py` owns validation,
source routing, freshness checks, and bounded relevant-record selection.

- `project.json`: ten code/status summaries audited at main
  `713232c74c890a61d7f6f6f4b67d5fed984b883c` on 2026-11-21. Branch status cards
  explicitly identify PR #27 and PR #29 as open at that audit. New work in the
  Q&A branch is not described as already merged. Review these cards whenever the
  selected runtime or team branch status changes, and before the presentation.
- `news.json`: two dated official NVIDIA releases opened and reviewed on
  2026-11-21, including the November 17 fiscal-results release and November 12
  leadership announcement. Company reports and forecasts remain attributed
  company statements. This small digest is not comprehensive news coverage.

The news review window expires **2026-11-23 14:40 UTC** (23:40 KST); the project
cards expire **2026-11-28 14:40 UTC**. Expired and future-reviewed entries are
excluded. Publication dates are also checked: a freshly reviewed old release
does not become current news. The runtime accepts only news published within
31 days of the request and reviewed for at most 48 hours. Re-read a source and
check for updates before extending its review window; do not only change dates.

## Review workflow

```bash
python tools/qa_news.py validate data/qa
python tools/qa_news.py candidates --output data/local/news-candidates-2026-11-22.json
```

Candidate collection fetches metadata from the official newsroom RSS feed. It
does not read or verify article contents and emits a different, runtime-ineligible
schema. A reviewer must open the actual article, confirm the publication date,
write a short paraphrase with explicit attribution, select specific query cues,
and record `reviewer`, `reviewer_type`, `reviewed_at` and `expires_at` in the
reviewed digest. Full publisher articles are not copied into the repository.

Schema validation checks structure and dates; it does **not** establish semantic
truth. An automated model's review is recorded as `agent`, not human review.
News source URLs are restricted to first-party NVIDIA and SEC domains; company
press releases are not independent confirmation of company performance claims.
No per-question network request is made, so this adds no web-search wait to the
live answer path. Runtime source freshness is checked for every question.

## Answer boundaries

Project questions use code/documentation cards, not unrelated interview facts.
The model is asked to distinguish implementation from deployment and evaluation,
and not to invent a model-selection rationale. Missing project details produce
an explicit request for team confirmation. Mixed project/current-news questions
are marked as only partially covered when the project route is selected.

Personal buy/sell requests receive a general risk-comparison response, without
recommendations, current share prices, or promises. Personal allegations receive
an evidence-limitation response without confirming the premise or asserting that
no report exists anywhere. A leadership announcement does not justify a private
allegation or a causal inference about it. General advice and historical persona
questions continue to use the existing dialogue pipeline.

The route classifier uses bounded bilingual lexical patterns. It is an explicit
fallback layer, not a general fact checker or complete sensitive-topic detector.
Novel wording and ambiguous follow-ups require integration tests and review.
