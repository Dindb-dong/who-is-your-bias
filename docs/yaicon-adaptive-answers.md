# Question-adaptive spoken answers

Implementation follow-up for [PR #24](https://github.com/Dindb-dong/who-is-your-bias/pull/24),
2026-11-18 (Asia/Seoul). Repository: `Dindb-dong/who-is-your-bias`.
Target branch: `feat/yaicon-live-latency-nov17`; previous verified head:
`ac5da0a5d4e5ade41c260a1e4752f1886dd6ad44`. Integrated main baseline:
`56b1868fac897fda18b258e79d70dd55c152c540` (through PR #26).
Local task branch: `feat/yaicon-adaptive-answers-nov18`.

## What changed

The baseline had generic “Adapt length to the question” guidance, fixed
1–3-sentence legacy prompts, a one-sentence Korean Jensen prompt, and PR #24's
optional two-sentence live prompt. There was no explicit question-type policy.

The new opt-in `adaptive-answer.v1` policy uses lightweight English/Korean
patterns **without an additional LLM call**. It supplies a per-turn sentence
target and response structure to the existing generator. It does not train a
model, add a persona graph, change TTS, or supply missing project knowledge.

| Question / example | Default target | Structure |
| --- | --- | --- |
| Factual: “What is an AI factory?” | 1–2 sentences | Direct answer, necessary qualification |
| Explanation: “왜 Qwen을 선택했나요?” | 2–4 | Main reason/mechanism, consequence |
| Comparison: “Compare local inference versus an API.” | 3–5 | Difference, benefit, limitation, suitable use |
| Procedure: “프로젝트 파이프라인은?” | 3–5 | Overall flow, key stages in order |
| Advice: “실패가 두려운 학생에게 조언한다면?” | 2–4 | Advice, relevant sourced experience or criterion |

The first sentence should contain the key answer (ideally ≤18 English words or
15 Korean eojeol). English whole-answer targets are about 45/95/120 words.
These are **prompt targets**, not guaranteed counts or hard cuts. Qualifications
and citations take priority; evidence should never be padded to fill a target.
These examples describe **routing**, not verified answers in the current corpus.

Current-turn requests override the default: “한 문장으로 / in one sentence” → 1;
“짧게 / briefly” → 1–2; “더 자세히 / in detail” → 3–5. The last recognized
instruction wins. The recognizer is deterministic and bounded; it is not a
general language-understanding model and cannot resolve every negation or mixed
request. Matching takes precedence comparison → advice → procedure → explanation
→ factual. Unrecognized questions use the factual target.

Bare format follow-ups retain the preceding **user** topic. For example,
“Why reuse rockets?” → “한 문장으로” → “더 자세히” searches the original topic,
without appending these format-only words to the retrieval query. A new question
such as “Briefly, what is a GPU?” remains a new topic. Past assistant claims are
never promoted to evidence. Old detail requests do not control the next turn.

## Enable and inspect

Set this in the existing API environment, then restart the API process:

```dotenv
BIAS_ADAPTIVE_RESPONSES=true
```

Default is `false`; historical behavior remains unchanged until enabled.
This applies to normal LLM-backed `/api/chat`, `/api/chat/stream`, and
`/api/chat/live`. It takes precedence over `BIAS_LIVE_CONCISE` when both are on.
Profile-only ablations and explicitly supplied versioned/custom prompt builders
remain unchanged. No web UI change is required; requests use normal text.

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How do you build conviction for a high-risk decision? Explain in detail.","language":"en","mode":"behavior"}'
```

Use your actual API host/port. Configured health reports
`answer_policy: adaptive-answer.v1` and the same `live_response_policy`.
Generated turns expose `context.answer_policy`: detected question type, detail,
sentence targets, `evidence_limited`, version, and `enforcement: prompt_guidance`.
Fixed evidence previews / blocked requests do not claim adaptive generation.
The live benchmark includes policy configuration in its continuity receipt.

## Preserved constraints

- Select policy after existing evidence, private-information and temporal gates.
  With no retrieved evidence, detailed requests remain limited to 1–2 English
  sentences or the existing one-criterion Korean speculative format. Existing
  simulation labels and no-invented-facts restrictions remain in force.
- Supported Korean answers can use several sentences, but still prohibit
  invented details, first-person attribution and direct quotations. The whole
  Korean draft is reviewed **before** emitting text/speech segments; this work
  does not claim token-by-token Korean speech or shorter review latency.
- Snapshot policy instructions are counted before evidence/history packing.
  Only complete supplied evidence records authorize citation IDs.
- Legacy/versioned prompt builders retain their original outputs. The
  production wrapper adds presentation guidance only when explicitly enabled.
- Do not reduce the provider token cap to force short answers or truncate
  generated sentences/citations. Citation checks remain mechanical; they do not
  prove that a statement is supported by its source.

## Validation and remaining acceptance

Implementation checks are in `tests/test_answer_policy.py`, alongside the
existing live, dialogue, grounding, context-budget, provider and benchmark tests.
Current checkpoint: 53 focused policy/live-concise checks passed; broader
regression and independent code review are in progress.
Stub LLM / synthetic audio tests are **not** measured Qwen answer quality,
Jensen fidelity, browser audibility or latency improvements.

Before the presentation, run representative EN/KO questions on the actual
LLM/TTS deployment with the flag off/on. Include factual, comparison, procedure,
advice, bare follow-ups and unsupported questions. Inspect actual sentence
counts, answer correctness, citation support, first-text/first-audio time and
total completion. Longer comparison answers may take longer overall while still
giving the key answer first. See the [latency runbook](yaicon-live-latency.md) for
the existing timing protocol. Backend policy tests can run on any supported
computer; the final latency measurement must use the intended demo deployment.
