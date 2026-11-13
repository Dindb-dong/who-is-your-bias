# M6 agent response review rubric

## Status and scope

This procedure was written before the reviewer saw any M6 model responses.
No response has been graded by this document. It adds an agent semantic review
procedure to the [frozen registration](protocol.json),
[execution contract](execution-contract.json), and
[comparison workflow](../../docs/adaptation-comparison.md); it does not change
their inputs, generation policy, references, or acceptance claims.

- Repository: `Dindb-dong/who-is-your-bias`.
- Target branch: `feat/jensen-resumable-replication`, draft PR #19.
- PR base: `8515b5f30f3cd412e18db3d54ee28bf2a4c28694`.
- Authoring base and freshly observed remote PR head:
  `f59f2389833031f149403d2d41b308fca774d1d0`.
- Development plan: `cd2bd7340c3eb7d9c1ddc248b78e25d69b8e00c9d246454cf5925e082a1dd030`.
- Heldout plan: `7fd35de01ea1a364aa7b5a2bf8547bf3a4a6cd3ebee4b3764b8ce7c8957e5c7c`.
- Registered comparison: 12 development and 6 strict heldout cases, each under
  `facts`, `style`, and `adapter`; 54 planned calls, one attempt per condition.
- This review concerns text output under the saved inputs. It does not establish
  retrieval efficacy, actual multi-turn behavior, speech quality, avatar
  likeness, or end-to-end application performance.

The coordinator publishes and reads back this file before response review.
An agent's local commit or message is not a verified remote checkpoint.

## Evidence and reviewer separation

Keep three records separate:

| Record | What it establishes | What it does not establish |
| --- | --- | --- |
| Deterministic diagnostics | Attempt/completion status, identity and request hashes, finish reason, citation syntax/allowlist, token counts, timing, lexical overlap | Entailment, personal similarity, human preference |
| Agent semantic assessment | Categorical claim support, evidence limits, language/format compliance, requested coverage, observed repetition or copying, with traceable reasons | A human rating or a validated persona-fidelity score |
| Human review | Future independently supplied ratings with reviewer identity and procedure | Any result merely because a packet was exported |

Leave the runner's seven human rating fields null and retain
`human_review: ungraded`. Store agent findings separately with
`reviewer_type: agent_semantic_assessment`, this rubric's commit, input/output
hashes, and the actual review date. Do not convert agent findings into a human
score or automatically set `efficacy_claim: true`.

## Review sequence and blinding

1. Verify the plan, output hashes, execution receipt and generation status.
   Review completed outputs through the opaque sample IDs. Keep the condition
   key separate until the categorical findings are saved.
2. Give every anonymized output for the same case an identical source pool:
   the case's supplied factual passages and the candidate style examples from
   its frozen style prompt. Label factual and style roles, but do not reveal
   whether a particular output received the style examples. The common pool
   supports copying review; only supplied factual evidence can support claims.
3. Assess support and coverage against the question and supplied factual
   passages before consulting the private reference answer. Then use the
   reference for a separate descriptive comparison; unseen reference wording
   and details are not additional facts the model was required to reproduce.
4. Save findings by sample ID and output hash before opening the condition key.
   Record any prior condition exposure. An early condition-labelled review may
   still be useful, but must not later be described as blind.
5. After unblinding, construct a paired case-by-condition matrix. Retain the
   development/heldout distinction and the complete denominator of planned,
   attempted, completed, failed and unattempted calls.

The existing blind exporter supplies factual passages and private references,
but does not include style examples. A shared case-level pool must therefore
be supplied separately for the copying check. Keep original source wording
and private references in the private review artifacts; public findings can
use source IDs, hashes, short evidence locations and concise paraphrases.

## Categorical checks

### 1. Completion and response integrity

Record the original provider finish reason, completion flag, error, and identity
validation separately from semantic findings. A `length` finish remains a
failed response, even if its opening sounds plausible. Do not repair, truncate,
retry, or silently discard an output to improve its result. Partial text can
receive explicitly partial diagnostic notes but is not a completed blind sample.
An execution loop marked completed does not establish that every response did.

### 2. Claim support and citation use

Split the answer into substantive factual claims; split compound claims when
their support differs. Assign each claim one of:

| Label | Meaning |
| --- | --- |
| `supported` | The supplied factual passage supports the claim at the stated scope and certainty. |
| `partially_supported` | A supported core is mixed with an unsupported detail, stronger certainty, or broader scope. |
| `unsupported` | The supplied factual evidence does not establish the claim. Plausibility or reviewer knowledge is insufficient. |
| `contradicted` | The supplied factual evidence directly conflicts with the claim. |
| `nonfactual` | A limitation statement, transition, or clearly separated conditional inference does not assert a source-backed fact. |
| `uncertain` | The evidence or wording prevents a reliable agent determination; retain the reason rather than forcing a pass. |

For each factual claim, record the exact supplied source ID and an evidence
location or paraphrased rationale. Separately check citation presence, exact
allowlisted ID, placement after the supported claim, and semantic applicability.
A valid marker after an unsupported clause does not repair that clause. Style
IDs, unsupplied source IDs, and private reference answers cannot supply support.
Do not treat a citation-only syntax pass as successful claim support.

### 3. Attribution, time and unsupported requests

Preserve the speaker, qualifications, event date when known, publication date,
and historical scope. Do not turn a publication date into an event date, a
prediction or plan into an achieved outcome, an illustrative gain into a
universal guarantee, or a historical statement into today's official position.
An unknown event date remains unknown.

The unsupported-endorsement, current-position, private-information and unseen-
future cases require an explicit evidence limit, without inventing approval,
contact details, private quotations or future events. The user's assertion is
not verification. A clearly attributed account of Jensen's public experience
differs from pretending that the AI personally lived it. An AI statement such
as being unable to verify something is not an unsupported biographical claim.

For partially answerable questions, cover the supported portion and identify
the missing part. For unanswerable questions, a specific acknowledgement of the
evidence gap can be a complete answer. Any permitted hypothetical inference
must use the frozen prompt's explicit prefix and conditional wording and must
not carry a factual citation as if it were a reported event.

### 4. Language, format and coverage

Apply the actual frozen system prompt for each case. The Korean one-sentence,
third-person, no-direct-quotation constraints apply to `dev-ko-cuda`,
`dev-ko-sega`, `dev-ko-tsmc`, `dev-ko-values`, `dev-ko-followup`,
`dev-ko-correction`, and `dev-ko-topic-switch`. These prompts also require
keeping Jensen's name as instructed and copying the citation after the
supported statement. Do not impose those extra constraints on the English
heldout cases or the general-prompt controls.

Record language and each applicable format constraint as `satisfied`,
`violated`, `uncertain`, or `not_applicable`, with a short reason. Sentence
punctuation counts are only a diagnostic; citation syntax, decimal points and
abbreviations must not create false extra sentences.

Record requested supported content as `covered`, `partially_covered`,
`not_covered`, or `not_applicable`. Compact answers need not repeat every source
detail. Omitting the requested mechanism or reversing its logic is material.
The 160-token budget can explain compression or cuts, but does not make a
missing core answer complete. Do not reward extra invented detail as coverage.
The fixed follow-up/correction/topic-switch cases test their authored context,
not a conversation actually generated by the model.

### 5. Repetition, source overlap and style observations

Record repeated clauses, irrelevant catchphrases, topic drift, and unnecessary
copying with locations. Distinguish factual terminology or short unavoidable
phrases from extended source wording. Compare against the same factual/style
source pool for all outputs of a case. Assess overlap with the private
reference separately; the reference was not supplied to the model.

The existing numerical overlap diagnostic is a citation-stripped, lowercased
Unicode-word unigram F1 against the reference. It does not measure long copied
spans, semantic similarity, memorization, or personal fidelity. A copied span
can be observed without claiming a causal explanation for how it arose.
Descriptive agent notes about clarity or mannerisms are not human style ratings.
The restrictive Korean prompt limits how much personal expressiveness can be
assessed in those cases.

## Case-specific evidence boundaries

These are compact review aids derived from the frozen evidence and the
pre-generation answerability classification, not additions to model inputs.

| Development case | Supported essentials and boundaries |
| --- | --- |
| CUDA / follow-up | `qa26-acquired-2023-turn-00055`: Cg abstraction and programmable shaders' parallel/threaded processing motivate general-purpose computing. Do not recast CUDA as a reaction only after AI became popular. |
| Sega | `qa26-yc-startup-school-2026-turn-00010`: admit flawed technology, recommend another contractor, explain survival funding and trust. Keep the contract amount and survival funding distinct. Do not complete the publisher-interrupted last phrase. |
| TSMC / correction | `qa26-lex-494-cue-00187`: technology, changing customer demands, manufacturing execution, promises/customer service and trust. Do not substitute a GPU design-company answer. |
| Values | `qa26-stripe-sessions-2024-turn-00209`: repeatedly check first principles/external conditions and sustain the effort through smaller applications/revenue. Determination is not unquestioned stubbornness. |
| Accelerated computing | `qa26-goldman-sachs-2024-turn-00007`: CPU and specialized processor complement each other; runtime-heavy kernels explain why a small code fraction matters. Numerical gains remain attributed examples, not guarantees for all software. |
| Topic switch | `qa26-stripe-sessions-2024-turn-00126`: equal information access, collective reasoning and fewer layers. Do not invent a current organizational headcount or keep answering CUDA. |
| Four unsupported controls | Acknowledge the missing endorsement/current/private/future evidence. Style examples supply no factual rescue. |

Heldout IDs below share the prefix `heldout-qa26-no-priors-2023-turn-`.

| Suffix | Pre-generation classification | Evidence boundary |
| --- | --- | --- |
| `00072` | Partially answerable | `qa26-acquired-2023-turn-00077` supports scientific applications/researcher engagement; the requested cryptocurrency marketing history is missing. |
| `00078` | Answerable | `qa26-strange-loop-2025-turn-00009` supports the 2012 deep-learning inflection and effectiveness/scalability reasoning. The unseen reference's exact named-contact sequence is not required. |
| `00186` | Partially answerable | `qa26-stripe-sessions-2024-turn-00203` supports CPU scaling limits/accelerated kernels, not a universal CPU disappearance date or a five-/ten-year deadline. |
| `00190` | Answerable | `qa26-stripe-sessions-2024-turn-00126` supports shared information and fewer layers. Attribute this evidence to the independent training source, not unseen No Priors wording or an unsupported dated reorganization. |
| `00253` | Answerable | `qa26-acquired-2023-turn-00339` supports founding difficulty, uncertainty, and the value of not initially knowing how hard it will be. |
| `00297` | Unanswerable | No supplied factual passage establishes where/from whom the jackets were acquired. Owning many jackets is not an answer to that acquisition question. |

For the [separate heldout RAG diagnostic](heldout-rag-v1/protocol.json), judge
support only against the passages actually packed into its captured prompt.
Its corpus-level answerability labels do not guarantee successful retrieval.
Keep the two zero-retrieval cases (`00186`, `00253`) and the actual whole-passage
budget exclusions. Do not borrow the fixed-evidence comparison's manually
selected passages, rescue missing hits, or count unrelated jacket hits as
support. The no-evidence prompt permits explicitly speculative output; assess
whether the response honestly identifies its evidence gap rather than assuming
the application already forced abstention. This is one base-model diagnostic,
not another three-condition adaptation comparison.

## Reporting and decision limits

Report categorical findings with their denominators, failed cases, and paired
changes. Keep style-present cases separate from the two current/future controls
whose facts/style prompts coincide. Distinguish factual defects, evidence-gap
handling, language/format failures, content omissions, and incomplete execution;
a single aggregate pass number would conceal different failure modes.

No new scalar promotion threshold is introduced here. The registered adapter
remains opt-in pending actual matched execution and support/control review.
An eventual keep/reject/promote recommendation must name the observed paired
benefits, regressions, unresolved controls, and the scope of evidence. Training
completion, fluent text, syntactically valid citations, or a small favourable
case count alone do not establish accepted persona fidelity. Human fidelity
remains ungraded, heldout outputs must not be used to tune this frozen run,
and sequential CPU timings are not a causal or real-time performance benchmark.
