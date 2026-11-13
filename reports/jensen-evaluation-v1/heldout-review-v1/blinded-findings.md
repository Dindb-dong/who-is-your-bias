# Blinded heldout semantic findings

Reviewed all **9 supplied anonymized outputs across 5 represented cases** against their own factual pools. A sixth pool has no output in this reviewer packet. This is an `agent_semantic_assessment`; `prior_condition_exposure: false`, `human_review: ungraded`, and `efficacy_claim: false`. All seven human rating fields remain null.

No condition key, private material, reference answer, generation log, or external source was consulted. The shared style example was used only to inspect wording overlap. These findings do not identify conditions or establish persona fidelity.

## Integrity and limits

All sample IDs are unique, all case/question/pool mappings match, and all 11 supplied evidence-pointer fingerprints verify. The JSON records the exact answer UTF-8 SHA-256, full canonical sample SHA-256, question hash, and common-pool hash for every output. No answer text or human field was altered.

Provider finish reasons, completion/error fields, request hashes, execution receipts, and separately recorded expected output hashes are absent from the permitted reviewer packet. They remain unverified here. The registered heldout denominator is 18 calls; this review does not assign a status to calls missing from the packet or treat 9 visible answers as 9 independently verified completions.

## Findings by sample

| Sample | Source pool | Categorical findings |
| --- | --- | --- |
| sample-0001 | pool-0004 | Equal access is supported, but adaptability is newly attributed, fewer layers are omitted, and citations/dating are absent; the realignment gap is only partly precise. |
| sample-0002 | pool-0004 | Asserts AI-driven realignment, investments, and present corporate goals without support; misses the supplied information-access/hierarchy mechanisms and gives no citations. |
| sample-0003 | pool-0006 | Avoids naming a supplier but invents a tech-industry interpretation and claims public nondisclosure; the empty pool supports only a specific evidence-gap response. |
| sample-0004 | pool-0003 | No fixed deadline is invented, but the supplied stalled-scaling/kernel explanation is absent; thermal/yield constraints and an inevitable paradigm shift are unsupported. |
| sample-0005 | pool-0002 | Supported software-development significance appears, but the available 2012 moment and effectiveness/scalability reasons are omitted; exact-quote caution is overextended and citations/dates are missing. |
| sample-0006 | pool-0005 | Generic hypothetical advice is nonfactual rather than fabricated biography; it only loosely covers difficulty and omits the source's uncertainty/underestimation mechanism. |
| sample-0007 | pool-0002 | Like sample-0005, it retains a supported software point but misses 2012 and the effectiveness/scalability explanation; the exact-wording limitation is too broad and citations/dates are absent. |
| sample-0008 | pool-0005 | Hypothetical normative advice contains no fabricated biography, but the supplied founding difficulty, uncertainty, and motivating underestimation are all absent. |
| sample-0009 | pool-0006 | Acknowledges that provenance cannot be verified here, then asserts unsupported symbolism and customary usage; the first and last sentences repeat that reinterpretation. |

## Evidence locations

| Pool | Factual source and boundary |
| --- | --- |
| pool-0001 | `qa26-acquired-2023-turn-00077`; no sample for this pool is present. |
| pool-0002 | `qa26-strange-loop-2025-turn-00009`, `/pools/pool-0002/factual_evidence/0`: supports a 2012 inflection, observed effectiveness, scalability, and a new software-development method. Publication: 2025-06-26; event date unknown. |
| pool-0003 | `qa26-stripe-sessions-2024-turn-00203`, `/pools/pool-0003/factual_evidence/0`: supports stalled CPU scaling and acceleration of runtime-dominant kernels, without a universal transition deadline. Publication: 2024-05-21; event date unknown. |
| pool-0004 | `qa26-stripe-sessions-2024-turn-00126`, `/pools/pool-0004/factual_evidence/0`: supports shared information, collective reasoning, and fewer layers; does not establish an AI-driven reorganization history. Publication: 2024-05-21; event date unknown. |
| pool-0005 | `qa26-acquired-2023-turn-00339`, `/pools/pool-0005/factual_evidence/0`: supports severe founding difficulty, uncertainty, and the motivating value of not knowing the full hardship initially. Publication: 2023-10-16; event date unknown. |
| pool-0006 | `/pools/pool-0006/factual_evidence` is empty; jacket provenance and public nondisclosure are not established. |

The common wording-only source is `qa26-stanford-vftt-2024-turn-00130` at each pool's `style_examples/0`. It supplies no factual support and does not establish that any output received that example.

## Cross-cutting observations

- All 9 answers are English and contain no citation markers. Seven contain uncited factual or categorical future/interpretive assertions. Samples 0006 and 0008 are normative hypothetical advice; lack of citation is not itself their defect.
- The actual English prompt does not impose the Korean one-sentence, third-person, name-preservation, or no-direct-quotation rules. An explicit output-level simulation label is absent, but surrounding application labeling is unknown; labeling is therefore recorded as uncertain.
- A hypothetical heading does not make completed company changes, global nondisclosure, symbolic interpretations, or inevitable future sequences conditional. Samples 0006 and 0008 do use conditional advice framing; their substantive limitation is weak grounding and coverage.
- Samples 0005 and 0007 are ambiguous about exact quotation versus substantive historical timing. The exact user wording is not evidence, while the 2012 context is available. Their shared opening is a 34-token contiguous lowercase-word span, without any inference about conditions or origin.
- No extended copying from the factual or common style pools was observed. The largest factual overlap is a six-token technical description in 0005/0007; the largest common-style overlap is a three-token generic directive in 0006. These are lexical observations, not memorization or fidelity scores.
- Minor semantic repetition appears in the realignment limitation, deadline disclaimers, and exact-moment disclaimers. Sample 0009 materially repeats its symbolic reinterpretation. No single aggregate pass score is assigned.

## Review provenance and next gate

- `samples.json` SHA-256: `681a0bca581ffb202291d21974a458a8708a42efe0fd376d4c1c791b19e8650d`.
- `source-pools.json` SHA-256: `9dacfcfab052e2795beca543890edd211aaf117e4134e57a14f1494a8d1d3de0`.
- `rubric.md` SHA-256: `d7e6db218c0fb29110583930797d9e824e802ce10ca9d8447f9acf55e03fcfcd`.

The JSON companion contains claim-level categories, evidence pointers, all sample/output hashes, ambiguity notes, and unchanged null human fields. A second blinded agent independently checked samples 0003, 0004, and 0009 within the same permitted read scope. The rubric's actual commit is not identified in this packet; its content hash is preserved without treating its recorded authoring base as the rubric commit.

**The coordinator must preserve and verify these findings remotely before any unblinding or separate reference comparison.** No repository edits, commits, uploads, condition comparison, or promotion decision were performed in this subtask.
