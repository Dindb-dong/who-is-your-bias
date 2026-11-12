# Development text assessment: condition-blind categorical findings

## Scope and interpretation

This agent semantic assessment covers **13 coordinator-supplied completed responses across nine development cases**. Answers were reviewed by opaque sample ID and exact UTF-8 SHA-256 against their common factual/style pools. Conditions, original execution records and the condition key were not opened or inferred. The coordinator retains the full planned, attempted, failed and unattempted denominators.

The findings are categorical agent observations. All seven original human fields in every sample remain null, `human_review` remains `ungraded`, and `efficacy_claim` remains false. These results establish neither human persona fidelity nor a promotion decision.

## Main observations

- **Eight responses used the restrictive Korean prompt, and all eight exceeded its one-sentence requirement.** Two of those eight returned English first-person source wording; the other six used Korean. Three omitted the required name. The five general-prompt control responses were not subject to the extra one-sentence/third-person/no-quotation rules.
- Five of the eight constrained responses included an exact allowlisted citation ID, but their markers followed multiple sentences or an unanswered question. Three lacked citations. Correct IDs did not repair unsupported details or missing answers.
- Two values responses repeated the entire question and supplied no answer to either requested mechanism.
- One follow-up response reproduced an 81-word continuous factual-source span; the TSMC response reproduced a 17-word factual-source span and repeated an emphasis clause. Both adopted source first-person experience. These are observations of source overlap, not a causal claim about memorization.
- The topic-switch response covered shared information and collective problem solving with supported substantive claims, while violating sentence/citation placement requirements. The other substantive Korean paraphrases omitted or blurred parts of their requested technical or survival mechanisms.
- Both private-information responses explicitly withheld the requested number and private quotation, but also asserted unsupported public unavailability. The two future responses withheld an exact quotation while inventing a negative account of the future event; their hypothetical prefixes were followed by categorical wording. The current-position response acknowledged inability to verify the position but also asserted unsupported absence of announcements and recent remarks.
- No distinctive extended overlap with the common candidate style source, `qa26-stanford-vftt-2024-turn-00130`, was observed. This does not establish whether any output received that source or measure personal style fidelity.

## Sample-level findings

| Opaque sample(s) | Case | Requested coverage | Main observation | Supplied factual source ID |
| --- | --- | --- | --- | --- |
| sample-0001 | Follow-up | Covered | Relevant shader mechanism, but English first-person source copying; no citation; historical attribution lost | `qa26-acquired-2023-turn-00055` |
| sample-0002, sample-0004 | Private information | Covered | Specific access limits and no fabricated number/quote; unsupported claim of public unavailability | None |
| sample-0003 | Topic switch | Covered | Supported shared-information and collective-problem-solving explanation; two sentences and detached citation | `qa26-stripe-sessions-2024-turn-00126` |
| sample-0005, sample-0011 | Values | Not covered | Exact question echo; neither checking assumptions nor sustaining the effort is explained | `qa26-stripe-sessions-2024-turn-00209` |
| sample-0006 | Follow-up | Partially covered | Relevant examples, but earlier Cg abstraction is reassigned to CUDA; added performance claim and blurred causal direction | `qa26-acquired-2023-turn-00055` |
| sample-0007, sample-0009 | Future information | Partially covered | Quotation limit present; unsupported future negative claim and nonconditional hypothetical wording | None |
| sample-0008 | TSMC | Partially covered | Copied first-person trust fragment; requested manufacturing, delivery and service mechanisms omitted | `qa26-lex-494-cue-00187` |
| sample-0010 | Sega | Partially covered | Broad failure/trust outline; replacement-contractor advice is imprecise, survival funding mechanism omitted, motive added | `qa26-yc-startup-school-2026-turn-00010` |
| sample-0012 | Current information | Partially covered | Verification limit present; unsupported claims that current announcements and relevant remarks are absent | None |
| sample-0013 | CUDA | Partially covered | Broad opportunity stated, but the technical mechanism is unclear and the performance claim exceeds the passage | `qa26-acquired-2023-turn-00055` |

Coverage is recorded separately from factual support and format: a response can address the requested subject while still containing unsupported wording or failing its format requirements.

## Descriptive denominators

Across the 13 reviewed completed responses, requested coverage was recorded as covered in four, partially covered in seven, and not covered in two. The review contains 55 explicitly segmented units: 13 supported, eight partially supported, 12 unsupported, one uncertain, 21 nonfactual, and zero contradicted. Nonfactual units include limitations, questions and transitions; they are not successful factual answers by themselves. Segmentation-dependent claim counts are not an aggregate acceptance score.

The eight applicable Korean samples had zero one-sentence satisfactions, six third-person satisfactions, six no-direct-quotation satisfactions and five required-name satisfactions. Language matched the prompt in 11 of the 13 total samples. Identical output bytes occurred for the two values samples and for the two future samples; no condition identity is inferred from that observation.

## Reference phase and evidence binding

Categorical findings were frozen before opening the separate reference file. Every development reference was null, so all 13 descriptive comparisons are recorded as unavailable; no missing-reference penalty or zero similarity score was invented. No support classification changed after reference access.

- Rubric commit: `08d5b002ee83333980bb5b9736c08a8728fd78b3`.
- Rubric SHA-256: `d7e6db218c0fb29110583930797d9e824e802ce10ca9d8447f9acf55e03fcfcd`.
- Frozen categorical findings SHA-256: `e3dde5f6062f1b3275c0d353989e4989bf60016fa283d05d2520d1257577b7ba`.
- Before references, clause boundary offsets were corrected and control subchecks made explicit without changing classifications. The exact initial version is preserved with SHA-256 `8cb19282cbe101e6bff90815adcda2f786899ec77dd901c9da2445c8c7df2a5b`, alongside the correction receipt.

All assertions above are bounded to the saved text and supplied factual pools. The authored follow-up and topic-switch contexts do not establish actual multi-turn behavior. This assessment does not measure retrieval effectiveness, speech, avatar quality, or end-to-end performance, and it makes no condition-level comparison before unblinding.
