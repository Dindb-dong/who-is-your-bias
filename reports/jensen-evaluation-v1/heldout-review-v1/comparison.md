# Heldout comparison after preserved blind review

**Do not promote the learned adapter from this run.** All 18 heldout calls were attempted: facts completed **4/6**, style **5/6**, and adapter **0/6**. All nine unsuccessful answers hit the unchanged 160-token output limit. No completed style/adapter pair exists, so this run cannot establish a semantic or persona-fidelity gain from the learned weights.

## Preservation and scope

The [blind findings](blinded-findings.json) and their [written review](blinded-findings.md) were preserved at `e3ed0796e0124a068f84856985c35839040b4e88` before unblinding. The [fresh-readback receipt](fresh-readback.json) verifies the archive and all four payloads. Original JSON/Markdown hashes remain unchanged. This report joins the fixed findings to their condition/case keys and adds a separate descriptive reference comparison; it does not regrade support or fill human ratings.

The [comparison JSON](comparison.json) contains all 18 attempts, exact sample/output/pool/finding hashes, input file identities and reference-comparison boundaries. Complete private inputs and derived evidence remain in the separately preserved artifact, not in these public notes.

## Completion and semantic findings use different populations

| Condition | Attempted | Normal completion | Length failures | Completed outputs with an unsupported claim | Completed outputs with factual citation-presence violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Facts | 6 | 4 | 2 | 2 of 4 | 3 of 4 |
| Style examples | 6 | 5 | 1 | 3 of 5 | 4 of 5 |
| Learned adapter | 6 | 0 | 6 | Not assessed | Not assessed |

All nine completed outputs contain no citation markers. Two entrepreneurial-advice outputs are conditional normative advice; citation absence is not itself their defect, and they remain coverage problems rather than invented biographies. The other seven completed outputs have the citation-presence violations already identified by the blinded reviewer. Citation markers, factual support and requested coverage are assessed separately.

The preserved claim labels contain 8 supported, 1 partially supported, 15 unsupported, 15 nonfactual and 5 uncertain claims, with no contradicted claims. These counts describe the reviewer's claim segmentation. They are not an aggregate performance score. Adapter truncations are not assigned semantic scores.

## Every case and attempt

| Case suffix | Topic | Facts | Style examples | Learned adapter |
| --- | --- | --- | --- | --- |
| 00072 | Selecting early accelerated-computing applications | Length failure | Length failure | Length failure |
| 00078 | Recognizing deep learning as important | Complete: `sample-0005` | Complete: `sample-0007` | Length failure |
| 00186 | CPU-scaling transition timing | Length failure | Complete: `sample-0004` | Length failure |
| 00190 | Management and organizational architecture | Complete: `sample-0001` | Complete: `sample-0002` | Length failure |
| 00253 | Founding hardship and entrepreneurial advice | Complete: `sample-0008` | Complete: `sample-0006` | Length failure |
| 00297 | Jacket provenance | Complete: `sample-0003` | Complete: `sample-0009` | Length failure |

There are **four** jointly completed facts/style pairs, **zero** facts/adapter pairs and **zero** style/adapter pairs. There is no case with three completed outputs. The learned-weight contrast cannot be replaced with a comparison of only the surviving base responses.

## Paired findings retained from blind review

- **00072 — Selecting early accelerated-computing applications:** All three attempts hit the original output cap. There is no completed output to compare semantically.
- **00078 — Recognizing deep learning as important:** Facts and style both retain the new-software-method point while omitting the supplied 2012 inflection and effectiveness/scalability explanation. Both overfocus on verifying the user's exact phrasing and omit citations. The saved reviewer observed a shared 34-token opening; no claim of copying origin or fidelity follows.
- **00186 — CPU-scaling transition timing:** Only style completes. Its lack of a fixed deadline does not repair omission of the supplied stalled-scaling/kernel account or the unsupported inevitable future sequence. There is no completed facts/style pair for this case.
- **00190 — Management and organizational architecture:** Facts retains some shared-information content with an unsupported adaptability attribution and missing hierarchy detail. Style instead asserts unsupported AI-driven corporate changes and omits the supplied management mechanisms. The paired blind findings do not show a benefit from adding the wording example on this case.
- **00253 — Founding hardship and entrepreneurial advice:** Both completed outputs are conditional generic advice, not invented biography. Style loosely acknowledges difficulty; facts does not. Neither covers the source's hardship/uncertainty mechanism. This limited coverage difference does not establish persona fidelity.
- **00297 — Jacket provenance:** Both acknowledge an evidence gap but add unsupported reinterpretations. Facts adds an industry/nondisclosure claim; style replaces the literal acquisition question with symbolic meaning. The supplied factual pool was empty in both conditions.

## Descriptive comparison with previously withheld references

References were opened only after the blind findings were durably preserved. They were never given to generation. Differences involving unsupplied details are recorded descriptively and do not alter the original support labels or create new requirements.

| Case | Reference emphasis | Descriptive comparison and boundary |
| --- | --- | --- |
| 00072 | The reference describes selecting applications and consulting scientific-computing users about problems beyond their reach. | No completed output exists; no reference-coverage judgment is assigned. Neither truncated text nor absence of a reviewer sample is scored as semantic reference coverage. |
| 00078 | The reference places recognition around a historical inflection and describes simultaneous interest from several research groups. | Both completed answers miss the historical timing and the reference's concrete research-contact narrative. The historical-year omission was already identified against the independently supplied training evidence. The heldout research-contact details were not supplied. Their absence is descriptive only, not a new grounding defect or required quotation match. |
| 00186 | The reference treats the transition as already historical for particular workloads, rather than a single future date for all computing. | The sole completed style answer gives a prospective physical-ceiling explanation instead. Its saved blind unsupported-future and missing-kernel findings remain unchanged. The private reference's historical interval and named examples were not supplied and do not become new required details. |
| 00190 | The reference discusses intentional organizational architecture, spans of control and autonomy of experienced senior leaders. | Facts overlaps at the broad management topic through information sharing, but neither completed output gives the reference's organizational-design reasoning. Style substitutes an asserted AI-driven realignment narrative. Reference mechanisms are comparison-only. The supplied-source coverage and support categories retain their original factual-pool basis. |
| 00253 | The reference emphasizes how difficult and onerous starting a company is. | Style has a broad difficulty overlap; facts offers generic advice without that central hardship focus. Both remain advice-oriented rather than an account of founding hardship. The motivating-uncertainty omission came from the supplied training passage, not this shorter heldout reference. Normative advice is not relabeled as fabricated biography. |
| 00297 | The reference supplies a short, concrete personal source for the jackets. | Neither generated answer reproduces that unavailable personal-source detail; both instead add unsupported reinterpretations to their evidence-gap response. The empty supplied pool made the private detail unavailable. Not reproducing it is not a failure; the already-blinded unsupported additions are the relevant defect. |

## Decision and limits

Keep the completed29 adapter preserved as an optional experimental artifact; **do not promote it to the default**. Its 0/6 normal-completion outcome and the absence of any complete style/adapter pair prevent a supported improvement claim under the registered budget. This does not prove how the same weights would behave under a different future protocol.

The wording-example condition completes one more answer than facts, but that additional answer retains substantive unsupported and coverage defects. Four paired completed cases show no consistent source-grounded benefit. These observations do not justify a default wording-policy change.

This is a small English heldout generation comparison using fixed manually selected training evidence. It does not measure retrieval quality, actual multi-turn sessions, live speech/avatar behavior or human likeness. Sequential CPU timings are not a controlled latency comparison. Human ratings remain null and `efficacy_claim` remains false.
