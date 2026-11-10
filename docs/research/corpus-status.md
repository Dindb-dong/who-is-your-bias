# Integrated corpus status

Reviewed on **2026-11-10 UTC**. The active persona directory is `data/personas/elon-musk`. Active-source and fixture-review dates use UTC; the research coverage manifests retain their original curation dates.

## What is available

The integrated corpus contains **31 distinct source events/documents and 163 curated evidence records**, spanning public material dated **2003-10-08 through 2026-02-05**. Normal retrieval uses **159 reviewed records from 30 training sources**. Lex Fridman episode 438 remains the sole test source, with its original four reference records withheld from retrieval.

| Contribution | Sources | Evidence records | Retrieval records | Held-out records |
| --- | ---: | ---: | ---: | ---: |
| Original curated collection | 7 | 28 | 24 | 4 |
| Expansion A: original interviews | 12 | 64 | 64 | 0 |
| Expansion B: talks, testimony, authored documents | 12 | 71 | 71 | 0 |
| **Integrated total** | **31** | **163** | **159** | **4** |

The evidence consists of **3,057 words of short original paraphrases**, not 31 full redistributed transcripts. Acquisition manifests and full documents collected locally are separate from the reviewed retrieval index. Source breadth and successful retrieval checks do not establish personality fidelity.

The original source audit is in [sources.md](sources.md). Expansion ID indexes and their coverage manifests remain under `data/expansion/batch-a` and `data/expansion/batch-b`; [expansion-b.md](expansion-b.md) documents that research pass. This status file describes the merged active corpus and supersedes the original collection's numerical inventory.

### Explicit review before indexing

Every curated record in the active collection, including both indexed expansion batches, explicitly declares `review_status: "reviewed"` and `confidence: "source_grounded"`. These flags record the completed source-curation pass described in the research notes. They do not represent automatic fact verification, an independent audit, or measured persona fidelity. The migration adds review metadata to those already curated paraphrases only; it does not approve acquired documents, raw transcripts, or importer output.

`Evidence.review_status` defaults to `"draft"`. A record with missing review metadata stays out of retrieval even if it declares `confidence: "source_grounded"`. Review and grounding are separate indexing conditions: a record marked reviewed but still carrying `"unverified_transcript"` confidence also stays out. Before deliberately promoting an imported transcript, a curator must check its text, source, speaker attribution, and locator, then explicitly set both the review and grounding flags. Unknown speaker attribution is not resolved by the loader. Imported transcripts continue to declare draft status and unverified confidence.

## Provenance and event identity

All 31 active sources now carry a curator-assigned `source_event_id` and its provenance. A transcript page, an episode page, and excerpts from the same event belong to one family. The three Stanford 2003 clips count as one event. The TED 2022 recorded conversation and live interview remain different events, as do TIME's Boca Chica interview and its later live event. Popular Mechanics' related excerpts are already grouped as one interview by the expansion curator.

The merge retains verified acquisition metadata: **24 sources have a `transcript_url` and 9 have a `video_url`**. The Stanford clip group also retains all three exact transcript URLs. No direct `audio_url` was verified for these active source records; none was guessed from an episode title or URL pattern. Recode's known podcast episode link is stored as `audio_episode_url`, not misrepresented as a downloadable audio file. Video links from coverage manifests retain their provenance and alignment limitations.

These counts indicate links and reviewed material, not downloaded media assets. In particular, **42 records have supplied transcript timestamps, of which 38 are in train**. A transcript clock is not automatically a clock for every video or podcast version. TED's transcript cues and YC's podcast transcript are explicitly unaligned with their YouTube versions. No numeric timestamps were added to untimed prose.

Publication, event, and capture dates remain distinct. Examples include the 2008 Henry Ford interview published in 2009, the May 2004 Senate hearing later reproduced by GPO, and a 2018 Tesla post preserved in a 2021 webpage capture. Later written answers appended to the March 2004 hearing retain an unspecified submission date. The MIT material is an event host's excerpted report. The 2014 shareholder letter is co-signed by Musk and Ahuja, and its evidence explicitly retains that shared corporate voice.

### Split isolation and duplicate checks

`Corpus.load()` rejects an event family spanning train, validation, or test. The retrieval accessor repeats this check for corpora constructed or modified in memory, so bypassing the file loader does not put a held-out event into the index. A canonical source URL shared across splits is also rejected, including differences that consist only of tracking parameters, an anchor, or a trailing slash. Content-bearing URL parameters remain significant.

Inventory tools may use the explicit inspection option `Corpus.load(directory, check_splits=False)` to load an invalid split assignment and emit its quality report. This option skips only the final split-isolation check. Schema validation, event-ID validation, duplicate identifier checks, and evidence-to-source references remain enforced. Accessing `training_evidence` or constructing a `Retriever` still rejects the inspected corpus until its split violation is corrected.

`Corpus.quality_report()` exposes event groups, missing event IDs, duplicate source URLs, and duplicate evidence after Unicode, case, whitespace, and punctuation normalization. It remains inspectable when an in-memory corpus contains a split violation. The integrated corpus has:

- 31 distinct event identities and no missing event IDs.
- No repeated event families or event-family split violations.
- No duplicate canonical source URLs.
- No duplicate normalized evidence texts.

This is exact normalized duplicate checking. It does not detect every paraphrase or prove that independently published remarks express different ideas. Repetition of an idea in a separate public interview may legitimately make a formerly unanswerable question answerable from train data.

## Evaluation fixture migration

The original 20 retrieval questions still pass after the merge without changing their target evidence IDs. Twelve new development cases cover stated fear, critical feedback, missed predictions, hiring priorities, deadlines, uncertain projects, humor, childhood learning, requirements, engineering autonomy, limits of personal experience, and acknowledging possible errors. Six of the new cases are in Korean.

The first run after adding those questions passed 33 of 39 graded cases. All six failures were Korean lexical mismatches. The fix added general Korean concept aliases and particle handling. It did not branch on a source ID, evidence ID, or complete test question. Alias matching also avoids interpreting `책` (book) inside `정책` (policy) or `책임` (responsibility), and prefers `신뢰성` (reliability) over the shorter `신뢰` (trust).

All 135 added records were reviewed for whether they supply the four held-out details. The decisions are preserved in `heldout.json` as `answerability_review` metadata:

| Case | Requested detail | Decision after expansion | Related train evidence considered |
| --- | --- | --- | --- |
| `h01` | Final step in the five-step sequence described in 2024 | Retain abstention: the requested sequence is absent from reviewed train records | Earlier testimony about goals and requirements |
| `h02` | Fraction of removed components to add back | Retain abstention: train material supplies no fraction | Stanford's component-count discussion and requirements testimony |
| `h03` | Podcast playback-speed settings | Retain abstention: the settings remain absent | Public recollections about books and learning |
| `h04` | Memes explained as compressed information | Retain abstention: no train passage supplies that explanation | Humorous names and social-media jokes |

These decisions concern **the curated index**, not every unindexed passage of the original source documents. Broader requirements and engineering questions are positive retrieval cases even though the specific 2024 sequence remains withheld. A separate unit test verifies that a held-out topic may correctly be answered when an independent train event supplies the answer. The evaluator must not equate topical overlap with leakage.

### Current measured results

The development retrieval run uses `k=4`:

| Check | Result |
| --- | --- |
| Answerable cases with a relevant top-four hit | **32 / 32** |
| Cases correctly returning no curated evidence | **7 / 7** |
| Total graded cases | **39 / 39** |
| Injection prompts requiring generated-answer review | **2 ungraded** |
| Returned held-out source events | **0** |

These are development smoke tests whose questions and aliases were reviewed together. They are **not** an unseen generalization benchmark. A relevant hit is not a measurement of generated answer entailment. Lexical confidence is not a calibrated probability of correctness. Injection cases remain ungraded because retrieval alone cannot establish prompt-injection resistance.

The full Python regression suite passed **141 tests at the initial integration checkpoint** and **151 tests at the subsequent review-gating and inspection-load checkpoint**. These counts are scoped to those runs; concurrent project work may add tests, and the final project total is reported separately. The latest targeted run passed all **42 corpus/retrieval/ingestion/evaluation tests**. Coverage checks cross-split event aliases, canonical URL aliases, same-event excerpts kept within one split, malformed event IDs, duplicate reporting, independent train answers to held-out topics, invalid evaluation targets, Korean alias collisions, and continued unsupported-query abstention. It also verifies that unknown-speaker transcripts with missing review fields remain unindexed, explicit review and grounding are independent gates, deliberate curator promotion enables indexing, and inspection mode can report split errors without permitting retrieval. One existing Starlette/httpx deprecation warning remains unrelated to these changes.

Reproduce the checks from the repository root:

```bash
uv run bias validate
uv run bias evaluate
uv run pytest -q
```

## Remaining limits

All 163 `behavioral_observation` fields remain null. The collection supplies no measured facial motion, gesture frequency, gaze, voice characteristics, or emotional state. Designed avatar cues remain editorial animation choices.

The corpus is uneven: interviews supply selected personal reflections while congressional testimony and company documents emphasize technical and commercial advocacy. Historical plans are not proof of achieved outcomes, and public assertions are not access to private thoughts. A response should preserve dates, speaker attribution, uncertainty, and the distinction between an individual answer and a corporate statement.

The four held-out references provide a small isolation check. Broader evaluation still requires independently authored questions, human review of generated answers, and source-disjoint test events that were not used to tune retrieval or persona instructions.
