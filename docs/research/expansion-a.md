# Interview corpus expansion A

## Delivered scope

This batch adds **12 original interview events and 64 reviewed evidence records** from **8 publisher families**, published from October 2012 through February 2026. Thirty records have publisher transcript cue or transcript timestamps. The committed evidence totals **1,090 words of original paraphrase**, with no intended verbatim quotations.

This batch is integrated into the active persona corpus. Its index preserves the contribution boundary, while the canonical corpus stores each source and evidence text once. The additions all use `split: train`; the existing Lex Fridman episode 438 remains held out. The batch alone does not establish persona accuracy, memory of an entire interview, model training, or voice/gesture imitation quality.

- `data/expansion/batch-a/index.json`: source and evidence IDs resolving to the canonical active corpus.
- `data/personas/elon-musk/`: canonical source metadata and reviewed evidence text.
- `data/expansion/batch-a/coverage.json`: word budgets, families, public media links, timeline boundaries, and four separate acquisition candidates.

## Source inventory and text budgets

| Source ID | Publication date | Original publisher | Records | Paraphrase words / budget |
|---|---|---|---:|---:|
| `ted-2013` | 2013-03-19 | [TED](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript) | 6 | 112 / 180 |
| `ted-2017` | 2017-04-30 | [TED](https://www.ted.com/talks/elon_musk_the_future_we_re_building_and_boring?view=transcript) | 6 | 103 / 180 |
| `ted-2022-recorded` | 2022-04-18 | [TED](https://www.ted.com/talks/elon_musk_a_future_worth_getting_excited_about?view=transcript) | 6 | 108 / 180 |
| `ted-2022-live` | 2022-04-15 | [TED](https://www.ted.com/talks/elon_musk_elon_musk_talks_twitter_tesla_and_how_his_brain_works_live_at_ted2022?view=transcript) | 6 | 113 / 180 |
| `dwarkesh-2026` | 2026-02-05 | [Dwarkesh Podcast](https://www.dwarkesh.com/p/elon-musk) | 7 | 124 / 180 |
| `yc-2016` | 2016-09-15 | [Y Combinator](https://www.ycombinator.com/blog/elon-musk-on-how-to-build-the-future) | 6 | 113 / 180 |
| `recode-2018` | 2018-11-02 | [Recode / Vox](https://www.vox.com/2018/11/2/18053428/recode-decode-full-podcast-transcript-elon-musk-tesla-spacex-boring-company-kara-swisher) | 5 | 79 / 100 |
| `wired-2012` | 2012-10-21 | [WIRED](https://www.wired.com/2012/10/ff-elon-musk-qa/) | 5 | 74 / 100 |
| `time-2021-boca-chica` | 2021-12-13 | [TIME](https://time.com/6127757/elon-musk-interview-person-of-the-year-2021/) | 4 | 63 / 100 |
| `time-2021-live` | 2021-12-14 | [TIME](https://time.com/6128235/elon-musk-interview-person-of-the-year/) | 4 | 64 / 100 |
| `popular-mechanics-2019` | 2019-02-25 | [Popular Mechanics](https://www.popularmechanics.com/space/moon-mars/a26513651/elon-musk-interview-spacex-mars/) | 4 | 66 / 100 |
| `gq-2015` | 2015-12-12 | [GQ](https://www.gq.com/story/elon-musk-mars-spacex-tesla-interview) | 5 | 71 / 100 |

The tighter 100-word budgets reflect source-specific retrieval limits. The six sources with 180-word budgets remain below the default 200-word limit. Word counts apply to the committed evidence text; this document deliberately adds provenance rather than repeating the substantive interview summaries. Interview titles, headings, and names serve only to identify the source. No complete article, transcript, screenshot, or raw media is committed in this batch.

## What is and is not evidence

Every record paraphrases an answer attributable to Musk. A host’s flattering characterization, the journalist’s analysis of his personality, another guest’s statement, and unrelated site content are excluded. Historical statements and forecasts retain their context; a dated assertion about a company or technology is not independent proof that the assertion was accurate or remains current.

The expanded themes include formative experiences, reading and entertainment, usefulness, criticism, failure, uncertainty, hiring, work priorities, design, and verbal humor in identifiable exchanges. `topics` and `situations` are editorial retrieval tags. They are not diagnoses, measured personality traits, labels of inner emotion, or automatically learned behavior. Korean topic aliases support bilingual retrieval without claiming a Korean primary transcript.

`behavioral_observation` is null in every record. A source can establish that a verbal joke occurred, but this batch does not infer a facial expression, gesture, gaze direction, or vocal timing from written text. Even where an article includes a reporter’s description of laughter or mannerisms, that description has not been promoted into a motion-control record.

## Transcript and media clocks

### TED: four separate interviews

The TED pages expose public `__NEXT_DATA__` containing `videoData`, `videoPlayerData`, and the English `transcriptData.translation.paragraphs[].cues[]`. This was read through ordinary public HTTPS access. Each committed numeric time is the observed cue `time` in milliseconds divided by 1,000; the label identifies the one-based publisher paragraph and passage. The timeline is named `ted_transcript_cue`.

The TED embed advertises a separate YouTube version. Its duration differs from the TED media in several cases. The batch records both observed durations and the video ID, but **does not claim that the TED cue time is a matching YouTube seek time**. Only start times are retained; no guessed cue end or gesture interval is added.

| Interview | Observed YouTube ID | TED player duration | YouTube duration |
|---|---|---:|---:|
| TED2013 | `IgKWPdJWuBQ` | 1,244 s | 1,265 s |
| TED2017 | `zIwLWfaAg-8` | 2,437 s | 2,451 s |
| TED2022 recorded | `YRvf00NooN8` | 3,983 s | 3,985 s |
| TED2022 live | `cdZZpaB2kDM` | 3,181 s | 3,286 s |

The publication dates come from `videoData.publishedAt`. There are discrepancies between TED’s structured `recordedOn` field and its written descriptions: the recorded 2022 interview’s description explicitly says April 6 while the field says April 16; the live interview’s description explicitly says April 14 while the field says April 10. The committed `recorded_at` values follow the explicit written descriptions and carry that provenance note. TED2013’s structured recording date also differs from the date of its [contemporaneous publisher blog report](https://blog.ted.com/transforming-transportation-elon-musk-at-ted2013/), so an exact recorded day is not asserted for that source.

Public page JSON was retained only in the ignored `data/raw/expansion-a` working cache for reproducibility of the cue extraction. These cache files are not source assets for redistribution.

### Y Combinator

The [publisher transcript](https://www.ycombinator.com/blog/elon-musk-on-how-to-build-the-future) opens with Craig Cannon’s podcast introduction, followed by Sam Altman’s interview. The named timeline is `publisher_podcast_transcript`. The observed video ID `e56PbkJdmZ8` comes from the publisher’s embed; the batch does not equate podcast times with video times.

### Dwarkesh Podcast

The [host’s page](https://www.dwarkesh.com/p/elon-musk) identifies the February 5, 2026 interview and contains the transcript and its chapter times. Its linked YouTube ID is `BYXbuik3dgA`. Because the page has chapter anchors rather than precise per-answer timing, committed locators name both the chapter and the specific exchange. They do not assign the chapter’s beginning as the onset of every later answer.

### Recode, WIRED, TIME, Popular Mechanics, and GQ

These records use publisher section or passage locators. Recode’s public JSON-LD verifies original publication on November 2, 2018 and an update on November 5. Its opening establishes that the conversation took place on Halloween night. The associated Vox Media podcast listing provides an acquisition entry point, but no numerical timing is copied from its chapter list into a passage.

The two TIME records represent distinct events: an interview at Boca Chica on December 3, 2021, published December 13; and a live interview with Edward Felsenthal on December 13, published December 14. Both pages expose publisher HLS links recorded in `coverage.json`. Their edited video durations and rough automatic transcripts do not cover or accurately transcribe every written quote, so records rely on the clearer edited interview passages. Unrelated promotional content returned above the article body is excluded.

Popular Mechanics explicitly says its February 2019 article continues the interview excerpted in January. Only **one event** is counted. A photograph caption referring to a different appearance is not treated as an interview video. An apparent transcription error about the Moon’s atmosphere is outside the selected evidence.

WIRED’s interviewer Chris Anderson is the former WIRED editor, a different person from TED’s Chris Anderson. GQ’s original October 2015 answers are separated from the reporter’s interpretations, earlier interviews reproduced in the feature, and observations at other events.

## Acquisition candidates excluded from reviewed evidence

The four `metadata_only_candidates` in `coverage.json` are not part of the 12-source total and must not be indexed as reviewed persona evidence:

- [Lex Fridman’s first Musk interview](https://lexfridman.com/elon-musk/): publisher date and MP3 are verified; a publisher transcript was not located in this pass.
- [Lex Fridman episode 252](https://lexfridman.com/elon-musk-3/): publisher date, chapter list, and MP3 are verified; chapter names alone do not support answer paraphrases.
- [2015 Reddit AMA thread](https://www.reddit.com/r/IAmA/comments/2rgsan/i_am_elon_musk_ceocto_of_a_rocket_company_ama/): public answers and a linked verification tweet are visible, but exact original date and stable comment-level provenance require another pass. A public JSON request returned HTTP403 and was not circumvented.
- [2016 Reddit AMA thread](https://www.reddit.com/r/spacex/comments/590wi9/i_am_elon_musk_ask_me_anything_about_becoming_a/): public replies remain visible while the introduction is removed. Date and comment provenance remain unverified.

A third-party index was used only to discover a Reddit thread URL; none of its paraphrases, inferred author traits, or transcripts are evidence records. Joe Rogan, Nikhil Kamath, other Code Conference appearances, and FT were not added without a reviewed primary answer source.

## Validation performed

All 12 source objects and 64 evidence records passed the repository’s `bias.models.Source` and `bias.models.Evidence` Pydantic validators. A direct integrity check also verified unique IDs, resolvable source references, no ID or source-URL overlap with the starter corpus, 4–8 records per source, explicit timelines for timed records, null behavioral observations, exact coverage counts, and every source-specific text budget.

This validation checks structure and consistency. It does not replace the source review described above and does not measure retrieval performance or the fidelity of a generated persona. Public accessibility of a source is provenance, not a license for full-text redistribution or a voice/likeness asset. The existing local media review pipeline should handle any acquired clip before it is used for voice or movement inference.
