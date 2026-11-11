# Observation pilot: artifact and version audit

Audit date: **2026-11-11**. This status applies to the preserved
[TED observation pilot](../../data/observations/ted-pilot.v1.json), not to all
interviews in the text corpus. See [measurement and review methods](../situation-observations.md)
for the extraction commands and interpretation.

## What has actually been verified

| Item | Verified result | Limit |
| --- | --- | --- |
| Source exchanges | Four original Q&A records; question/answer text matches the selected publisher cues and pinned text digests | Publisher labels are agent checked; no acoustic speaker review |
| Audio linkage | Four exchange-to-ASR links from three distinct cached runs; 19 contiguous speech-rate windows | ASR word-time estimates; all four whole-span rates remain unknown |
| Visual linkage | Ten distinct linked windows; 512 sampled frames, eight excluded frames and two smile candidates | Frame geometry is not a verified smile, laugh or gesture judgment |
| Source separation | TED2013 observation training; TED2017 observation validation | Both are text-training events; neither is an unseen text-test event |
| Review import | Untouched template reproduces four unreviewed exchanges and an empty runtime index | Zero accepted human observations, listening ratings or reviewed pauses |
| Reproduction | Rebuilt from the original publisher, audio, ASR, frame and geometry artifacts | Feature recomputation only; no new ASR, MediaPipe or synthesis run |

The preserved execution used 0.812537 s wall time, 0.804984 s process CPU time and
168,386,560 bytes of process high-water RSS. These values describe the original
feature recomputation, not model inference, end-to-end speech latency or the
intended demonstration machine. The artifact provenance verifies 13 visual
cache windows; three do not overlap the selected exchanges and are excluded from
the ten-window result count.

The audit recomputed the complete report from the original cache. With hydrated
Q&A v1, every field except the new execution measurements matched the preserved
report. With hydrated Q&A v2, the only additional difference was the whole-corpus
input digest. The original source question and answer strings for all four
records also matched their exact publisher cue selections and recipe hashes.
No source wording or media was added to these audit notes.

## Binding to Q&A v1 and v2

| Text materialization | Records | Train / validation / test | `evidence.jsonl` SHA-256 |
| --- | ---: | --- | --- |
| Hydrated `elon-musk-original-qa.v1` | 34 | 22 / 4 / 8 | `bb9276d5c1b270532f7e15eb4a43a89d4af791737e6f161504b179a25638a04a` |
| Hydrated `elon-musk-original-qa.v2` | 92 | 80 / 4 / 8 | `fe331ba1bfabcedac8ba2a3dc20595dbc4edba1b235918727f64ee768c4f5590` |

The canonical source-event split file is unchanged:
`d9ef0051c278e376074dbe8cc12de1c67658774930c32e0b5ab1a526f6355295`.
Each complete serialized evidence record below has the same canonical JSON
digest in both local materializations and in the observation report:

| Exchange | Unchanged `exchange_sha256` |
| --- | --- |
| `qa-ted-2013-energy` | `d3c6f01fefb7236e72ed9584f4b7036662d244c348854ffb541acbbade9f08b1` |
| `qa-ted-2013-efficiency` | `d367c837e97717df2349b4787b13754ff1ac7509b4e3942c7bdf045427898b5b` |
| `qa-ted-2017-tunnels` | `62f70828e94fbcb37cf9be4e43ab6e1813da3443f83ac2960e06a9d3d32d366b` |
| `qa-ted-2017-capacity` | `719547115a063bf34214e7696b1c6398ac0c43d8fddc23c2739bd5204fb832a9` |

These digests use `tools.references.workflow.digest_json` on the JSONL record,
before model loading adds derived/default fields. They are not hashes of
rendered quotations or normalized model objects.

The four existing measurements remain attributable to the same exchanges in v2.
**The preserved report remains a v1 run.** Its corpus hash, report ID and review
binding must not be rewritten to imply that all 92 exchanges were measured.
A newly built report, including a rerun on v1, has new execution metadata and a
new byte hash; it requires a newly generated review bundle. Existing annotations
are bound to the exact original report, not merely to matching record IDs.

## Preserved file inventory

There are six files under `data/observations` in this pilot. The audit verified
their byte hashes; their contents were not regenerated or relabeled.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [Plan](../../data/observations/ted-pilot-plan.v1.json) | 1,792 | `05480b912a71f51893402bccf6141e56637c7274a696f4da16a2eae17d505e6e` |
| [Measured report](../../data/observations/ted-pilot.v1.json) | 68,433 | `582333582827dd2af2d81a7452025915d2d01d469e564afd96f83b9c8565f60b` |
| [Review page](../../data/observations/ted-pilot-review/index.html) | 3,567 | `afd9fcb91866788e66572c8a3cce258d158788a79944dd89d4d51a21777e0105` |
| [Blank template](../../data/observations/ted-pilot-review/review-template.json) | 3,458 | `a59a185d156015845345a9a8647a316861d441c5b8b33a7bd938c7c10448bbcb` |
| [Ungraded summary](../../data/observations/ted-pilot-review/ungraded-summary.json) | 1,180 | `bb8a14d5789c93c46cbe98316ca7ac5f964a8eb01890c06cb10f9b40911935fd` |
| [Agent visual screening](../../data/observations/visual-screening.v1.json) | 2,290 | `617cc3f2d94b5353c36286204750a2054fb66c780e45fafd90815abadb52f10c` |

The template and ungraded summary both bind to the measured report's byte hash.
The screening record documents two contact sheets inspected by an agent. It
expressly records no audio listening, continuous-track verification or
image-to-speaker verification. Opening the review page does not change those
states.

## Review safeguards and outstanding evidence

The importer checks the report hash, per-exchange digest and source event. It
requires the modality-specific review checks, respects observed intervals and
camera exclusions, and admits a pause only when all three reviewed acoustic
boundaries lie in one complete audio run. The additional regression cases reject
video-only acoustic timestamps, pauses spanning separate audio runs and complete
boundary coverage borrowed from a different run. All **31 observation tests**
passed after that correction. Importing the preserved blank template still
produces the exact same ungraded summary.

These structural checks cannot establish that someone actually listened or
watched. Human judgments must be supplied and identified as such. Accepted
agent judgments cannot enter the runtime index; neither can validation/test
events. The existing source-geometry preview library is a separate artifact and
does not become human reviewed because this pipeline exists.

R8 is therefore a measured pilot with review tooling, **not completed human
behavior validation**. TED2022 recorded and both Lex test events remain
unmeasured. The pilot contains no Korean source rate measurements, accepted
audible laughter, semantic nod/tilt or hand gestures, validated pre-answer
pauses, or learned reaction policy. Naturalness, voice resemblance and perceptual
lip synchronization remain separate listening/viewing outcomes. More text
coverage, ASR agreement, container timing checks and browser tests cannot fill
those missing judgments.
