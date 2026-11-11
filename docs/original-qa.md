# Original interview Q&A pilot

This implements R1–R4 of the [proposal alignment plan](proposal-alignment.md):
the B/C examples now retain an actual interviewer prompt and the respondent's
original answer. A summary is never labeled an original answer. The old
`data/personas/elon-musk` corpus and historical evaluation files remain available
for explicit reproduction.

## Bounded default and versioned full selections

| Materialization | Training | Development | Test | Intended use |
| --- | ---: | ---: | ---: | --- |
| Bundled bounded excerpts | 5 | 1 | 4 | Small, inspectable default without redistributing full interviews |
| Full local v1 (preserved) | 22 | 4 | 8 | Initial selection and observation bindings |
| Full local v2 | 80 | 4 | 8 | Source-wide training coverage for the controlled reference-answer experiment |

The recommended full local v2 contains 92 selected exchanges: 84 complete
contiguous question/answer turns and eight expressly partial selections. It
preserves all 34 v1 selections exactly, including every development and test
record. Ambiguous continuations are not silently joined into an invented
continuous answer. An interviewer prompt may be a statement inviting a response
rather than a grammatical question.

The ten bundled records are all marked `completeness: excerpt`. Their combined
question/answer wording is at most 25 whitespace-delimited words per original
source event. `Evidence.text` is derived from `question_answer.answer` at load
time so the JSONL does not duplicate that wording. These short excerpts cannot
substitute for the full experiment. Runtime health metadata exposes the selected
corpus version, materialization, complete/excerpt counts and covered situations.

Source text, timing, speaker attribution and situation annotations were reviewed
by an **AI agent** against publisher transcripts. They are not human ratings or
verified acoustic measurements. Publisher transcripts can contain errors.

### Source-wide training coverage in v2

Before running held-out inference, the training selection was expanded by a
source-wide rule across the entire TED2013 and TED2017 transcripts. A pair must
have explicit publisher labels at both starts, directly adjacent interviewer and
respondent turns (allowing standalone stage directions), and either a question
mark or at least eight interviewer words and eight respondent words. Every pair
meeting that rule is included. The [complete training inventory](../data/qna/elon-musk-v1/training-inventory.v2.json)
records the status of every interviewer turn without distributing source text.

| Training event | Interviewer turns | Eligible Q&As | Brief nonquestions excluded | Fragmented starts excluded | No unambiguous answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| TED2013 | 39 | 28 | 3 | 2 | 6 |
| TED2017 | 74 | 52 | 16 | 1 | 5 |

The 58 additional records cover later solar, financing, autonomy, trucks,
battery manufacturing, rocket reuse, Mars and motivation discussions as well as
earlier technical follow-ups. Newly added records retain the preceding explicit
interviewer turn as original context when available, with its own cue IDs and
digest. They do not claim coverage of all possible questions about those topics.
The inclusion rule used training transcripts only; no held-out generation or
held-out score was used to select the expansion. The original v1 selection,
split manifest and development report remain unchanged.

## Source events and evaluation separation

The [split manifest](../data/qna/elon-musk-v1/split-manifest.json) was fixed before
selecting this pilot's exchanges or adjusting its retrieval parameters.

| Event | Split | Publisher source |
| --- | --- | --- |
| TED2013 | train | [Original interview](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript) |
| TED2017 | train | [Original interview](https://www.ted.com/talks/elon_musk_the_future_we_re_building_and_boring?view=transcript) |
| TED2022 recorded | validation | [Original interview](https://www.ted.com/talks/elon_musk_a_future_worth_getting_excited_about?view=transcript) |
| Lex #400, 2023 | test | [Publisher transcript](https://lexfridman.com/elon-musk-4-transcript/) |
| Lex #438, 2024 | test | [Publisher transcript](https://lexfridman.com/elon-musk-and-neuralink-team-transcript/) |

The two test interviews are complete event holdouts, including other publications
of the same recording. Only Musk's explicitly attributed turns are selected from
the multi-guest Lex #438 transcript. Known event IDs, canonical duplicate URLs,
and the frozen source registry are checked before indexing. Newly discovered
mirrors require event review; a different URL does not establish independence.

These sources had been inspected in earlier project work. The selections are a
small, hand-curated **development pilot**, not a preregistered blind study, a
random sample, or evidence of broad persona generalization. Do not mix the old
summary corpus into this study: its historical splits differ, and it includes
material from events withheld here. The default index admits only reviewed,
source-grounded records from this corpus's train events. Test questions and
reference answers are evaluator data, never runtime retrieval material.

## Reproduce the full local corpus

Run from the repository root after installing the base Python dependencies:

```bash
python -m tools.qna hydrate \
  --selection data/qna/elon-musk-v1/selection-recipes.v2.json \
  --output data/local/qna/elon-musk-v2
BIAS_CORPUS_DIR=data/local/qna/elon-musk-v2 python -m bias.cli serve
```

The command fetches only the five ordinary, allowlisted publisher pages. It uses
no login, cookie, paywall bypass or source-media download. A cached public HTML
page or TED `__NEXT_DATA__` JSON can be supplied with a repeatable
`--raw SOURCE_ID=/absolute/path` argument. Omitting `--selection` preserves the
original v1 selection; use that default with `--bounded` to reproduce the
committed excerpts. Existing outputs are never overwritten. Hydration never
silently switches the runtime corpus: `BIAS_CORPUS_DIR` is an explicit choice.

Each selection pins the canonical transcript content digest, original question
and answer digests, exact contiguous cue IDs, review basis and event identity.
Original preceding context, where present, is independently digest-checked.
Unexpected source changes fail the operation and require a new reviewed recipe.
The output is validated before an atomic directory rename. Full transcripts and
hydrated source wording remain in ignored local storage. The repository retains
reproduction recipes and bounded excerpts, not a transcript redistribution.

Both selections were exercised against cached publisher inputs: v1 produced
22/4/8 records and v2 produced 80/4/8, matching every pinned original
question/answer digest. This
verifies extraction and attribution bookkeeping; it does not establish that the
generated responses resemble Musk.

## Situation-aware retrieval

`situations.v1` defines `explanation`, `reflection`, `praise`, `challenge`,
`advice`, `humor`, `disclosure_request`, and `conversation`. These labels describe
an observable conversational request. They do not assert an internal emotional
state. Original records reject unknown or duplicated labels. All eight labels
occur in v2 training; the narrower v1 training selection covers seven.

The query classifier prioritizes specific requests over generic question words,
and English token boundaries prevent substring matches. B and C use the same
BM25 index over original questions, answers, and reviewed topics. C applies a
fixed 1.35 score multiplier to matching situations **after** the identical
topical-confidence gate. A situation match cannot make an irrelevant record
eligible. The topical floor introduced in `qa-bm25-situation.v2` requires a concrete reviewed topic anchor or
multiple topic matches; a lone generic word such as *future* or *risk* is
insufficient. This gate was revised using the four development questions,
before the independent test run. All four remain unsupported after the v2
training expansion under this conservative lexical gate. This is a measured
coverage limitation rather than evidence of successful retrieval.

`qa-bm25-situation.v3` preserves that gate and adds ordinary Korean aliases for
tunnels, traffic, costs and innovation only in original-Q&A mode. All six
bundled starter prompts (three English and three Korean) retrieve relevant
bundled excerpts. Archived summary-corpus tokenization is unchanged.

One preserved v1 development check asks how to reduce tunnel costs. B ranks the
cost-objection exchange first (4.724787), followed by the explanation of a
cost-reduction method (4.425883). C promotes that explanatory exchange to
5.974942 and first place. This demonstrates that the reviewed situation can
change a meaningful topical choice. It is a development regression, not an
independent persona-quality result. C need not differ from B on every question,
and a changed ranking is not automatically a better answer.
The [preserved development report](../data/qna/elon-musk-v1/retrieval-development.json)
records both rankings, input digests and the four unsupported development cases.
Its numerical scores belong to v1 and are not presented as v2 results.

This is a transparent lexical baseline. It does not claim to be a neural
semantic retriever or a learned model of the person's reactions. Ambiguous
questions such as a bare *How?* require conversational context; the index does
not invent the missing subject.

## Prompt and timing contracts

`original-qa.v1` gives A/B/C identical system instructions and the same profile.
Only the selected original exchanges differ. C gets no additional personality,
emotion or animation instructions. Each example identifies the original
question/answer, context kind, language, date and completeness. An example that
exceeds the declared context budget is dropped whole, never silently truncated.
The reference-evaluation runner records exactly which examples reach the model.

Question and answer **starts** are publisher cue timestamps on the named source
clock. End proxies are stored separately from acoustic end boundaries, which
remain null. No response delay or speaking rate can be computed as a verified
measurement from those proxies. TED and YouTube clocks are not interchangeable.
The single selected unlabeled Lex continuation was reviewed explicitly and is
listed in its recipe; the general parser leaves unassigned speakers unknown.

Run the focused regressions with:

```bash
python -m pytest tests/qna -q
```

They cover changed-source rejection, speaker ambiguity, frozen event splits,
held-out exclusion, shared labels, meaningful reranking, topical abstention,
identical prompt instructions, complete-exchange budgets, timing integrity,
preserved v1 selections, source-wide inclusion counts and supported bilingual
starter prompts.
