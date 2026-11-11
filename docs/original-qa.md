# Original interview Q&A pilot

This implements R1–R4 of the [proposal alignment plan](proposal-alignment.md):
the B/C examples now retain an actual interviewer prompt and the respondent's
original answer. A summary is never labeled an original answer. The old
`data/personas/elon-musk` corpus and historical evaluation files remain available
for explicit reproduction.

## Two materializations of one versioned selection

| Materialization | Training | Development | Test | Intended use |
| --- | ---: | ---: | ---: | --- |
| Bundled bounded excerpts | 5 | 1 | 4 | Small, inspectable default without redistributing full interviews |
| Locally hydrated selections | 22 | 4 | 8 | Actual controlled generation and reference-answer experiment |

The full local corpus contains 34 selected exchanges: 32 complete contiguous
question/answer turns and two expressly partial selections. The partial training
selection ends before a further continuation; the partial test selection is the
first response before an interviewer acknowledgment and later elaboration. These
are not silently joined into an invented continuous answer. An interviewer prompt
may be a statement inviting a response rather than a grammatical question.

The ten bundled records are all marked `completeness: excerpt`. Their combined
question/answer wording is at most 25 whitespace-delimited words per original
source event. `Evidence.text` is derived from `question_answer.answer` at load
time so the JSONL does not duplicate that wording. These short excerpts cannot
substitute for the full experiment. Runtime health metadata exposes the selected
corpus version, materialization, complete/excerpt counts and covered situations.

Source text, timing, speaker attribution and situation annotations were reviewed
by an **AI agent** against publisher transcripts. They are not human ratings or
verified acoustic measurements. Publisher transcripts can contain errors.

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
python -m tools.qna hydrate --output data/local/qna/elon-musk-v1
BIAS_CORPUS_DIR=data/local/qna/elon-musk-v1 python -m bias.cli serve
```

The command fetches only the five ordinary, allowlisted publisher pages. It uses
no login, cookie, paywall bypass or source-media download. A cached public HTML
page or TED `__NEXT_DATA__` JSON can be supplied with a repeatable
`--raw SOURCE_ID=/absolute/path` argument. Use `--bounded` to reproduce the
committed excerpt materialization. Existing outputs are never overwritten.

Each selection pins the canonical transcript content digest, original question
and answer digests, exact contiguous cue IDs, review basis and event identity.
Unexpected source changes fail the operation and require a new reviewed recipe.
The output is validated before an atomic directory rename. Full transcripts and
hydrated source wording remain in ignored local storage. The repository retains
reproduction recipes and bounded excerpts, not a transcript redistribution.

The complete command was exercised against cached publisher inputs: it produced
22/4/8 records and matched every pinned original question/answer digest. This
verifies extraction and attribution bookkeeping; it does not establish that the
generated responses resemble Musk.

## Situation-aware retrieval

`situations.v1` defines `explanation`, `reflection`, `praise`, `challenge`,
`advice`, `humor`, `disclosure_request`, and `conversation`. These labels describe
an observable conversational request. They do not assert an internal emotional
state. Original records reject unknown or duplicated labels. Seven of the eight
labels occur in the complete training corpus; `conversation` has no reviewed
training example in this version.

The query classifier prioritizes specific requests over generic question words,
and English token boundaries prevent substring matches. B and C use the same
BM25 index over original questions, answers, and reviewed topics. C applies a
fixed 1.35 score multiplier to matching situations **after** the identical
topical-confidence gate. A situation match cannot make an irrelevant record
eligible. `qa-bm25-situation.v2` also requires a concrete reviewed topic anchor or
multiple topic matches; a lone generic word such as *future* or *risk* is
insufficient. This gate was revised using the four development questions,
before the independent test run. All four are unsupported by this narrow
training corpus under that gate, which is reported rather than disguised as
successful retrieval.

One measured development check asks how to reduce tunnel costs. B ranks the
cost-objection exchange first (4.724787), followed by the explanation of a
cost-reduction method (4.425883). C promotes that explanatory exchange to
5.974942 and first place. This demonstrates that the reviewed situation can
change a meaningful topical choice. It is a development regression, not an
independent persona-quality result. C need not differ from B on every question,
and a changed ranking is not automatically a better answer.
The [preserved development report](../data/qna/elon-musk-v1/retrieval-development.json)
records both rankings, input digests and the four unsupported development cases.

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
identical prompt instructions, complete-exchange budgets and timing integrity.
