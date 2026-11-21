# Source-grounded conditional values

The v1 pilot stored six broadly phrased principles. This revision makes the
question's circumstances part of the representation: a source-backed value or
belief informs a strategy with explicit applicability and exceptions. The new
corpus is optional and the default live pipeline is unchanged.

## What was collected and what was constructed

Three long publisher transcripts and four clips from one further event were
captured: Stanford GSB 2024, No Priors 2023, Milken 2026, and Stanford ETL 2009.
Acquired 2023 and Stripe Sessions 2024 were also re-read. The optional corpus
keeps all original 32 records intact and adds 28 bounded editorial paraphrases.
Its 16 source resources correspond to 13 events, not 16 independent interviews.

AI-assisted extraction produced candidate observations with the speaker,
surrounding question, source location, original-capture hash and experience
group. A separate source review checked paraphrases and removed unsupported
links. The result is edited structured memory, not a trained extraction model
or human-certified description of Huang's psychology. Original transcripts
remain outside public Git; their summaries cannot be used as verbatim quotes,
speech-style examples, aligned audio or measured facial behavior.

The graph separates **values**, **beliefs**, **interaction style**, and
**conditional strategies**. An `explicit_statement` is a paraphrase of an
expressed position, not an exact quotation. Broader interpretations carry a
different epistemic label. Source events and experience groups are counted
separately: repeating a CUDA story does not create two independent experiences.

## The twelve strategies

All applicability conditions and exceptions below are editorial applications.
They are not statements that Huang answered these hypothetical questions.

| Strategy | When it can help | Essential distinction | Primary support |
| --- | --- | --- | --- |
| Reassess before persisting | Continue or abandon an uncertain idea | Adverse reaction versus a refuted central premise | Stripe; Stanford GSB |
| Test before costly commitment | Scarce resources and testable uncertainty | Reduce testable risk; do not demand certainty or guarantee success | Acquired RIVA 128; No Priors |
| Choose need before building | Project scope or build versus reuse | Distinctive unmet need versus duplicating an adequate solution | Stanford GSB; No Priors |
| Match guidance to experience | Coaching and delegation | Novice support versus experienced ownership | No Priors; GSB; Stanford ETL |
| Organize around contribution | Responsibility, titles and useful work | Relevant capability versus prestige; retain accountability | GSB cleaning-work account; Acquired |
| Communicate under uncertainty | Team setbacks and difficult information | Explain knowns and unknowns without invented reassurance | Stanford GSB |
| Adapt process to project phase | Exploration versus reliable delivery | Bounded experimentation versus dependencies on a stable service | No Priors; ETL founding account |
| Allocate by opportunity cost | Competing uses of time, expertise or funds | Unique contribution and horizon versus headline opportunity | Stanford ETL; Milken |
| Deploy with accountability | AI deployment with consequential errors | Application-specific safeguards; dated views are not current regulations | Milken |
| Separate task from career | Automation and changing work | A profession's purpose versus one task; some jobs can still disappear | Milken |
| Disagree without losing the problem | Collaboration despite disagreement | Workable cooperation versus pretending endorsement or accepting harm | Milken |
| Judge leadership by fit | Role suitability and succession | Develop capability; incumbency does not establish suitability | No Priors; Stanford ETL |

For example, a disappointing pilot should trigger a check of its assumptions.
If repeatable evidence refutes the central premise, persisting solely from
conviction is inappropriate. If the premise remains plausible and the problem
is still unmet, a limited next experiment may be reasonable, subject to runway.
The graph must preserve both branches rather than repeating "never give up."

## Runtime behavior and limits

The existing advice/factual boundary runs first. A deterministic bilingual
matcher requires a match in every configured situation-cue group. Within each
group, alternative terms and conservative English inflections are accepted.
Rules are ranked by matched cues, then conjunction specificity, and at most two
are normally selected. This is a lexical relevance mechanism, not a learned
semantic interpretation of the situation.

Every selected strategy carries its linked claims, support, qualifications and
exceptions. The model must decide whether the conditions actually hold. The
existing context packer only includes complete source-backed units. If the
four-evidence budget cannot fit a secondary strategy, it is omitted rather than
stripped of its qualifications. Source bindings include persona, event,
experience group, text hash, metadata hash, review status and snapshot timing.

There is no extra online extraction or LLM call. Longer context can nevertheless
increase generation latency; a local request-time benchmark is not end-to-end
Mac, TTS or avatar latency. Explicit news/private-fact/project questions cannot
use this memory to manufacture missing evidence.

`BIAS_PERSONA_MEMORY_FORMAT=flat` renders the same nodes and relationship
meanings in prose. `graph` uses typed relationships. Neither performs graph
neural-network training, graph traversal reasoning by a separate model, or the
fine-tuning stage of ThinkPersona.

After the first frozen experiment, a v2-only response contract was added to
prevent narrating the graph instead of answering the question. It requests
practical advice followed by a relevant qualification or next step, without
internal identifiers or a rule-by-rule explanation. Existing adaptive/live
and stricter Korean length policies take priority. Otherwise the English
default is at most two short sentences, approximately 60 words, unless the
user explicitly requests another format. This is prompt guidance, not a hard
output cut; all source evidence, conditions and exceptions remain available.
The contract's token cost participates in normal whole-memory budget checks.

## Evidence and evaluation

The frozen comparison uses one pinned Qwen3-4B model and identical decoding and
answer budgets. It compares original card/RAG, expanded RAG, conditional flat
memory, and semantically matched graph memory. The twelve advice cases form six
context-contrast pairs; four further cases test unsupported/current/private
claims. Questions and scoring rules were fixed before answer generation.

The ordinary lexical RAG retrieved no evidence on these novel scenarios in the
frozen inputs. Therefore graph versus baseline includes source selection and
extra information. Flat versus graph isolates representation more directly;
it is not a test of whether a graph intrinsically causes faithful personality.
The pilot cannot separate conditional rules from all benefits of supplying
better-selected evidence, and it does not compare with real held-out Jensen
answers. No test pass, source count or retrieval hit is a persona accuracy score.

See [execution status](persona-values-v2-execution.md),
[frozen protocol](../reports/persona-rules-2026-11-21/protocol.md), and
[corpus setup](../data/personas/jensen-huang-values-v2/README.md).

The [completed first comparison](../reports/persona-rules-2026-11-21/README.md)
did **not** show an improvement: advice pass rate was 29.2% for the initial
graph versus 54.2% for the original card/RAG, including truncations as failures.
The output contract is evaluated separately as a development iteration on
already-seen scenarios. Neither result may be substituted for the other.

## Research basis

- [ThinkPersona](https://aclanthology.org/2026.acl-long.449.pdf): distinguish
  value-system decision factors, cognitive patterns and event influences;
  retain what is explicitly stated versus inferred. This project does not
  reproduce its synthesized reasoning data or SFT training.
- [SIBPersona](https://aclanthology.org/2026.findings-emnlp.745.pdf): separate
  memory, speaking style and situation-dependent response strategy, grounded in
  evidence pairs. Inferred internal states are not observed private emotions.
- [Generative Agents](https://arxiv.org/abs/2304.03442): retain original evidence
  pointers when forming higher-level reflections. This design inspiration does
  not transfer that paper's results to a real-person avatar.

Independent human source review and evaluation on unseen events/scenarios are
still needed before promoting this optional candidate to the live default.
