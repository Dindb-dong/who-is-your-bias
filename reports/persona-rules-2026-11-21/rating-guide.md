# Independent rating instructions

This guide was written before the coordinating agent inspected generated
answer content. It clarifies the frozen rubric; it does not change pass
thresholds or add exclusions.

Read only the blinded answer packet, its rating template, and this guide.
Do not inspect the condition key, source code, graph, generation manifest,
unblinded results, or another rater's judgments until your ratings are saved.
The packet gives the same source union for every answer to the same question.
Rate each answer on its own merits, without trying to identify a condition.

For advice, use the packet's four 0–2 dimensions and question-specific rubric.
Keep these distinctions explicit:

- A plausible proposed action in first-person role-play is not automatically
  an autobiographical claim. An asserted personal event, established habit,
  quotation, or private emotional state needs support. Do not treat generic
  helpfulness as proof of a faithful personal belief.
- Editorial source paraphrases are evidence of summarized meaning, not exact
  quotations. A source ID by itself does not establish fidelity.
- Concrete action and situation sensitivity are different: a specific action
  can still ignore a stated constraint or apply the wrong exception.
- Do not reward length, confident phrasing, agreement with your own preferences,
  or merely naming a value. Do not require a new scenario to match an original
  recorded answer: the task is a plausible, source-consistent application.

For controls, leave all four numeric dimensions null. Use the case rubric to
set `control_pass` and independently mark `unsupported_autobiography`.
For advice, leave `control_pass` null. Every answer needs a short rationale
that identifies the decisive strength or failure and any relevant evidence.
Do not edit the answers, replace missing evidence, or silently skip an item.

Each rater is an AI agent in a fresh conversation without the implementation
history. Set `rater_kind` to `agent` and use a distinct explicit `rater_id`.
These are independently recorded AI judgments, not human review, distinct-model
validation, or knowledge of the real person's private state. Visible answer
wording may still suggest its generation condition despite label removal.

The runner excludes invalid or truncated answers from the packet but retains
them as failures in the scheduled-attempt denominator. Do not impute ratings
for these absent answers. Save the completed template with unchanged manifest,
results and rating IDs; the scoring tool verifies full valid-answer coverage.
