# Source notes for conditional reaction and speaking style

Reviewed on 2026-11-21 by an AI source curator. This is a small, source-grounded
editorial delivery layer, not a trained personality model or a measurement of
Jensen Huang's internal state. The five observations in
[`persona_delivery.v1.json`](../data/personas/jensen-huang-values-v2/persona_delivery.v1.json)
use five existing factual evidence anchors from three interview events. The
Stanford GSB event supplies most of the observations; five observations do not
represent five independent sources.

## What was read

The curator inspected the original publisher speaker turns in the recovered
source archive, rather than inferring wording, humor or delivery from the public
editorial summaries alone. The archive is
`jensen-persona-source-capture-2026-11-21.zip`, SHA-256
`4b80acb5175ec9f1b6c8d388c369ab344f934bf2f3e52a2453c697a1ff2e3024`.
Its acquisition record and durable recovery information are in
[the v2 execution record](persona-values-v2-execution.md).

| Observation | Source and exact captured-text locator | Bound evidence |
| --- | --- | --- |
| Face uncertainty with the team | Stanford GSB, `gsb2024.txt`, lines 125–139: response to a question about motivating employees after a market-price decline, and the following exchange about visible leadership | `jensen-values-gsb-04` |
| Coach without displaying status | Stanford GSB, lines 148–153: response about reviewing colleagues' work and explaining reasoning; No Priors, `no-priors2023.txt`, lines 225–244: executive independence versus early-career mentoring | `jensen-values-gsb-06`, `jensen-values-np-03` |
| Disagree while retaining cooperation | Milken, `milken2026.txt`, lines 963–970, speaker timestamp 40:36: the opening of Huang's answer about cooperation with a partner despite different positions | `jensen-values-mil-06` |
| Explain one concrete step | Stanford GSB, lines 150–156: making reasoning understandable and giving colleagues context | `jensen-values-gsb-06` |
| Invited, low-stakes humor | Stanford GSB, lines 127–149: short self-directed asides during the leadership discussion, an explicitly retracted teasing aside, and a cleaning-work anecdote before returning to the substantive point | `jensen-values-gsb-02`, `jensen-values-gsb-04` |

These locators are line numbers in the hash-pinned private review text, not
audio-aligned timestamps. The Milken timestamp is a publisher transcript marker;
it has not been checked against audio. No raw transcript is newly added to Git,
and this layer contains no verbatim dialogue examples.

## Primary sources and capture integrity

- [Stanford GSB: Jensen Huang on How to Use First-Principles Thinking to Drive Decisions](https://www.gsb.stanford.edu/insights/jensen-huang-how-use-first-principles-thinking-drive-decisions), published 2024-04-25.
  - `gsb2024.html`: 276,063 bytes; SHA-256 `2c7f3b7db25844b14788092c9832c7e4fa77bcbf49ac913feb477219d52c037e`.
  - `gsb2024.txt`: 54,161 bytes; SHA-256 `9af4d688035bafd04f39f4ae2f822454c7a4fddbb598ce7abe5c36268df5b8b6`.
- [No Priors: Jensen Huang](https://sarah-guo.herokuapp.com/blog/jensenhuang), published 2023-04-20.
  - `no-priors2023.html`: 61,323 bytes; SHA-256 `477b8267b3e9e50a15bfd58dfa0bc94f6d5d41fe1c6c57465cb06f020090ae23`.
  - `no-priors2023.txt`: 48,625 bytes; SHA-256 `d507fd0cdf33b4e79f47ca03e82a2fc4baf4dedc019a88ecccfb499a3f04ba30`.
- [Milken Institute: Leading in the Age of AI](https://milkeninstitute.org/sites/default/files/2026-05/LeadingAgeAIAConversationNVIDIACEOJensenHuang_Transcript_GC26.pdf), recorded 2026-05-04; precise publication date unverified, availability observed 2026-11-21.
  - `milken2026.pdf`: 714,015 bytes; SHA-256 `60ff9415bc786753aa7e8ee3e10b1c3ff312f855ab4251ff43ae7365f22eda62`.
  - `milken2026.txt`: 45,707 bytes; SHA-256 `744a3673495d32de224f02c9a252fd22fd526e8588c3527b4f2e423edcbfdaf0`.

Each JSON anchor additionally pins the exact editorial evidence text and parsed
source metadata already in the corpus. Those five bindings were checked with
the existing `_validate_anchors` function. This prevents a changed source record
or evidence summary from silently keeping an old delivery interpretation valid.

## Observations versus application policy

The source observation and the generated-response instruction are separate.
For example, the GSB transcript contains humor even within a serious leadership
discussion. Restricting generated humor to an explicitly invited, low-stakes
demo or coding question is a conservative application decision. It is not a
claim that Huang only jokes in those circumstances.

Similarly, a short first sentence, one explanatory example, and one useful next
step are interaction choices for this live avatar. The source supports visible
reasoning, acknowledged uncertainty and civil disagreement; it does not prove
that this exact response template reproduces the person's behavior in every
situation. The ethical exceptions in the delivery layer are application
boundaries, not a complete inventory of Huang's ethical convictions.

Humor is optional and should be suppressed for distress, injury, discrimination,
private allegations, financial loss or safety failures even when the question
also includes a humor cue. The runtime must enforce its broader factual and
sensitive-topic guards before applying this layer. A lexical cue match only
selects a possible response style; it does not prove the stated condition holds.

No style observation authorizes a new first-person biography, a purported
quotation, a current event claim, or an assertion about another person's hidden
motives. Source observations are audit metadata, not additional factual support
for the user's question.

## What this does not establish

- There has been no human coding, inter-rater reliability study or acoustic
  validation of these observations.
- The text captures do not establish pitch, timbre, speaking rate, pause timing,
  laughter acoustics, facial expression or gesture. They cannot demonstrate a
  voice-cloning improvement.
- New hypothetical responses are not known interview answers. Appropriate
  reaction and delivery must be evaluated separately from source fidelity and
  conditional value application.
- The source passages above are construction material, not held-out evaluation
  data. Evaluation questions written later are authored transfer probes, not
  unseen real interviews.
