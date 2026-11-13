# Jensen original dialogue, materialization v4

This is a new reproducible acquisition, not recovery of the interrupted run's
reported 617 pairs or 105 training answers. Nine publisher events yielded 554
question/answer exchanges: 551 complete attributed exchanges and three explicitly
marked excerpts. The factual projection admits 410 answers of at least eight
words; 551 complete answers enter the style projection before model token/length
filtering. No speech or motion eligibility is inferred from publisher text.

The source registry and event-family split registration are in
`data/acquisition/jensen-huang`. The two reserved test families (WEF Davos 2026 and
No Priors 2023) have not been indexed, adapted on, or acquired by ordinary build.
All nine sources exposed in PR #17 remain train/development families.

Full publisher documents, canonical turns, exact original answers, and local
runtime corpus are preserved in separate durable artifact packages. The public
recipe lists only hashes, IDs, counts, parser identity, and locators. Root run
artifact receipts identify the persistent packages.

## Corrections and evidence limits

- The Strange Loop publisher appended website topic tags to its final response.
  The exact known terminal tag block is removed and before/after hashes are in
  the extraction audit. Unverified four-part timecodes remain literal locators.
- YC switches between `Jensen Huang (guest):` and `Jensen Huang:`. Parser v2
  recognizes both explicit labels, while preserving Garry Tan turns. Earlier v3
  under-extracted this event; it is superseded for new training. The v4 recipe
  includes the recovered Sega question/answer and has a different content hash.
- Goldman Sachs legal boilerplate is excluded at its printed boundary; the
  original publisher PDF remains intact. Its official-host retry succeeded after
  the alternate catalog host returned HTTP 503. The failed attempt is preserved.
- Acquired host preamble is outside the selected interview body. No source
  paragraph is silently treated as a new labeled speaker.
- Review is structural agent review of publisher attribution and boundaries.
  Independent human factual/speaker quality scores remain ungraded. Publisher
  and automatic-transcript errors can remain in original wording.
- Neither source timestamps nor this dataset establish acoustic alignment.
  Every speech or visual interval needs separate source-clock and speaker review.

## Rebuild

With the separately preserved `source-combined-v2` raw input and the exact current
registry, choose fresh output/report directories:

```bash
PYTHONPATH=src:. python -m tools.persona_data build \
  --registry data/acquisition/jensen-huang \
  --acquisition /absolute/path/to/source-combined-v2 \
  --output /absolute/path/to/new-original-dialogue \
  --report /absolute/path/to/new-report
```

The acquisition package selection combines successful batch-01 sources with the
successful official-host Goldman PDF retry in batch-02; the earlier failure is
retained in its original immutable attempt log. Snapshot hashes in the recipe
must match before treating a re-download as the same dataset.
