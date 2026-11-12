# Per-track coverage and probe cleanup

Both regional source acquisition processes completed using the original
collector published in `9a6bada20771e1e563c76ac4abeb2d208ec9e359`, whose source
SHA-256 is `1ec3f127ae5d592762133dc61c0d2fe76a9bb1213811ae0f598bfb8ffa1fd2a9`.
Their original receipts and implementation identity remain the evidence for
those runs. This follow-up was applied after both processes exited successfully.

A bounded code review found that an internally continuous but shortened audio
track could pass the original packet-gap check while video supplied the full
container duration. The collector now also checks each track's first and last
decode timestamps against the registered VOD extent, normalized to the earliest
track timestamp, with an explicit 250 ms endpoint tolerance for packet/frame
rounding. A full video track cannot mask a missing audio tail or delayed start.
The original packet walk's backward-step threshold is explicitly reported as
1 ms; its maximum permitted internal gap remains 250 ms.

The probe is also closed and reaped when packet parsing fails early. Termination
uses a five-second wait, followed by a kill and another bounded wait if needed.

The 21 focused tests cover the registered HLS integrity rules, normal per-track
rounding, a single/short/delayed/overlong audio track, invalid durations, and
parser-error termination with the kill fallback. The tests use small synthetic
values and a fake probe process. No neural model or source-media decode was
performed by this test suite.

Coverage checks derived from the saved actual acquisition timestamps are a
separate post-acquisition validation. They do not retroactively change which
implementation executed the acquisitions.
