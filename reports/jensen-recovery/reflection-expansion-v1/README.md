# Reflective-source expansion, separate from frozen v4

This additive work addresses PR18 Section 5's source-format and value-dimension
coverage. It does not modify the frozen v4 corpus, text-adapter-v1 training data,
M6 questions, evidence selection, prompts, or implementation hashes.

## Actual evidence coverage before expansion

[coverage-matrix.json](coverage-matrix.json) maps six dimensions to 14 illustrative
anchors from 12 distinct original Q&A records. Each anchor contains its exact
question, answer, record, and original-source hashes, source URL, event family,
transcript locator, eligibility, and reading-supported limitation. The matrix is
an agent review, not a human quality grade, exhaustive coverage score, or claim
that every eligible record was selected in the smaller training export.

| Dimension | Existing original sources represented |
| --- | --- |
| Failure and recovery | YC Startup School; Acquired |
| Work and purpose | Stanford View From the Top; Acquired |
| Leadership and information | Stripe Sessions; Lex Fridman |
| Judgment and conviction | Stripe Sessions; Acquired |
| Responsibility and obligations | Lex Fridman; Stanford View From the Top |
| Technical explanation | Acquired; Lex Fridman; Goldman Sachs |

The existing artifacts contain reflective interview answers and four complete
keynote recordings. Before this expansion, they contain no acquired formal oral
history or commencement recording. Reflective interview answers therefore do
not close those two format gaps. The frozen source metadata also retains some
legacy editorial-attribution prose despite identifying original publisher Q&A;
the matrix records that inconsistency without changing the sealed corpus.

## Acquisition preparation

[source-plan.json](source-plan.json) records four official sources and their
current access state. The Computer History Museum catalog exposes an original
MP4 and transcript PDF for its 2024-08-09 oral history. Caltech's public original
ceremony has a publisher chapter for Jensen Huang's 2024 address. NTU's linked
2023 YouTube video is private; its separate public university FTP directory
offers the full ceremony ISO. Carnegie Mellon's original 2026 address is an
additional official player lead. A catalog entry or metadata probe is not an
acquired recording.

Only public publisher pages, players, and download links are used. Acquisition
will proceed one full source at a time after the preparation checkpoint is
remotely saved. Actual bytes will be probed and hashed, saved in bounded chunks
when needed, then freshly read back and reconstructed before a full recording
is marked durable. The artifact index will retain stable storage IDs and full
source-clock provenance; it will not publish expiring delivery URLs.

All newly inspected material belongs to this separate expansion and is excluded
from the current sealed experiments. Any later corpus use requires its own
review and train/development assignment. These inspected sources must not be
reintroduced as unseen heldout material.

## Resume boundary

The preparation checkpoint was remotely verified at
`f2cf6bad53493aae7522c5b63612e654b850365d`. The complete CHM oral-history MP4
and original PDF are now acquired, saved, and freshly read back. All 29 stored
artifacts match their byte counts and SHA-256 values. The 26 restored video
chunks reconstruct the complete 653,931,658-byte recording exactly;
[media-artifacts.json](media-artifacts.json) records every stable identifier and
the recovery recipe. [acquisition-status.json](acquisition-status.json) records the actual
bytes, source clock, decoder checks, and remaining commencement gap. The next
step is to acquire a complete official commencement recording, with DVD
metadata/range inspection as a bounded fallback if public video players remain
unavailable. No full ceremony ISO has been downloaded. Private raw
publisher metadata is under `reflection-expansion-v1/discovery/` in the shared
artifact directory. The existing lossless 24 MiB tar-chunk workflow in
`tools/persona_data/media_chunks.py` is the verified recovery path.

## Bounded NTU address extraction

[ntu-extraction-plan.json](ntu-extraction-plan.json) records the exact ordinary
HTTP byte range selected from the university's public ceremony ISO. The DVD has
one program and one cell for the whole ceremony; it has no separate publisher
chapter for Jensen. Twelve small MPEG samples were inspected with their actual
PTS clocks. The selected range starts while the vice president is still at the
lectern and ends after the ceremony announcer has taken over. It is intended to
include the complete address and surrounding ceremony, subject to a further
continuous opening/closing review after acquisition.

The separate collector `tools/persona_data/archive_range.py` refuses a server
that ignores the requested range before consuming the response body. It checks
the exact Content-Range, total size, observed source identity, byte count, and
hash; it checkpoints each 16 MiB append. An explicit `--resume` verifies the
saved prefix and discards only an uncheckpointed interrupted append. Example:

```sh
python -m tools.persona_data.archive_range \
  reports/jensen-recovery/reflection-expansion-v1/ntu-extraction-plan.json \
  /absolute/artifacts/ntu-commencement-2023-address-with-context.vob
```

The receipt labels the result as derived archive bytes, keeps the original DVD
program-stream PTS clock, and makes no full-ISO claim. Decoder checks and a hash
do not establish semantic speech completeness; that remains a separately
recorded agent review. No YouTube-to-DVD clock offset is assumed. This work is
outside every frozen training and evaluation input.
