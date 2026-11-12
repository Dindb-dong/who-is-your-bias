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

Preparation is complete; no expansion media file has been acquired yet.
The next step is to verify the public CHM delivery and a complete official
commencement recording, then save the first actual source batch. Private raw
publisher metadata is under `reflection-expansion-v1/discovery/` in the shared
artifact directory. The existing lossless 24 MiB tar-chunk workflow in
`tools/persona_data/media_chunks.py` is the verified recovery path.
