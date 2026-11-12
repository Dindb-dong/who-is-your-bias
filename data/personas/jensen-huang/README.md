# Jensen Huang source seed

The default studio now selects `jensen-huang`. This directory contains **three
NVIDIA publications and eight editorial summaries**, checked on 2026-11-12.
They are dated 2024–2025 and retain the publisher's attribution. They are not
original interview Q&A, a trained style adapter, a speech reference or measured
behavior. The projection flags permit factual retrieval only.

## Primary sources

- [COMPUTEX 2024 keynote report](https://blogs.nvidia.com/blog/computex-2024-jensen-huang/), published 2024-06-02: acceleration and efficiency, the system roadmap, and physical AI.
- [Blackwell Ultra announcement](https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning), published 2025-03-18: Huang's attributed reasoning-compute statement, separately labeled NVIDIA product explanations.
- [COMPUTEX 2025 keynote report](https://blogs.nvidia.com/blog/computex-2025-jensen-huang/), page date 2025-05-18: AI factories, the software ecosystem, and robot simulation. The recording date remains unspecified in this seed.

All are in the retrieval training partition. No held-out evaluation, current
2026 position coverage, independent product benchmark, acoustic timing or human
quality evaluation is claimed. Summary text is newly written, not a reproduced
transcript. The complete publications are linked rather than redistributed.

The [exact corpus snapshot](../../snapshots/jensen-huang-2026-11-12.json) uses a
2026 availability cutoff and a 2026 current-evidence window. A question about
"currently" or "latest" therefore abstains instead of treating these historical
statements as current. Explicitly dated and general questions can retrieve them.

## Identity boundaries

The Musk corpora, snapshots, measured clips, adaptation and presentation reports
remain historical artifacts. They are not renamed or copied into this seed.
Runtime motion requires a matching declared persona, or source/event/media
membership in the active corpus for frozen catalogs without a subject field.
The existing Musk library does not satisfy that condition for Jensen.

Server speech additionally requires `BIAS_TTS_PERSONA_ID=jensen-huang` after
configuring a matching voice, or `BIAS_TTS_PERSONA_ID=generic` for a non-personal
voice. An unbound or mismatched endpoint is unavailable for playback. A request
cannot switch to a different voice name without updating its server binding.
These fields record operator declarations; they do not verify a speaker by
biometrics. Browser speech remains a generic alternative.

The displayed rig and eyes-then-head interaction are designed prototypes. This
change does not qualify a Jensen likeness or finish the photo/video avatar
generation work described in PR #15.

Historical reproduction uses explicit paths:

```sh
BIAS_CORPUS_DIR=data/qna/elon-musk-v1 \
BIAS_PERSONA_SNAPSHOT=data/snapshots/elon-musk-2026-11-11.json \
uv run bias serve
```
