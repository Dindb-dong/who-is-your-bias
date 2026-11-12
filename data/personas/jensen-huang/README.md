# Jensen Huang retrieval corpus

The default studio selects `jensen-huang`. The corpus contains **32 editorial
evidence records from nine dated publishers' sources**. It includes eight NVIDIA
summaries and 24 paraphrased exchanges from six interview events dated 2023–2026.
Every interview entry retains a publisher URL, publication date, source locator
and acquisition ID. Unconfirmed recording dates remain empty. The projection
flags permit factual retrieval only. See the [acquisition catalog](../../acquisition/jensen-huang/source-catalog-v2.md).

## Primary sources

- [COMPUTEX 2024 keynote report](https://blogs.nvidia.com/blog/computex-2024-jensen-huang/), published 2024-06-02: acceleration and efficiency, the system roadmap, and physical AI.
- [Blackwell Ultra announcement](https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning), published 2025-03-18: Huang's attributed reasoning-compute statement, separately labeled NVIDIA product explanations.
- [COMPUTEX 2025 keynote report](https://blogs.nvidia.com/blog/computex-2025-jensen-huang/), page date 2025-05-18: AI factories, the software ecosystem, and robot simulation. The recording date remains unspecified in this seed.

The retrieval partition includes all nine sources. The interview entries use
editorial paraphrases of the speakers' exchanges. The source links point to the
publisher transcripts. Stripe's date refers to its official video release.

The [exact corpus snapshot](../../snapshots/jensen-huang-2026-11-12.json) uses a
2026 availability cutoff and a 2026 current-evidence window. Dated interview
statements can support attributed answers. Questions about current positions
require separately eligible evidence. When retrieval returns no evidence and
an LLM is configured, the studio marks its response as hypothetical before
streaming it. The prompt asks for conditional reasoning in third person and
preserves uncertainty about actual events and positions.

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
