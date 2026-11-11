# Free Musk avatar acquisition follow-up

Checked **2026-11-11 at 14:05 UTC**. This follow-up covers the three free Sketchfab listings in the [shortlist](research/musk-3d-avatar-shortlist.md). It records current public metadata and the result of each official download request. **No model package was acquired, imported or qualified.**

## Result and exact blocker

All three public model endpoints returned HTTP 200 and declared `isDownloadable: true` and `isProtected: false`. Each documented `/v3/models/{uid}/download` endpoint returned **HTTP 401** with this response:

```json
{"detail":"Authentication credentials were not provided."}
```

The requests supplied no account credentials. Public metadata access and the absence of a model password do not provide authenticated download access. The official [download instructions](https://sketchfab.com/developers/download-api/downloading-models) require a signed-in Sketchfab user and return temporary archive links after an authenticated request. The [Download API overview](https://sketchfab.com/developers/download-api) describes glTF, GLB and USDZ delivery; it does not promise the creator's original FBX or Blender source package.

Direct listing retrieval through the web retrieval service returned HTTP 403. The documented public Data API was independently reachable with ordinary unauthenticated HTTPS requests. These are different observed outcomes: the concrete package blocker is the download endpoint's HTTP 401, not an inference from the listing-page error.

The [acquisition report](../reports/avatar-acquisition/2026-v1/acquisition-report.json) retains request URLs, local UTC timestamps, server Date headers, response statuses, SHA-256 digests and the exact public metadata/download-error bodies. The server Date headers and local retrieval time are separate fields; a response may have been cached. No login, signup, purchase, seller contact or viewer-asset extraction was performed.

## Newly confirmed public metadata

These counts are **Sketchfab API metadata**, not measurements of acquired geometry. The field `faceCount` is reproduced as an API face count; animation count does not establish facial morphs, skinning, independent gaze or lip-sync to new speech.

| Candidate | API face / vertex counts | Animation / material / texture counts | Published / updated dates reported by API | License recorded by API |
| --- | --- | --- | --- | --- |
| [LasquetiSpice — Talking Elon Musk Avatar Character for Narration](https://sketchfab.com/3d-models/talking-elon-musk-avatar-character-for-narration-8ede6f6ead9b4dc79ea51c050fefb594) | 16,668 / 8,542 | 1 / 1 / 2 | 2022-02-15 / 2022-07-18 | CC Attribution; exact URL points to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Vadim Fedenko — Elon Musk Low Poly](https://sketchfab.com/3d-models/elon-musk-low-poly-e647578649c34e29be81c8f94ecaabd2) | 25,202 / 17,719 | 1 / 10 / 3 | 2020-05-24 / 2022-07-20 | CC Attribution; exact URL points to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [KingOfThePirates — Elon Musk](https://sketchfab.com/3d-models/elon-musk-8406baac0f634d5084a7a33efa296dd1) | 50,000 / 25,000 | 0 / 1 / 1 | 2024-06-28 / 2024-06-28 | Free Standard; exact URL points to [Sketchfab licenses](https://sketchfab.com/licenses) |

The KingOfThePirates account's current display name is `ᗰOᑎKEY ᗪ. ᒪᑌᖴᖴY`; the stable username and model UID are retained separately. The metadata did not provide a NoAI field or a complete selected-item agreement. Their absence is not evidence of unrestricted usage. No 2026 likeness, ownership provenance, detailed mouth controls or deployment permission has been established by this metadata check.

Primary metadata endpoints:

- [LasquetiSpice public model record](https://api.sketchfab.com/v3/models/8ede6f6ead9b4dc79ea51c050fefb594)
- [Vadim Fedenko public model record](https://api.sketchfab.com/v3/models/e647578649c34e29be81c8f94ecaabd2)
- [KingOfThePirates public model record](https://api.sketchfab.com/v3/models/8406baac0f634d5084a7a33efa296dd1)

## Concrete next acquisition and qualification step

The first free package to inspect is **LasquetiSpice's speaking character**. Its smaller declared geometry, one animation and explicit CC BY 4.0 link make it a practical acquisition test. This is an inspection order based on listing metadata, not a likeness or facial-rig recommendation. Fedenko is the second animation candidate; KingOfThePirates remains a static-appearance candidate until its actual rig is inspected. Zero animation clips does not prove that a file has no bones or morphs.

1. An authorized user signs in to Sketchfab, opens the exact model listing above and uses its official download action. Retain the actual selected license/attribution, download date and original archive. There is no need to supply account passwords or tokens in chat. The authenticated Download API is an alternative for an existing authorized integration; its short-lived archive URL must not be committed.
2. Supply that original downloaded archive to this project locally, with its license evidence. Preserve it under an ignored `data/local/avatar-candidate/` directory. Record its SHA-256 before extraction and keep the source package unchanged. A thumbnail or embedded viewer cannot substitute for the package.
3. Follow the existing [provenance and conversion procedure](avatar-audition.md#1-register-and-convert-a-local-package). Preserve buffers and textures from a glTF archive. If conversion is needed, use the existing `tools/avatar_convert.py` procedure to produce an embedded GLB and a conversion report.
4. Run the existing inspector against the actual GLB, using the acquired package's provenance record:

```bash
npm ci --prefix tools/avatar --ignore-scripts

node tools/avatar_audition.mjs inspect \
  data/local/avatar-candidate/candidate.glb \
  data/local/avatar-candidate/qualification.json \
  --provenance data/local/avatar-candidate/provenance.json \
  --audio apps/web/public/presentation/narration.wav
```

5. Inspect the real bones, skin weights, morph displacement and animation tracks. Determine whether the speaking loop can be disabled and whether independent jaw, lips, eyelids and gaze controls exist. Then perform the same-audio browser audition, English/Korean mouth checks, interruption checks and dated appearance comparison described in the [shortlist qualification procedure](research/musk-3d-avatar-shortlist.md#5-audition-procedure-for-the-application). Keep the candidate at `inspection_only` until that evidence exists.

The next required input is therefore an **officially downloaded local package with retained license evidence**. The application already has conversion, structural inspection and browser audition paths; this follow-up does not replace the current proxy or certify a new persona asset.
