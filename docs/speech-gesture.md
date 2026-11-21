# Speech-driven gestures (EMAGE)

The studio can move the avatar's torso, arms and fingers from the synthesized
voice. A pretrained EMAGE model generates SMPL-X motion from the finished audio
of each response. The browser then plays it on the avatar rig, using the audio
clock. The motion follows the rhythm of the speech. It does not follow the
meaning of the answer, and it is not a measurement of the persona's own
gestures.

## Flow

1. With **Speech-driven gestures (EMAGE)** enabled, `POST /api/speech/prepare`
   receives `"gesture": true`. The toggle is in Settings → Conversation
   movement, or add `?gesture=emage` to the URL.
2. The server synthesizes the audio, then runs lip sync and EMAGE together.
   The extra latency is whichever of the two is slower. EMAGE took about 0.27×
   the audio duration on a laptop CPU.
3. The response gains a `gesture` object:

   | Field | Meaning |
   | --- | --- |
   | `status` | `available`, `not_configured`, `unsupported_audio`, `unavailable`, `failed` or `timeout` |
   | `message` | Reason for a fallback status |
   | `motion_url` | Link to the motion data. It lives in the same short-lived clip store as the audio. |
   | `frames`, `fps` | Number of frames, at 30 fps |
   | `width` | 165 values per frame |
   | `format` | Little-endian float32 SMPL-X axis-angle (55 joints) |
   | `audio_sha256` | The audio the motion belongs to |

4. The browser checks that the motion matches the audio. It then applies the
   team's calm filter (calm 0.7, smooth 3 frames, the same as
   `tools/gaussian_head/smplx/preview.html`). Finally it converts the motion
   (`apps/web/src/avatar/speechGesture.ts`):
   - spine 1–2 → `spine`, spine 3 → `chest` (both at 60%)
   - collars, shoulders, elbows, wrists → shoulder / upper arm / forearm / hand
   - the 30 finger joints → the matching finger controls
   - head, neck, jaw and the pelvis/legs are left alone. Head direction stays
     with attention and the Gaussian head, and the speaker stays in place.

   Each joint becomes a change from an SMPL-X reference pose whose arms point
   along the rig's relaxed neutral. EMAGE arms hanging down therefore leave the
   avatar at its neutral. When a rig lacks a bone (the stylized portrait has
   no collar or spine bones), that bone's rotation is passed to the nearest
   bone below it. EMAGE lowers the arm partly through the collar, so without
   this the portrait's arms would swing out. The rig's 70° per-joint limit
   still applies. The
   motion fades in over 250 ms and out over 350 ms, and drops to 15% with
   reduced motion.

Any failure keeps the verified audio playable with designed movement. While
gestures are enabled, streaming (PCM) voice switches to per-response audio,
because EMAGE needs the finished clip.

## Server setup

EMAGE runs as a resident worker in its own Python environment
(`tools/gaussian_head/gesture/emage_worker.py`), so torch never becomes an API
dependency. The worker is warmed in the background at startup and reused for
every response.

```
BIAS_EMAGE_PYTHON=<emage env>/python      # e.g. the env made by emage_infer.sbatch
BIAS_EMAGE_REPO=<character-policy checkout> # contains models/emage_audio.py
BIAS_EMAGE_DEVICE=cpu                     # or cuda; empty = cuda when available
BIAS_EMAGE_TIMEOUT=60                     # seconds per response
```

`GET /api/health` reports `providers.gesture.configured` and `ready`.

## Licence

EMAGE code and weights, BEAT2 and generated motion are research/education only.
Never commit or redistribute motion outputs, audio or weights. The motion is held
only in the short-lived clip store.
