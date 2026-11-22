# Speech-driven gestures (EMAGE)

The studio can move the avatar's torso, arms and fingers from the synthesized
voice. A pretrained EMAGE model generates SMPL-X motion from the finished audio
of each response. The browser then plays it on the avatar rig, using the audio
clock. The motion follows the rhythm of the speech. It does not follow the
meaning of the answer, and it is not a measurement of the persona's own
gestures.

## Flow

1. With **Speech-driven gestures (EMAGE)** enabled, `POST /api/speech/prepare`
   receives `"gesture": true`. The toggle is in Settings → Voice playback, or
   add `?gesture=emage` to the URL (this also selects Connected voice).
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

Any failure keeps the verified audio playable with designed movement.

## Streaming voice (live PCM)

EMAGE already generates in windows of 64 frames (2.1 s). Each window starts
from the last 4 frames of the one before. With streaming voice
(`POST /api/chat/live` with `"gesture": true`), the server feeds each
synthesized sentence's PCM to a worker session in one-second pieces. It runs
each window as soon as its audio exists. The new frames go out on the same
event stream as the audio:

| Event | Meaning |
| --- | --- |
| `gesture_state` | `streaming` or a fallback status, sent after the first sentence. Includes `window_frames`. |
| `gesture` | `start_frame`, `frame_count`, base64 float32 poses. Frames are contiguous from 0. Frame f plays at f/30 s of the turn's concatenated PCM. |
| `gesture_end` | Total frames, `complete` or `stopped`. Sent before `audio_end`. |

Each window is decoded on its own, so the 4 overlapping frames are cross-faded
(slerp). The browser applies the calm filter causally: it shrinks toward the
mean of the frames received so far. If playback reaches the last received
frame, the browser holds it while fading out, then fades back in when more
frames arrive.

Measured on a laptop CPU:

- **Accuracy.** Inside each window, the streamed motion matches whole-clip
  generation within 0.05° on average. Near window boundaries it differs by
  2–3°.
- **Speed.** Each window takes about 0.25–0.45 s to compute.
- **Live start.** With the stub voice, EMAGE took over from designed movement
  about 1.6 s after the speech started. It then ran to the end without a stall.

`BIAS_EMAGE_WINDOW` (8–64) shortens the windows. With a shorter window, less
audio is needed before the first motion. This matters for voices that produce
audio only at real-time speed: the motion is ready about 1.3 s after its audio
with 32 frames, and about 0.85 s with 20. However, the model was trained on 64
frames, and the motion from shorter windows has not been reviewed visually.

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
BIAS_EMAGE_WINDOW=                        # streaming window (8-64 frames); empty = 64
```

`GET /api/health` reports `providers.gesture.configured` and `ready`.

## Licence

EMAGE code and weights, BEAT2 and generated motion are research/education only.
Never commit or redistribute motion outputs, audio or weights. The motion is held
only in the short-lived clip store.
