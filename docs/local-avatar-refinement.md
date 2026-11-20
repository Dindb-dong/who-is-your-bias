# Local composite avatar refinement, 2026-11-18

Branch `feat/yaicon-lipsync-gesture`, based on local `a9c98c3`. Last fetched origin/main is `56b1868`; no remote refresh or publication performed in this task.

The exact local 240k Jensen head is retained. For its verified BIN hash only, the previous head-only opacity correction removes white shoulder/background splats, body materials are cloned with reduced jacket gloss and skin highlights, and studio rim lighting is reduced. Source material and head-cut shader callbacks are preserved and restored on detach.

Experimental learned motion now uses a designed sparse schedule, at most two gestures, separated by eight seconds. Existing gesture cues or Rhubarb silence endings select the start times. A single gesture is used when timing is absent. Motion is stretched by 1.35, arms scaled to .55, wrist/fingers to .38. Designed hand cues are suppressed while that controller is active. Torso support is designed and bounded to .015 radians. This schedule has no learned text-to-gesture semantic mapping. Speaker/quality review gates remain intact.

Verification uses the previous 20.72-second full-turn Qwen3-TTS waveform and exact transcript, replayed through mocked chat/speech endpoints in a separate browser. No new LLM/TTS inference or response-latency measurement was performed. Gaussian expression source remained tracked_speech. Browser errors were empty. Arm angle range changed from .52 to .2276 rad, hand .4351 to .0788, finger .7795 to .0279. Jaw range remained .1498 rad. Long resting intervals and return to idle were checked. This verifies bounded motion and playback, not human-rated naturalness or Jensen identity.

Local evidence is under project data/local/avatar_runs/refinement-20261118, outside Git. checks.json identifies the built bundle, waveform hash and frame samples. refined.webm contains a canvas recording with the same generated voice. idle.png and speaking.png show actual WebGL renders.

Remaining limitations include generic body shape/clothing geometry, missing real jacket detail, limited source gestures, no measured semantic alignment, some boundary artifacts, and local voice generation latency. Photoreal upper-body reconstruction still requires suitable images/video and an appropriate trainable representation. No API selection or paid inference was introduced.
