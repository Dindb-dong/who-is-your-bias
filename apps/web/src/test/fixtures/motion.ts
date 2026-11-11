/** Synthetic protocol fixture. No person or source recording was measured. */
import type { AvatarClip, MotionSelection } from '../../avatar/types';

export function motionFixture(): { clip: AvatarClip; selection: MotionSelection } {
  const clip: AvatarClip = {
    schema_version: 'avatar-clip.v1', id: 'synthetic-face', duration_ms: 1000, max_gap_ms: 500,
    source: { source_id: 'synthetic', source_timeline: 'test_clock', source_start_seconds: 10,
      video_sha256: 'a'.repeat(64), observations_sha256: 'b'.repeat(64), review_sha256: 'c'.repeat(64) },
    review: { status: 'geometry_preview', reviewer: 'protocol fixture', track_id: 'synthetic-track',
      track_continuity_reviewed: false, source_clock_reviewed: false, speaker_association_reviewed: false,
      attribution_basis: 'Synthetic test geometry; no speaker attribution.' },
    calibration: { neutral_frame_ms: -500, face_neutral: { jawOpen: 0.1 }, bone_neutral: {},
      coordinate_basis: 'right_handed_y_up', bone_rotation_space: 'local_delta',
      parent_reference: { clip_sha256: 'd'.repeat(64), neutral_frame_ms: 4500, window_start_ms: 5000, rotation_operation: 'retained_parent_local_delta' } },
    frames: [{ time_ms: 0, face: { jawOpen: 0.2 } }, { time_ms: 500, face: { jawOpen: 0.4 } }, { time_ms: 900, face: { jawOpen: 0.15 } }],
    limitations: ['Protocol fixture only.'],
  };
  const selection: MotionSelection = {
    schema_version: 'motion-selection.v1', library_id: 'test-library', library_version: '1', primitive_id: 'synthetic-face',
    family: 'face', source_id: 'synthetic', source_event_id: 'synthetic-event', source_timeline: 'test_clock',
    source_start_seconds: 10, source_end_seconds: 11, duration_ms: 1000, review_status: 'geometry_preview',
    selection_basis: 'source_context_retrieval', context_summary: 'A synthetic test context.', context_alignment: 'asr_word_matches',
    score: 1, clip_url: '/api/motion/clip/synthetic-face', clip_sha256: '0'.repeat(64), parent_clip_sha256: 'd'.repeat(64),
  };
  return { clip, selection };
}

export async function encodedMotion(clip: AvatarClip) {
  const bytes = new TextEncoder().encode(JSON.stringify(clip));
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return { bytes, hash: Array.from(digest, (part) => part.toString(16).padStart(2, '0')).join('') };
}
