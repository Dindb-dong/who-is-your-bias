import type { MotionMode } from '../avatar/types';
import type { MotionPlaybackState } from '../hooks/motionPlayback';

export function MotionNotice({ state, mode, noMatch, reducedMotion }: {
  state: MotionPlaybackState; mode: MotionMode; noMatch: boolean; reducedMotion: boolean;
}) {
  if (mode === 'disabled') return null;
  const selection = state.selection;
  const waiting = state.status === 'loading';
  if (!selection && !noMatch && !reducedMotion && !state.error) return null;
  const alignment = selection?.context_alignment === 'asr_word_matches' ? 'Caption context matched to ASR word times'
    : selection?.context_alignment === 'reviewed_word_anchors' ? 'Reviewed word anchors'
      : selection?.context_alignment === 'publisher_metadata_mapping' ? 'Publisher metadata mapping'
        : 'Nearby context; audio timing unverified';
  return <aside aria-label="Source movement status" style={{ marginTop: 10, padding: '12px 15px', border: '1px solid #dce4d1', borderRadius: 10, background: '#eef2e7', color: '#4f5f45', fontSize: 11, lineHeight: 1.6, overflowWrap: 'anywhere' }}>
    <div role="status" aria-live="polite"><strong>{reducedMotion ? 'Measured replay paused for reduced motion' : state.error ? 'Measured replay unavailable' : waiting ? 'Verifying source movement' : state.status === 'playing' ? 'Playing measured source movement' : state.status === 'completed' ? 'Source movement replay completed' : 'No eligible source movement matched this answer'}</strong></div>
    {state.error && <p style={{ margin: '4px 0 0' }}>{state.error}</p>}
    {selection && <>
      <p style={{ margin: '3px 0 0' }}>{selection.review_status === 'geometry_preview' ? 'Unreviewed geometry preview' : 'Speaker-reviewed source track'} · {selection.family === 'face' ? 'Face and head' : 'Upper body'} · {selection.source_id}</p>
      <details style={{ marginTop: 5 }}><summary style={{ cursor: 'pointer' }}>Source context and review</summary>
        <p style={{ marginTop: 5 }}>{selection.context_summary}</p>
        <p style={{ marginTop: 4 }}>{selection.source_start_seconds.toFixed(2)}–{selection.source_end_seconds.toFixed(2)} s · {selection.source_timeline}</p>
        <p style={{ marginTop: 4 }}>{alignment}. {selection.review_status === 'geometry_preview' ? 'Speaker association remains unreviewed.' : 'Source association has an explicit review.'}</p>
        <p style={{ marginTop: 4 }}>One replay at recorded speed, independent of speech timing. Source context does not establish a personal gesture policy or matching emotion.</p>
      </details>
    </>}
  </aside>;
}
