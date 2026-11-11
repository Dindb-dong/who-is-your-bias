import { afterEach, describe, expect, it, vi } from 'vitest';
import { createEventParser, getMotionClip, getMotionLibrary } from './api';
import type { Turn } from './api';
import { parseAvatarClip } from './avatar/validation';
import { encodedMotion, motionFixture } from './test/fixtures/motion';

afterEach(() => vi.unstubAllGlobals());

async function responseFixture() {
  const fixture = motionFixture();
  const encoded = await encodedMotion(fixture.clip);
  fixture.selection.clip_sha256 = encoded.hash;
  const fetcher = vi.fn<typeof fetch>(async () => new Response(encoded.bytes, { headers: { 'content-type': 'application/json' } }));
  vi.stubGlobal('fetch', fetcher);
  return { ...fixture, ...encoded, fetcher };
}

describe('measured motion download boundary', () => {
  it('checks exact response bytes and binds source, review and parent calibration before playback', async () => {
    const f = await responseFixture();
    const clip = await getMotionClip(f.selection, 'geometry_preview', new AbortController().signal);
    expect(clip.id).toBe(f.selection.primitive_id);
    expect(clip.calibration.neutral_frame_ms).toBe(-500);
    expect(f.fetcher.mock.calls[0]?.[0]).toBe('/api/motion/clip/synthetic-face?preview=true');
    expect(f.fetcher.mock.calls[0]?.[1]).toMatchObject({ redirect: 'error' });
  });

  it.each(['disabled', 'reviewed'] as const)('does not fetch unreviewed geometry in %s mode', async (mode) => {
    const f = await responseFixture();
    await expect(getMotionClip(f.selection, mode, new AbortController().signal)).rejects.toThrow('opt-in');
    expect(f.fetcher).not.toHaveBeenCalled();
  });

  it('never follows external or unexpected clip URLs', async () => {
    const f = await responseFixture(); f.selection.clip_url = 'https://elsewhere.example/motion.json';
    await expect(getMotionClip(f.selection, 'geometry_preview', new AbortController().signal)).rejects.toThrow('URL');
    expect(f.fetcher).not.toHaveBeenCalled();
  });

  it('rejects replaced bytes even when the replacement is valid motion JSON', async () => {
    const f = await responseFixture();
    f.fetcher.mockImplementation(async () => new Response(new TextDecoder().decode(f.bytes) + ' ', { headers: { 'content-type': 'application/json' } }));
    await expect(getMotionClip(f.selection, 'geometry_preview', new AbortController().signal)).rejects.toThrow('selection hash');
  });

  it('rejects mismatched provenance even if a payload hash is supplied for it', async () => {
    const f = motionFixture(); f.clip.source.source_start_seconds = 20;
    const encoded = await encodedMotion(f.clip); f.selection.clip_sha256 = encoded.hash;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(encoded.bytes, { headers: { 'content-type': 'application/json' } })));
    await expect(getMotionClip(f.selection, 'geometry_preview', new AbortController().signal)).rejects.toThrow('provenance');
  });

  it('never accepts an unreviewed track labeled as reviewed', async () => {
    const f = motionFixture(); f.clip.review.status = 'speaker_reviewed'; f.selection.review_status = 'speaker_reviewed';
    const encoded = await encodedMotion(f.clip); f.selection.clip_sha256 = encoded.hash;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(encoded.bytes, { headers: { 'content-type': 'application/json' } })));
    await expect(getMotionClip(f.selection, 'reviewed', new AbortController().signal)).rejects.toThrow('requires reviewed');
  });

  it('rejects channels outside the selected family', async () => {
    const f = await responseFixture(); f.selection.family = 'upper_body';
    await expect(getMotionClip(f.selection, 'geometry_preview', new AbortController().signal)).rejects.toThrow('geometry family');
  });

  it('rejects a source duration mismatch before making a network request', async () => {
    const f = await responseFixture(); f.selection.source_end_seconds = 12;
    await expect(getMotionClip(f.selection, 'geometry_preview', new AbortController().signal)).rejects.toThrow('interval');
    expect(f.fetcher).not.toHaveBeenCalled();
  });

  it('preserves optional library absence and rejects impossible counts', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(Response.json({ available: false, status: 'not_configured', reviewed_runtime_primitives: 0, preview_runtime_primitives: 0 }))
      .mockResolvedValueOnce(Response.json({ available: true, status: 'available', id: 'test', version: '1', reviewed_runtime_primitives: 4, preview_runtime_primitives: 2 }));
    vi.stubGlobal('fetch', fetcher);
    expect((await getMotionLibrary(new AbortController().signal)).available).toBe(false);
    await expect(getMotionLibrary(new AbortController().signal)).rejects.toThrow('Inconsistent');
  });
});

describe('retained source-neutral calibration', () => {
  it('allows external neutral only with the complete consistent parent reference', () => {
    const { clip } = motionFixture();
    expect(parseAvatarClip(clip, true).calibration.neutral_frame_ms).toBe(-500);
    const noParent = structuredClone(clip); delete noParent.calibration.parent_reference;
    expect(() => parseAvatarClip(noParent, true)).toThrow('neutral_frame_ms');
    const inWindow = structuredClone(noParent); inWindow.calibration.neutral_frame_ms = 100;
    expect(parseAvatarClip(inWindow, true).calibration.neutral_frame_ms).toBe(100);
  });
  it.each([
    { clip_sha256: 'invalid' }, { neutral_frame_ms: -1 }, { window_start_ms: 120001 },
    { rotation_operation: 'recalibrated' }, { neutral_frame_ms: 4000 },
  ])('rejects malformed or inconsistent retained references: %j', (changes) => {
    const { clip } = motionFixture();
    const value = { ...clip, calibration: { ...clip.calibration, parent_reference: { ...clip.calibration.parent_reference, ...changes } } };
    expect(() => parseAvatarClip(value, true)).toThrow();
  });
});

it('validates motion metadata in SSE and rejects motion for a profile-only result', () => {
  const { selection } = motionFixture();
  const turn: Turn = { id: 'turn', session_id: 'session', text: 'Test', mode: 'behavior', language: 'en', provider: 'evidence', citations: [],
    behavior: { duration_ms: 2000, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 0 }, motion: selection },
    grounding: { status: 'grounded', retrieved_ids: ['source'], invalid_citation_ids: [] }, metrics: { retrieval_ms: 0, generation_ms: 0, total_ms: 0 } };
  const emit = vi.fn(); const parser = createEventParser(emit);
  parser.push(`event: result\ndata: ${JSON.stringify({ turn })}\n\n`);
  expect(emit).toHaveBeenCalledOnce();
  const bad = createEventParser(() => undefined);
  expect(() => bad.push(`event: result\ndata: ${JSON.stringify({ turn: { ...turn, mode: 'profile' } })}\n\n`)).toThrow('Invalid result');
});
