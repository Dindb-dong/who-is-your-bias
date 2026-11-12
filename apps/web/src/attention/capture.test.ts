import { describe, expect, it, vi } from 'vitest';
import { CaptureLease, captureError, requestScreenCapture } from './capture';

function media() {
  const target = new EventTarget();
  const track = Object.assign(target, { readyState: 'live', stop: vi.fn(() => { track.readyState = 'ended'; }) });
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream;
  return { track, stream };
}
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (error: Error) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }

describe('capture ownership', () => {
  it('requests display capture synchronously before the first await', async () => {
    const lease = new CaptureLease(); const source = media(); const request = vi.fn(() => Promise.resolve(source.stream));
    const pending = lease.start(request); expect(request).toHaveBeenCalledOnce();
    expect(await pending).toBe(source.stream); lease.stop();
  });

  it('stops a late-granted stream after the user canceled the permission prompt', async () => {
    const lease = new CaptureLease(); const prompt = deferred<MediaStream>(); const source = media();
    const pending = lease.start(() => prompt.promise); lease.stop(); prompt.resolve(source.stream);
    expect(await pending).toBeNull(); expect(source.track.stop).toHaveBeenCalledOnce();
  });

  it('does not allow an earlier request to replace a newer input', async () => {
    const lease = new CaptureLease(); const first = deferred<MediaStream>(); const a = media(); const b = media();
    const stale = lease.start(() => first.promise);
    expect(await lease.start(() => Promise.resolve(b.stream))).toBe(b.stream);
    first.resolve(a.stream); expect(await stale).toBeNull();
    expect(a.track.stop).toHaveBeenCalledOnce(); expect(b.track.stop).not.toHaveBeenCalled();
    lease.stop(); expect(b.track.stop).toHaveBeenCalledOnce();
  });

  it('releases tracks and listeners when the browser ends sharing', async () => {
    const ended = vi.fn(); const lease = new CaptureLease(ended); const source = media();
    await lease.start(() => Promise.resolve(source.stream)); source.track.dispatchEvent(new Event('ended'));
    expect(source.track.stop).toHaveBeenCalledOnce(); expect(ended).toHaveBeenCalledOnce();
    source.track.dispatchEvent(new Event('ended')); expect(ended).toHaveBeenCalledOnce();
  });

  it('explicit Stop releases capture without depending on an ended event', async () => {
    const ended = vi.fn(); const lease = new CaptureLease(ended); const source = media();
    await lease.start(() => Promise.resolve(source.stream)); lease.stop(); lease.stop();
    expect(source.track.stop).toHaveBeenCalledOnce(); expect(ended).not.toHaveBeenCalled();
  });

  it('ignores errors from a canceled request and reports errors from the current request', async () => {
    const lease = new CaptureLease(); const first = deferred<MediaStream>();
    const pending = lease.start(() => first.promise); lease.stop(); first.reject(new DOMException('Denied', 'NotAllowedError'));
    expect(await pending).toBeNull();
    await expect(lease.start(() => Promise.reject(new DOMException('Denied', 'NotAllowedError')))).rejects.toHaveProperty('name', 'NotAllowedError');
  });

  it('rejects already-ended streams and still closes their tracks', async () => {
    const lease = new CaptureLease(); const source = media(); source.track.readyState = 'ended';
    await expect(lease.start(() => Promise.resolve(source.stream))).rejects.toThrow('live video track');
    expect(source.track.stop).toHaveBeenCalledOnce();
  });

  it('gives useful denied/missing/busy/activation explanations', () => {
    expect(captureError(new DOMException('', 'NotAllowedError'))).toContain('Permission');
    expect(captureError(new DOMException('', 'NotFoundError'))).toContain('No camera');
    expect(captureError(new DOMException('', 'NotReadableError'))).toContain('busy');
    expect(captureError(new DOMException('', 'InvalidStateError'))).toContain('button');
  });
});


describe('screen capture focus', () => {
  it('asks to keep focus before invoking the screen picker, without an await', async () => {
    const order: string[] = []; const source = media();
    class Controller { setFocusBehavior(value: string) { order.push(value); } }
    const getDisplayMedia = vi.fn((options) => { order.push('picker'); expect(options.audio).toBe(false); expect(options.controller).toBeInstanceOf(Controller); return Promise.resolve(source.stream); });
    const pending = requestScreenCapture({ getDisplayMedia }, Controller);
    expect(order).toEqual(['no-focus-change', 'picker']); await pending;
  });
  it('falls back to ordinary video-only capture when focus control is unsupported', async () => {
    const source = media(); const getDisplayMedia = vi.fn(() => Promise.resolve(source.stream));
    await requestScreenCapture({ getDisplayMedia }, undefined);
    expect(getDisplayMedia).toHaveBeenCalledWith({ video: true, audio: false });
  });
  it('treats focus control failure as optional and never repeats the media request', async () => {
    const source = media(); const getDisplayMedia = vi.fn(() => Promise.resolve(source.stream));
    class Controller { setFocusBehavior() { throw new Error('Unsupported focus control'); } }
    await requestScreenCapture({ getDisplayMedia }, Controller);
    expect(getDisplayMedia).toHaveBeenCalledOnce(); expect(getDisplayMedia).toHaveBeenCalledWith({ video: true, audio: false });
  });
});
