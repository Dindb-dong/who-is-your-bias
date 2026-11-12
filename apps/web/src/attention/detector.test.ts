import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LocalFaceDetector } from './detector';

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessageerror: (() => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();
  constructor(readonly path: string, readonly options: unknown) { FakeWorker.instances.push(this); }
  emit(data: unknown) { this.onmessage?.({ data }); }
}
const bitmap = () => ({ close: vi.fn() }) as unknown as ImageBitmap;
const latest = () => FakeWorker.instances.at(-1)!;

beforeEach(() => { FakeWorker.instances = []; vi.stubGlobal('Worker', FakeWorker); vi.stubGlobal('createImageBitmap', vi.fn()); });
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('local detector worker ownership', () => {
  it('uses a classic same-origin worker and waits for its explicit readiness', async () => {
    const detector = new LocalFaceDetector(); const worker = latest(); const image = bitmap();
    expect(worker.path).toBe('/vision/attention.worker.js');
    expect(worker.options).not.toHaveProperty('type', 'module');
    await expect(detector.detect(image, 10)).rejects.toThrow('not ready'); expect(image.close).toHaveBeenCalledOnce();
    worker.emit({ type: 'ready' }); await detector.ready; detector.close(); expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it('transfers only one frame at a time and closes rejected inputs', async () => {
    const detector = new LocalFaceDetector(); const worker = latest(); worker.emit({ type: 'ready' }); await detector.ready;
    const one = bitmap(); const two = bitmap(); const pending = detector.detect(one, 100);
    expect(worker.postMessage).toHaveBeenLastCalledWith({ type: 'frame', id: 1, bitmap: one, observedAtMs: 100 }, [one]);
    await expect(detector.detect(two, 101)).rejects.toThrow('already in flight'); expect(two.close).toHaveBeenCalledOnce();
    worker.emit({ type: 'result', id: 1, observedAtMs: 100, boxes: [] });
    expect(await pending).toEqual({ boxes: [], observedAtMs: 100 }); detector.close();
  });

  it('ignores an out-of-order frame id and resolves the matching result', async () => {
    const detector = new LocalFaceDetector(); const worker = latest(); worker.emit({ type: 'ready' }); await detector.ready;
    const pending = detector.detect(bitmap(), 100); const resolved = vi.fn(); void pending.then(resolved);
    worker.emit({ type: 'result', id: 0, observedAtMs: 0, boxes: [] }); await Promise.resolve(); expect(resolved).not.toHaveBeenCalled();
    worker.emit({ type: 'result', id: 1, observedAtMs: 100, boxes: [] }); await pending; expect(resolved).toHaveBeenCalledOnce(); detector.close();
  });

  it('terminates initialization on Stop and ignores a late ready event', async () => {
    const detector = new LocalFaceDetector(); const worker = latest(); const ready = expect(detector.ready).rejects.toHaveProperty('name', 'AbortError');
    detector.close(); worker.emit({ type: 'ready' }); await ready;
    expect(worker.terminate).toHaveBeenCalledOnce(); await expect(detector.detect(bitmap(), 10)).rejects.toThrow('closed');
  });

  it('rejects pending inference on Stop and ignores late results', async () => {
    const detector = new LocalFaceDetector(); const worker = latest(); worker.emit({ type: 'ready' }); await detector.ready;
    const pending = expect(detector.detect(bitmap(), 100)).rejects.toHaveProperty('name', 'AbortError');
    detector.close(); worker.emit({ type: 'result', id: 1, observedAtMs: 100, boxes: [] }); await pending;
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it('turns initialization and worker failures into actionable errors', async () => {
    const detector = new LocalFaceDetector(); const worker = latest(); const ready = expect(detector.ready).rejects.toThrow('prepare:vision');
    worker.onerror?.(); await ready; expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it('bounds initialization time and releases a nonresponsive worker', async () => {
    vi.useFakeTimers(); const detector = new LocalFaceDetector(); const ready = expect(detector.ready).rejects.toThrow('timed out');
    vi.advanceTimersByTime(25_001); await ready; expect(latest().terminate).toHaveBeenCalledOnce();
  });

  it('bounds inference time without queuing later frames', async () => {
    vi.useFakeTimers(); const detector = new LocalFaceDetector(); const worker = latest(); worker.emit({ type: 'ready' }); await detector.ready;
    const pending = expect(detector.detect(bitmap(), 100)).rejects.toThrow('stopped responding');
    vi.advanceTimersByTime(8_001); await pending; expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it('closes untransferred bitmaps when posting a frame fails', async () => {
    const detector = new LocalFaceDetector(); const worker = latest(); worker.emit({ type: 'ready' }); await detector.ready;
    const image = bitmap(); worker.postMessage.mockImplementationOnce(() => { throw new Error('transfer failed'); });
    await expect(detector.detect(image, 100)).rejects.toThrow('transfer failed');
    expect(image.close).toHaveBeenCalledOnce(); expect(worker.terminate).toHaveBeenCalledOnce();
  });
});
