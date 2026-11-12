import type { FaceBox } from './types';

export interface DetectionFrame { boxes: FaceBox[]; observedAtMs: number; }
interface WorkerMessage { type: 'ready' | 'result' | 'error'; id?: number; boxes?: FaceBox[]; observedAtMs?: number; message?: string; }

/** A classic worker is intentional: MediaPipe's WASM loader uses importScripts. */
export class LocalFaceDetector {
  readonly ready: Promise<void>;
  private readonly worker: Worker;
  private closed = false;
  private initialized = false;
  private sequence = 0;
  private pending: { id: number; resolve: (result: DetectionFrame) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> } | null = null;
  private readyResolve!: () => void;
  private readyReject!: (error: Error) => void;
  private readyTimer: ReturnType<typeof setTimeout>;

  constructor() {
    if (typeof Worker === 'undefined' || typeof createImageBitmap === 'undefined') throw new Error('This browser cannot run local video tracking. Try a current desktop browser or use the pointer preview.');
    this.ready = new Promise((resolve, reject) => { this.readyResolve = resolve; this.readyReject = reject; });
    this.worker = new Worker('/vision/attention.worker.js', { name: 'local-face-detection' });
    this.readyTimer = setTimeout(() => this.fail(new Error('The local face detector timed out. Check that the vision assets are installed and try again.')), 25_000);
    this.worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      if (this.closed) return;
      const message = event.data;
      if (message.type === 'ready') { this.initialized = true; clearTimeout(this.readyTimer); this.readyResolve(); }
      else if (message.type === 'error') this.fail(new Error(message.message || 'The local face detector stopped. Try the pointer preview or restart capture.'));
      else if (message.type === 'result' && this.pending?.id === message.id) {
        const pending = this.pending!; this.pending = null; clearTimeout(pending.timer);
        pending.resolve({ boxes: message.boxes ?? [], observedAtMs: message.observedAtMs ?? NaN });
      }
    };
    this.worker.onerror = () => this.fail(new Error('The local face detector could not load. Run npm run prepare:vision, then restart the studio.'));
    this.worker.onmessageerror = () => this.fail(new Error('The video frame could not be read. Restart tracking or use the pointer preview.'));
    this.worker.postMessage({ type: 'init' });
  }

  detect(bitmap: ImageBitmap, observedAtMs: number): Promise<DetectionFrame> {
    if (this.closed || !this.initialized || this.pending) { bitmap.close(); return Promise.reject(new Error(this.closed ? 'The detector is closed.' : !this.initialized ? 'The detector is not ready.' : 'A frame is already in flight.')); }
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(new Error('Face detection stopped responding. Restart tracking or use the pointer preview.')), 8_000);
      this.pending = { id, resolve, reject, timer };
      try { this.worker.postMessage({ type: 'frame', id, bitmap, observedAtMs }, [bitmap]); }
      catch (error) { bitmap.close(); this.fail(error instanceof Error ? error : new Error('The video frame could not be transferred.')); }
    });
  }

  close(): void { this.fail(new DOMException('Tracking stopped.', 'AbortError')); }

  private fail(error: Error): void {
    if (this.closed) return;
    this.closed = true; clearTimeout(this.readyTimer);
    this.readyReject(error);
    if (this.pending) { clearTimeout(this.pending.timer); this.pending.reject(error); this.pending = null; }
    this.worker.terminate();
  }
}
