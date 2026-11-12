/** Owns permissions and tracks even when a permission prompt resolves after Stop. */
export class CaptureLease {
  private generation = 0;
  private stream: MediaStream | null = null;
  private detach: (() => void)[] = [];

  constructor(private readonly onEnded: () => void = () => undefined) {}

  async start(request: () => Promise<MediaStream>): Promise<MediaStream | null> {
    this.stop();
    const ticket = this.generation;
    // Invoke request before any await: screen capture needs transient activation.
    let stream: MediaStream;
    try { stream = await request(); }
    catch (error) { if (ticket !== this.generation) return null; throw error; }
    if (ticket !== this.generation) { stream.getTracks().forEach((track) => track.stop()); return null; }
    if (!stream.getVideoTracks().some((track) => track.readyState === 'live')) {
      stream.getTracks().forEach((track) => track.stop());
      throw new Error('The selected input does not have a live video track.');
    }
    this.stream = stream;
    for (const track of stream.getVideoTracks()) {
      const ended = () => { if (ticket === this.generation) { this.stop(); this.onEnded(); } };
      track.addEventListener('ended', ended);
      this.detach.push(() => track.removeEventListener('ended', ended));
    }
    return stream;
  }

  stop(): void {
    this.generation++;
    this.detach.forEach((detach) => detach()); this.detach = [];
    // stop() does not fire ended; callers own their UI cleanup explicitly.
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
  }
}

export function captureError(error: unknown): string {
  const name = error instanceof Error ? error.name : '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Permission was not granted. Choose Camera or Share screen to try again.';
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'No camera was found. Connect a camera, share a screen, or try the pointer preview.';
  if (name === 'NotReadableError' || name === 'TrackStartError') return 'The video input is busy or unavailable. Close the other camera app and try again.';
  if (name === 'InvalidStateError') return 'Start screen sharing from its button while this page is active.';
  return error instanceof Error ? error.message : 'The video input could not start. Try again or use the pointer preview.';
}

interface FocusController { setFocusBehavior(behavior: 'no-focus-change'): void; }
type FocusControllerConstructor = new () => FocusController;

/** Best effort: sharing another tab should keep this studio focused where supported. */
export function requestScreenCapture(devices: Pick<MediaDevices, 'getDisplayMedia'>,
  Controller: FocusControllerConstructor | undefined = (globalThis as typeof globalThis & { CaptureController?: FocusControllerConstructor }).CaptureController,
): Promise<MediaStream> {
  let controller: FocusController | undefined;
  try {
    if (Controller) { controller = new Controller(); controller.setFocusBehavior('no-focus-change'); }
  } catch { controller = undefined; }
  // No await here: user activation is still active for the browser's picker.
  return devices.getDisplayMedia({ video: true, audio: false, ...(controller ? { controller } : {}) } as DisplayMediaStreamOptions);
}
