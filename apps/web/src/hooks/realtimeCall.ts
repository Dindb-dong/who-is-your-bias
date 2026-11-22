import { createPcmPlayback, type PcmChunk, type PcmState } from './pcmPlayback';

/** Browser side of /api/realtime: microphone PCM16 at 16 kHz up, Sophon reply audio and EMAGE frames down. */
export type CallStatus = 'idle' | 'connecting' | 'listening' | 'hearing' | 'replying' | 'ended' | 'failed';

export interface GestureMessage { type: 'gesture_state' | 'gesture' | 'gesture_end'; [key: string]: unknown }

export interface RealtimeCallHandlers {
  status(status: CallStatus, message?: string): void;
  /** Playback state of the current reply (feeds the avatar's audio clock). */
  audio(state: PcmState): void;
  replyStarted(responseId: string): void;
  transcript(responseId: string, delta: string): void;
  replyEnded(responseId: string, how: 'done' | 'cancelled'): void;
  gesture(message: GestureMessage): void;
}

export const MIC_RATE = 16000;
const FRAME = MIC_RATE / 10;   // 100 ms per upload

/** Linear resampler that keeps its phase across calls (mic blocks arrive at the device rate). */
export function createResampler(from: number, to: number) {
  // `position` is the read position within the current block; -1 means the previous block's last sample.
  const step = from / to; let position = 0; let last = 0;
  return (input: Float32Array): Float32Array => {
    const out: number[] = [];
    while (position + 1 < input.length) {
      const index = Math.floor(position), frac = position - index;
      const a = index < 0 ? last : input[index], b = input[index + 1];
      out.push(a + (b - a) * frac);
      position += step;
    }
    if (input.length) { position -= input.length; last = input[input.length - 1]; }
    return Float32Array.from(out);
  };
}

export function toPcm16(samples: Float32Array): ArrayBuffer {
  const view = new DataView(new ArrayBuffer(samples.length * 2));
  samples.forEach((value, index) => view.setInt16(index * 2, Math.max(-1, Math.min(1, value)) * 0x7fff, true));
  return view.buffer;
}

export function startRealtimeCall(handlers: RealtimeCallHandlers, options: { gesture: boolean; language: 'en' | 'ko' }) {
  let closed = false;
  let socket: WebSocket | undefined;
  let stream: MediaStream | undefined;
  let context: AudioContext | undefined;
  let player: ReturnType<typeof createPcmPlayback> | undefined;
  let chain: Promise<void> = Promise.resolve();
  let current: string | undefined;

  const stopPlayer = () => { player?.stop(); player = undefined; chain = Promise.resolve(); };
  const end = (status: CallStatus, message?: string) => {
    if (closed) return;
    closed = true;
    stopPlayer();
    try { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'stop' })); } catch { /* closing */ }
    socket?.close();
    stream?.getTracks().forEach((track) => track.stop());
    void context?.close().catch(() => undefined);
    handlers.status(status, message);
  };

  async function open() {
    handlers.status('connecting');
    try {
      // Echo cancellation keeps the avatar's own voice from interrupting it; headphones work best.
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    } catch {
      end('failed', 'Microphone access was denied or is unavailable.');
      return;
    }
    if (closed) { stream.getTracks().forEach((track) => track.stop()); return; }
    context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    // ScriptProcessor keeps this dependency-free; 4096 frames at 48 kHz is ~85 ms per block.
    const processor = context.createScriptProcessor(4096, 1, 1);
    const resample = createResampler(context.sampleRate, MIC_RATE);
    let pending = new Float32Array(0);
    processor.onaudioprocess = (event) => {
      if (closed || socket?.readyState !== WebSocket.OPEN) return;
      const block = resample(event.inputBuffer.getChannelData(0));
      const merged = new Float32Array(pending.length + block.length); merged.set(pending); merged.set(block, pending.length);
      let offset = 0;
      for (; offset + FRAME <= merged.length; offset += FRAME) socket.send(toPcm16(merged.subarray(offset, offset + FRAME)));
      pending = merged.slice(offset);
    };
    const mute = context.createGain(); mute.gain.value = 0;      // keep the processor running without hearing the mic
    source.connect(processor); processor.connect(mute); mute.connect(context.destination);

    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${scheme}://${location.host}/api/realtime`);
    socket.onopen = () => socket?.send(JSON.stringify({ type: 'start', gesture: options.gesture, language: options.language }));
    socket.onclose = () => end('ended');
    socket.onerror = () => end('failed', 'The realtime call connection failed.');
    socket.onmessage = (event) => {
      if (closed || typeof event.data !== 'string') return;
      let message: Record<string, unknown>;
      try { message = JSON.parse(event.data); } catch { return; }
      switch (message.type) {
        case 'ready': handlers.status('listening'); break;
        case 'user_speech': handlers.status(message.state === 'started' ? 'hearing' : 'listening'); break;
        case 'response': {
          const id = String(message.response_id);
          if (message.state === 'started') {
            stopPlayer(); current = id;
            player = createPcmPlayback((state) => { if (current === id) handlers.audio(state); });
            handlers.replyStarted(id); handlers.status('replying');
          } else if (id === current) {
            if (message.state === 'cancelled') { stopPlayer(); handlers.replyEnded(id, 'cancelled'); handlers.status('listening'); }
            else {
              const finishing = player, samples = Number(message.sample_count);
              chain = chain.then(() => finishing?.finish(samples)).then(() => {
                if (current === id) { handlers.replyEnded(id, 'done'); handlers.status('listening'); }
              }, () => undefined);
            }
          }
          break;
        }
        case 'audio': {
          const target = player;
          if (!target || message.turn_id !== current) break;
          // Pushes wait for playback room; chain them so gesture frames are never queued behind audio.
          chain = chain.then(() => target.push(message as unknown as PcmChunk));
          chain.catch(() => undefined);
          break;
        }
        case 'transcript': if (message.response_id === current) handlers.transcript(current!, String(message.delta ?? '')); break;
        case 'gesture_state': case 'gesture': case 'gesture_end':
          if (message.turn_id === current) handlers.gesture(message as GestureMessage);
          break;
        case 'error': end('failed', String(message.message ?? 'The realtime call failed.')); break;
      }
    };
  }
  void open();
  return { stop: () => end('ended') };
}
