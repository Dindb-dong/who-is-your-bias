import { transcribeQuestion } from '../api';
import type { MicrophoneController, MicrophoneState, OnQuestion } from './microphone';

const MAX_RECORDING_BYTES = 10_000_000;
interface Capture { abort: AbortController; recorder?: MediaRecorder; stream?: MediaStream; endedAt?: number }

export function recordingMimeType() {
  if (typeof MediaRecorder === 'undefined' || !navigator.mediaDevices?.getUserMedia) return undefined;
  return ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
    .find((type) => MediaRecorder.isTypeSupported(type));
}

/** Own the recording and transcription until the question is sent or cancelled. */
export function createRecordedQuestionController(mimeType: string,
  emit: (state: MicrophoneState) => void, onQuestion: OnQuestion): MicrophoneController {
  let active: Capture | undefined;
  let disposed = false;
  const clear = (error: string | null = null) => {
    const capture = active;
    active = undefined;
    capture?.abort.abort();
    const recorder = capture?.recorder;
    if (recorder) {
      recorder.ondataavailable = null; recorder.onstop = null; recorder.onerror = null;
      if (recorder.state !== 'inactive') recorder.stop();
    }
    capture?.stream?.getTracks().forEach((track) => track.stop());
    if (!disposed) emit({ listening: false, error });
  };
  return {
    stop: () => clear(),
    dispose() { disposed = true; clear(); },
    toggle(language) {
      if (disposed) return;
      if (active?.recorder?.state === 'recording') {
        active.endedAt = performance.now();
        emit({ listening: false, processing: true, error: null });
        active.recorder.stop();
        return;
      }
      if (active) return;
      const capture: Capture = { abort: new AbortController() };
      active = capture;
      emit({ listening: false, processing: true, error: null });
      void navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        if (active !== capture) { stream.getTracks().forEach((track) => track.stop()); return; }
        capture.stream = stream;
        const recorder = new MediaRecorder(stream, { mimeType });
        capture.recorder = recorder;
        const chunks: Blob[] = [];
        let bytes = 0;
        recorder.ondataavailable = ({ data }) => {
          if (active !== capture || !data.size) return;
          bytes += data.size;
          if (bytes > MAX_RECORDING_BYTES) {
            clear('The recording exceeded 10 MB. Please ask a shorter question or type.');
            return;
          }
          chunks.push(data);
        };
        recorder.onerror = () => clear('The microphone recording failed. Please try again or type.');
        recorder.onstop = () => {
          if (active !== capture) return;
          stream.getTracks().forEach((track) => track.stop());
          const endedAt = capture.endedAt ?? performance.now();
          if (!chunks.length) { clear('No audio was recorded. Please try again or type.'); return; }
          emit({ listening: false, processing: true, error: null });
          const recording = new Blob(chunks, { type: recorder.mimeType || mimeType });
          void transcribeQuestion(recording, language, capture.abort.signal).then((text) => {
            if (active !== capture) return;
            clear(text ? null : 'No speech was detected. Please try again or type.');
            if (text) onQuestion(text, endedAt);
          }).catch((error: unknown) => {
            if (active === capture) clear(error instanceof Error ? error.message : 'Could not transcribe the question.');
          });
        };
        recorder.start(250);
        emit({ listening: true, error: null });
      }).catch(() => {
        if (active === capture) clear('The microphone could not start. Check permission or type your question.');
      });
    },
  };
}
