import { describe, expect, it, vi } from 'vitest';
import { createMicrophoneController } from './microphone';
import type { MicrophoneState, Recognition, RecognitionEvent } from './microphone';

class FakeRecognition implements Recognition {
  static instances: FakeRecognition[] = [];
  lang = ''; interimResults = false; continuous = false;
  onresult: Recognition['onresult'] = null; onerror: Recognition['onerror'] = null; onend: Recognition['onend'] = null;
  start = vi.fn(); stop = vi.fn(); abort = vi.fn(() => this.onend?.());
  constructor() { FakeRecognition.instances.push(this); }
}
const result = (text: string): RecognitionEvent => ({ resultIndex: 0, results: { length: 1, 0: { isFinal: false, 0: { transcript: text } } } });

function setup() {
  FakeRecognition.instances = [];
  const states: MicrophoneState[] = []; const onText = vi.fn();
  const controller = createMicrophoneController((state) => states.push(state), onText, () => FakeRecognition);
  return { states, onText, controller };
}

describe('microphone capture ownership', () => {
  it('ignores old results and end events after stop and a new language capture', () => {
    const f = setup(); f.controller.toggle('en'); const first = FakeRecognition.instances[0];
    const lateResult = first.onresult; const lateEnd = first.onend;
    f.controller.stop(); f.controller.toggle('ko'); const second = FakeRecognition.instances[1];
    expect(first.lang).toBe('en-US'); expect(second.lang).toBe('ko-KR');
    lateResult?.(result('old transcript')); lateEnd?.();
    expect(f.onText).not.toHaveBeenCalled(); expect(f.states.at(-1)?.listening).toBe(true);
    second.onresult?.(result('새 질문')); expect(f.onText).toHaveBeenCalledWith('새 질문');
    expect(first.abort).toHaveBeenCalledTimes(1); f.controller.dispose();
  });

  it('detaches all callbacks before abort and never edits a draft after disposal', () => {
    const f = setup(); f.controller.toggle('en'); const instance = FakeRecognition.instances[0]; const lateResult = instance.onresult;
    const count = f.states.length; f.controller.dispose(); lateResult?.(result('late'));
    expect(f.states).toHaveLength(count); expect(f.onText).not.toHaveBeenCalled();
    expect(instance.onresult).toBeNull(); expect(instance.onend).toBeNull(); expect(instance.onerror).toBeNull();
  });

  it('releases capture on permission failure and permits a clean retry', () => {
    const f = setup(); f.controller.toggle('en'); const first = FakeRecognition.instances[0];
    first.onerror?.({ error: 'not-allowed' });
    expect(f.states.at(-1)?.error).toContain('permission'); expect(first.abort).toHaveBeenCalledTimes(1);
    f.controller.toggle('en'); expect(FakeRecognition.instances).toHaveLength(2);
    expect(f.states.at(-1)).toEqual({ listening: true, error: null }); f.controller.dispose();
  });
});
