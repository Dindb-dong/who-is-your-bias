import { describe, expect, it } from 'vitest';
import { createResampler, toPcm16 } from './realtimeCall';

describe('realtime call microphone path', () => {
  it('resamples 48 kHz blocks to 16 kHz continuously across block boundaries', () => {
    const resample = createResampler(48000, 16000);
    const ramp = Float32Array.from({ length: 4800 }, (_, i) => i / 4800);
    const out = [...resample(ramp.subarray(0, 1000)), ...resample(ramp.subarray(1000, 2333)), ...resample(ramp.subarray(2333))];
    expect(Math.abs(out.length - 1600)).toBeLessThanOrEqual(1);
    out.forEach((value, i) => expect(value).toBeCloseTo((i * 3) / 4800, 6));    // every third input sample, no seams
  });

  it('writes little-endian PCM16 with clipping', () => {
    const view = new DataView(toPcm16(Float32Array.from([0, 1, -1, 2, 0.5])));
    expect([0, 1, 2, 3, 4].map((i) => view.getInt16(i * 2, true))).toEqual([0, 32767, -32767, 32767, 16383]);
  });
});
