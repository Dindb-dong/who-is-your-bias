import type { createPcmPlayback, PcmChunk } from './pcmPlayback';

const minimumSeconds = 2.2;

export function bufferQwenVoice(player: ReturnType<typeof createPcmPlayback>) {
  const pending: PcmChunk[] = [];
  let expectedSeconds = 15;
  let firstPacketMs = 0;
  let audioSeconds = 0;
  let started = false;
  let holdUntilEnd = false;

  async function flush() {
    for (const packet of pending.splice(0)) await player.push(packet);
  }

  return {
    updateText(text: string) {
      const words = text.replace(/\[source:[^\]]*\]/gi, '').trim().split(/\s+/).length;
      expectedSeconds = Math.max(minimumSeconds, words * 0.44);
    },
    async push(packet: PcmChunk) {
      const now = performance.now();
      if (!audioSeconds) firstPacketMs = now;
      audioSeconds += packet.sample_count / packet.sample_rate;
      if (started) {
        // If synthesis falls behind playback, resume once the rest is ready.
        if (player.snapshot().firstDigitalPcmMs !== null && player.bufferedMs() <= 0) holdUntilEnd = true;
        if (!holdUntilEnd) { await player.push(packet); return; }
      }
      pending.push(packet);
      if (holdUntilEnd || audioSeconds < minimumSeconds) return;
      const productionRate = Math.max(0, audioSeconds - 0.24) / Math.max(0.01, (now - firstPacketMs) / 1000);
      const remaining = Math.max(0, expectedSeconds - audioSeconds);
      if (audioSeconds < minimumSeconds + remaining * Math.max(0, 1.06 - productionRate)) return;
      started = true;
      await flush();
    },
    flush,
  };
}
