import type { Language, StreamEvent } from '../api';

export interface CaptionDisplay { text: string; language: Language; label: string }
interface CaptionEntry {
  id: string; sequence: number; original: string;
  subtitle?: Extract<StreamEvent, { event: 'subtitle' }>['data'];
  startMs?: number; endMs?: number; sampleRate?: number;
}

/** One request owns the translations. PCM sample time, never token arrival,
 * selects the visible sentence. Late translations cannot revive past speech. */
export class CaptionTrack {
  private turnId?: string;
  private entries: CaptionEntry[] = [];
  constructor(readonly target: Language, readonly source: Language, readonly live: boolean) {}
  accept(event: StreamEvent): boolean {
    if (event.event === 'meta') {
      if (this.turnId || !event.data.turn_id) throw new Error('Caption turn ownership is missing or duplicated.');
      this.turnId = event.data.turn_id; return false;
    }
    if (event.event !== 'segment' && event.event !== 'subtitle' && event.event !== 'audio_segment' && event.event !== 'audio_segment_end') return false;
    const data = event.data;
    if (!this.turnId || data.turn_id !== this.turnId) throw new Error('Caption turn ownership changed.');
    if (event.event === 'segment') {
      if (data.sequence !== this.entries.length || this.entries.some((entry) => entry.id === data.segment_id)) throw new Error('Caption segments are reordered or duplicated.');
      this.entries.push({ id: data.segment_id, sequence: data.sequence,
        original: event.data.text.replace(/\[source:[A-Za-z0-9_.-]+\]/g, '').trim() });
    } else {
      const entry = this.entries[data.sequence];
      if (!entry || entry.id !== data.segment_id) throw new Error('Caption has no matching committed segment.');
      if (event.event === 'subtitle') {
        if (event.data.language !== this.target || entry.subtitle) throw new Error('Caption language changed or subtitle was duplicated.');
        entry.subtitle = event.data;
      } else if (event.event === 'audio_segment_end') {
        if (entry.sampleRate !== undefined && entry.endMs === Infinity) entry.endMs = event.data.sample_end / entry.sampleRate * 1000;
      } else {
        if (entry.startMs !== undefined) throw new Error('Caption audio timing was duplicated.');
        entry.startMs = event.data.sample_offset / event.data.sample_rate * 1000;
        entry.sampleRate = event.data.sample_rate;
        entry.endMs = event.data.sample_count === null ? Infinity :
          (event.data.sample_offset + event.data.sample_count) / event.data.sample_rate * 1000;
      }
    }
    return true;
  }
  active(audioElapsedMs: number, playing: boolean): CaptionDisplay | undefined {
    if (!this.live || !playing || !Number.isFinite(audioElapsedMs)) return undefined;
    const entry = this.entries.find((item) => item.startMs !== undefined && item.endMs !== undefined &&
      item.startMs <= audioElapsedMs && audioElapsedMs < item.endMs);
    if (!entry) return undefined;
    return entry.subtitle?.state === 'ready'
      ? { text: entry.subtitle.text, language: this.target, label: 'Korean captions · sentence timing' }
      : { text: entry.original, language: this.source,
          label: entry.subtitle?.state === 'unavailable' ? 'English original · Korean translation unavailable' : 'English original · Korean translation pending' };
  }
  transcript(): CaptionDisplay | undefined {
    if (this.live || !this.entries.length) return undefined;
    const ready = this.entries.every((entry) => entry.subtitle?.state === 'ready');
    return { text: this.entries.map((entry) => entry.subtitle?.state === 'ready' ? entry.subtitle.text :
      `[English original — translation ${entry.subtitle?.state === 'unavailable' ? 'unavailable' : 'pending'}] ${entry.original}`).join(' '),
      language: ready ? this.target : this.source, label: 'Translation transcript · not synchronized to audio' };
  }
}
