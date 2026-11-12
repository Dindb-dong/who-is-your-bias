import type { FaceBox, FaceTrack, TrackSnapshot } from './types';

export const TRACK_LOSS_MS = 700;
const MIN_SCORE = 0.6;
const MAX_TRACKS = 8;
const center = (box: FaceBox) => ({ x: box.x + box.width / 2, y: box.y + box.height / 2 });
const copyTrack = (track: FaceTrack): FaceTrack => ({ ...track, box: { ...track.box } });

export function validFaceBox(value: FaceBox): boolean {
  return [value.x, value.y, value.width, value.height, value.score].every(Number.isFinite)
    && value.x >= 0 && value.y >= 0 && value.width >= 0.015 && value.height >= 0.015
    && value.x + value.width <= 1.001 && value.y + value.height <= 1.001
    && value.score >= MIN_SCORE && value.score <= 1;
}

function overlap(a: FaceBox, b: FaceBox): number {
  const intersection = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x))
    * Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return intersection / Math.max(1e-6, a.width * a.height + b.width * b.height - intersection);
}

function matchCost(a: FaceBox, b: FaceBox): number {
  const ac = center(a); const bc = center(b);
  const distance = Math.hypot(ac.x - bc.x, ac.y - bc.y);
  const scale = Math.max(a.width, a.height, b.width, b.height);
  const ratio = (b.width * b.height) / (a.width * a.height);
  if (distance > Math.min(0.18, scale * 0.8) || ratio < 0.45 || ratio > 2.2) return Infinity;
  return distance / scale + 0.2 * (1 - overlap(a, b));
}

/**
 * Session-local geometric continuity only. Ambiguous assignments lose their old
 * IDs instead of implying recognition. A new face always needs a user selection.
 */
export class FaceTracker {
  private tracks: FaceTrack[] = [];
  private selectedId: number | null = null;
  private nextId = 1;
  private lastFrameMs = -Infinity;
  private lost = false;

  reset(): void { this.tracks = []; this.selectedId = null; this.nextId = 1; this.lastFrameMs = -Infinity; this.lost = false; }

  select(id: number | null, nowMs: number): TrackSnapshot {
    this.expire(nowMs);
    this.selectedId = id !== null && this.tracks.some((track) => track.id === id && !track.missing) ? id : null;
    this.lost = false;
    return this.snapshot(nowMs);
  }

  update(boxes: readonly FaceBox[], observedAtMs: number): TrackSnapshot {
    if (!Number.isFinite(observedAtMs) || observedAtMs <= this.lastFrameMs) return this.snapshot(this.lastFrameMs);
    this.lastFrameMs = observedAtMs;
    this.expire(observedAtMs);
    const detections = boxes.filter(validFaceBox).slice(0, MAX_TRACKS);
    const old = this.tracks;
    const costs = old.map((track) => detections.map((box) => matchCost(track.box, box)));
    const ambiguousOld = new Set<number>();
    const crowdedNew = new Set<number>();
    // Faces overlapping substantially cannot be safely distinguished by boxes.
    for (let a = 0; a < detections.length; a++) for (let b = a + 1; b < detections.length; b++) {
      if (overlap(detections[a], detections[b]) > 0.18) { crowdedNew.add(a); crowdedNew.add(b); }
    }
    for (let i = 0; i < old.length; i++) {
      const candidates = costs[i].map((cost, j) => ({ cost, j })).filter(({ cost }) => Number.isFinite(cost)).sort((a, b) => a.cost - b.cost);
      if (candidates.length > 1 && candidates[1].cost - candidates[0].cost < 0.22) ambiguousOld.add(i);
      if (candidates.length && crowdedNew.has(candidates[0].j)) ambiguousOld.add(i);
    }
    for (let j = 0; j < detections.length; j++) {
      const candidates = costs.map((row, i) => ({ cost: row[j], i })).filter(({ cost }) => Number.isFinite(cost)).sort((a, b) => a.cost - b.cost);
      if (candidates.length > 1 && candidates[1].cost - candidates[0].cost < 0.22) {
        ambiguousOld.add(candidates[0].i); ambiguousOld.add(candidates[1].i);
      }
    }
    const pairs = costs.flatMap((row, i) => row.map((cost, j) => ({ cost, i, j })))
      .filter(({ cost, i, j }) => Number.isFinite(cost) && !ambiguousOld.has(i) && !crowdedNew.has(j))
      .sort((a, b) => a.cost - b.cost);
    const usedOld = new Set<number>(); const usedNew = new Set<number>(); const next: FaceTrack[] = [];
    for (const { i, j } of pairs) {
      if (usedOld.has(i) || usedNew.has(j)) continue;
      usedOld.add(i); usedNew.add(j);
      next.push({ id: old[i].id, box: { ...detections[j] }, observedAtMs, missing: false });
    }
    for (let i = 0; i < old.length; i++) {
      if (usedOld.has(i)) continue;
      if (ambiguousOld.has(i)) {
        if (old[i].id === this.selectedId) { this.selectedId = null; this.lost = true; }
      } else next.push({ ...old[i], missing: true });
    }
    for (let j = 0; j < detections.length; j++) {
      if (!usedNew.has(j) && !crowdedNew.has(j)) next.push({ id: this.nextId++, box: { ...detections[j] }, observedAtMs, missing: false });
    }
    this.tracks = next;
    return this.snapshot(observedAtMs);
  }

  snapshot(nowMs: number): TrackSnapshot {
    this.expire(nowMs);
    return { tracks: this.tracks.map(copyTrack).sort((a, b) => a.id - b.id),
      selected: this.tracks.find((track) => track.id === this.selectedId) ? copyTrack(this.tracks.find((track) => track.id === this.selectedId)!) : null,
      selectionLost: this.lost };
  }

  private expire(nowMs: number): void {
    if (!Number.isFinite(nowMs)) return;
    this.tracks = this.tracks.filter((track) => nowMs - track.observedAtMs <= TRACK_LOSS_MS);
    if (this.selectedId !== null && !this.tracks.some((track) => track.id === this.selectedId)) {
      this.selectedId = null; this.lost = true;
    }
  }
}

export function displayBox(box: FaceBox, mirrored: boolean): FaceBox {
  return { ...box, x: mirrored ? 1 - box.x - box.width : box.x };
}

/** The overlay occupies only the video content, never its letterbox. */
export function containRect(sourceWidth: number, sourceHeight: number, width: number, height: number) {
  if (![sourceWidth, sourceHeight, width, height].every((value) => Number.isFinite(value) && value > 0)) return { x: 0, y: 0, width: 0, height: 0 };
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const w = sourceWidth * scale; const h = sourceHeight * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, width: w, height: h };
}
