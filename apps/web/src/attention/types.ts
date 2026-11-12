/** Display coordinates are normalized: left/top 0, right/bottom 1. */
export interface AttentionTarget {
  id: string;
  x: number;
  y: number;
  /** Capture time, using the same performance.now() clock as the renderer. */
  observedAtMs: number;
}

/** Unmirrored source-image coordinates; never an identity or demographic label. */
export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
}

export interface FaceTrack {
  id: number;
  box: FaceBox;
  observedAtMs: number;
  missing: boolean;
}

export interface TrackSnapshot {
  tracks: FaceTrack[];
  selected: FaceTrack | null;
  selectionLost: boolean;
}
