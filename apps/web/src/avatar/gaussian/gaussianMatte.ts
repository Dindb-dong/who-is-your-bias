const smooth = (low: number, high: number, value: number) => {
  const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
  return t * t * (3 - 2 * t);
};

/** Opacity adjustment tuned for the shared, white-background 240k head bundle in its rest pose. */
export function cleanOpacity(x: number, y: number, z: number, r: number, g: number, b: number) {
  const white = smooth(0.52, 0.92, Math.min(r, g, b)) * (1 - smooth(0.05, 0.18, Math.max(r, g, b) - Math.min(r, g, b)));
  const shoulder = 1 - smooth(-0.103, -0.072, y);
  const edge = Math.max(smooth(0.069, 0.115, Math.abs(x)), smooth(0.086, 0.131, y), 1 - smooth(-0.10, -0.041, z));
  const colorCut = white * Math.max(shoulder, edge) * 0.999;
  const outsideNeck = shoulder * smooth(0.067, 0.14, Math.abs(x));
  const belowCollar = 1 - smooth(-0.172, -0.119, y);
  return (1 - colorCut) ** 3 * (1 - outsideNeck * 0.94) * (1 - belowCollar * 0.98);
}
