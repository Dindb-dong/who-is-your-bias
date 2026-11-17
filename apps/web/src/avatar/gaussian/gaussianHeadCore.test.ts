import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadBundle, parseBundle, validateGaussianBundle } from './gaussianHeadCore';

const header = { schema: 'gaussian-head.v1', buffers: { m_expr: { dtype: 'f32', offset: 0, shape: [1, 2], bytes: 8 } } };
function validFixture() {
  const shapes: Record<string, number[]> = { v_shaped: [3, 3], expr_dirs: [1, 9], pose_dirs: [36, 9], lbs_weights: [3, 5], j_base: [5, 3], j_expr_dirs: [1, 15], faces: [1, 3],
    g_xyz: [1, 3], g_rot: [1, 4], g_scale: [1, 3], g_opacity: [1], g_color: [1, 3], g_binding: [1], m_expr: [1, 1], m_pose: [1, 15], m_trans: [1, 3] };
  let offset = 0;
  const buffers = Object.fromEntries(Object.entries(shapes).map(([name, shape]) => {
    const bytes = shape.reduce((a, n) => a * n, 1) * 4;
    const value = { dtype: name === 'faces' || name === 'g_binding' ? 'u32' : 'f32', shape, offset, bytes }; offset += bytes;
    return [name, value];
  }));
  const manifest = { schema: 'gaussian-head.v1', n_verts: 3, n_faces: 1, n_gauss: 1, n_expr: 1, n_frames: 1, fps: 15, parents: [-1, 0, 1, 1, 1], buffers };
  const binary = new ArrayBuffer(offset), B = parseBundle(manifest, binary);
  B.faces.set([0, 1, 2]); B.lbs_weights.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]); B.g_rot[0] = 1;
  return { manifest, binary, B };
}
describe('Gaussian local asset loading', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('rejects truncated or misaligned binary arrays before creating views', () => {
    expect(() => parseBundle(header, new ArrayBuffer(4))).toThrow(/truncated/);
    expect(() => parseBundle({ ...header, buffers: { m_expr: { ...header.buffers.m_expr, offset: 1 } } }, new ArrayBuffer(12))).toThrow(/Invalid/);
    expect([...parseBundle(header, new Float32Array([1, 2]).buffer).m_expr]).toEqual([1, 2]);
  });
  it('reports a missing manifest without attempting binary decoding', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('missing', { status: 404 })); vi.stubGlobal('fetch', fetcher);
    await expect(loadBundle('/models/missing', undefined)).rejects.toThrow(/manifest.*404/);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('hashes the exact binary bytes and forwards cancellation', async () => {
    const { manifest, binary: bytes } = validFixture();
    const fetcher = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(manifest))).mockResolvedValueOnce(new Response(bytes));
    vi.stubGlobal('fetch', fetcher);
    const controller = new AbortController();
    const loaded = await loadBundle('/models/head', controller.signal);
    const expected = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((v) => v.toString(16).padStart(2, '0')).join('');
    expect(loaded.sha256).toBe(expected);
    expect(fetcher).toHaveBeenLastCalledWith('/models/head.bin', { signal: controller.signal });
  });
  it('rejects missing geometry, nonfinite values and dangling bindings before attachment', () => {
    const { manifest, B } = validFixture();
    expect(() => validateGaussianBundle(manifest, B)).not.toThrow();
    expect(() => validateGaussianBundle({ ...manifest, n_verts: 4 }, B)).toThrow(/mismatched/);
    B.g_binding[0] = 2;
    expect(() => validateGaussianBundle(manifest, B)).toThrow(/unavailable face/);
    B.g_binding[0] = 0; B.faces[0] = 4;
    expect(() => validateGaussianBundle(manifest, B)).toThrow(/unavailable vertex/);
    B.faces[0] = 0; B.v_shaped[0] = NaN;
    expect(() => validateGaussianBundle(manifest, B)).toThrow(/Nonfinite/);
  });
});
