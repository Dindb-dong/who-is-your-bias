import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadBundle, parseBundle } from './gaussianHeadCore';

const header = { schema: 'gaussian-head.v1', buffers: { m_expr: { dtype: 'f32', offset: 0, shape: [1, 2], bytes: 8 } } };
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
    const bytes = new Float32Array([1, 2]).buffer;
    const fetcher = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(header))).mockResolvedValueOnce(new Response(bytes));
    vi.stubGlobal('fetch', fetcher);
    const controller = new AbortController();
    const loaded = await loadBundle('/models/head', controller.signal);
    const expected = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((v) => v.toString(16).padStart(2, '0')).join('');
    expect(loaded.sha256).toBe(expected);
    expect(fetcher).toHaveBeenLastCalledWith('/models/head.bin', { signal: controller.signal });
  });
});
