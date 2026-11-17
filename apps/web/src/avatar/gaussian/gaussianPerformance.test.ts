import { describe, expect, it } from 'vitest';
import { neutralPose } from '../timeline';
import { createGaussianPerformance, gaussianFaceWeights, parseGaussianExpressionDriver, type GaussianExpressionDriver } from './gaussianPerformance';

const hash = 'a'.repeat(64);
const header = { n_expr: 2, n_frames: 4, fps: 10 };
const B = { m_expr: new Float32Array([0.1, 0.2, 0.3, 0.4, 90, 91, 92, 93]), m_pose: new Float32Array(60).fill(0.8), m_trans: new Float32Array(12).fill(10) };
const input = { rigPose: neutralPose(), elapsedMs: 50, idleMs: 50, speaking: false };
function driver(): GaussianExpressionDriver {
  return { schema: 'gaussian-expression-driver.v1', subject_id: 'jensen-huang', bundle_sha256: hash, n_expr: 2,
    input_names: ['browInnerUp', 'jawOpen'], input_bounds: [[0, 1], [0, 1]], weights: [[0.8, 0], [0, 0.4]], bias: [0, 0],
    output_bounds: [[0, 0.8], [0, 0.4]], neutral_expr: [0, 0], jaw_open_radians: 0.4,
    training: { method: 'ridge_regression', dataset_sha256: 'b'.repeat(64), examples: 2, frame_indices: [0, 1], validation_frame_indices: [2, 3] } };
}
describe('Gaussian live expression controller', () => {
  it('interpolates only explicitly allowed source frames and does not damp silent brows', () => {
    const c = createGaussianPerformance(header, B, { trainingFrameIndices: [0, 1] });
    const value = c.sample(input);
    expect([...value.expr]).toEqual([expect.closeTo(0.2, 6), expect.closeTo(0.3, 6)]);
    expect(value.sourceLipResidual).toBe(true);
    expect(value.expressionSource).toBe('observed_replay');
    expect([...c.sample({ ...input, idleMs: 150 }).expr]).toEqual([...value.expr]);
    expect(c.diagnostics.replayFrameCount).toBe(2);
  });
  it('does not assume an unknown bundle uses a 70% training split', () => {
    const c = createGaussianPerformance(header, B);
    expect(c.sample({ ...input, idleMs: 300 }).expressionSource).toBe('static_reference');
    expect([...c.sample({ ...input, idleMs: 300 }).expr]).toEqual([...B.m_expr.slice(0, 2)]);
  });
  it('uses the live speech pose for the jaw and never adds tracked root motion', () => {
    const c = createGaussianPerformance(header, B, { trainingFrameIndices: [0, 1] });
    const rigPose = { ...neutralPose(), jawOpen: 0.7, browRaise: 0.6 };
    expect(c.sample({ ...input, rigPose, speaking: true }).pose[6]).toBeCloseTo(0.245);
    const silent = c.sample({ ...input, rigPose });
    expect(silent.pose[6]).toBe(0);
    expect([...silent.pose.slice(0, 6)]).toEqual([0, 0, 0, 0, 0, 0]);
    expect([...silent.trans]).toEqual([0, 0, 0]);
    expect(gaussianFaceWeights(rigPose, false).browInnerUp).toBe(0.6);
  });
  it('maps calibrated brows while silent and leaves source parameters unmodified', () => {
    const c = createGaussianPerformance(header, B, { driver: driver(), bundleSha256: hash });
    const rigPose = { ...neutralPose(), browRaise: 0.8, jawOpen: 0.9 };
    const silent = c.sample({ ...input, rigPose });
    expect(silent.expr[0]).toBeCloseTo(0.64);
    expect(silent.expr[1]).toBe(0);
    expect(silent.pose[6]).toBe(0);
    expect(silent.sourceLipResidual).toBe(false);
    expect(silent.expressionSource).toBe('calibrated');
    expect(B.m_expr[2]).toBeCloseTo(0.3);
  });
  it('reproduces a media seek and reduces source replay without wall clock state', () => {
    const c = createGaussianPerformance(header, B, { trainingFrameIndices: [0, 1] });
    const a = c.sample({ ...input, speaking: true });
    c.sample({ ...input, speaking: true, elapsedMs: 100000 });
    expect([...c.sample({ ...input, speaking: true }).expr]).toEqual([...a.expr]);
    expect(c.sample({ ...input, reducedMotion: true }).expressionSource).toBe('static_reference');
  });
  it('rejects mismatched heads, malformed weights and validation leakage', () => {
    expect(() => parseGaussianExpressionDriver(driver(), header, 'c'.repeat(64))).toThrow(/exact head/);
    const d = driver(); d.weights[0] = [NaN, 0];
    expect(() => parseGaussianExpressionDriver(d, header, hash)).toThrow(/arrays/);
    const leaked = driver(); leaked.training.validation_frame_indices = [1, 2];
    expect(() => parseGaussianExpressionDriver(leaked, header, hash)).toThrow(/overlapping/);
    expect(() => createGaussianPerformance(header, B, { driver: driver(), bundleSha256: hash, trainingFrameIndices: [0, 2] })).toThrow(/validation/);
    expect(() => createGaussianPerformance(header, B, { trainingFrameIndices: [1, 1] })).toThrow(/ordered/);
  });
});
