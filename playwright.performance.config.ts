import { defineConfig } from '@playwright/test';
import jensen from './playwright.jensen.config';

/** Real WebGL + real audio transport, synthetic Gaussian/driver test fixtures. */
export default defineConfig({
  ...jensen,
  testMatch: '**/jensen-gaussian.spec.ts',
  // Missing UI controls should report their own failure promptly rather than
  // consuming the full software-WebGL scenario budget.
  use: { ...jensen.use, baseURL: 'http://127.0.0.1:8002', actionTimeout: 15_000 },
  webServer: {
    command: 'uv run bias serve --host 127.0.0.1 --port 8002',
    env: { BIAS_CORPUS_DIR: 'data/personas/jensen-huang', BIAS_PERSONA_SNAPSHOT: 'data/snapshots/jensen-huang-2026-11-12.json', BIAS_MOTION_CATALOG: '' },
    url: 'http://127.0.0.1:8002/api/health', reuseExistingServer: !process.env.CI, timeout: 30_000,
  },
});
