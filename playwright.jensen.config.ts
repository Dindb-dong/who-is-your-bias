import { defineConfig } from '@playwright/test';
import legacy from './playwright.config';

/** Keep historical Musk fixtures separate from the active Jensen studio. */
export default defineConfig(legacy, {
  testIgnore: [],
  testMatch: '**/jensen-attention.spec.ts',
  retries: 0,
  use: { ...legacy.use, baseURL: 'http://127.0.0.1:8001' },
  webServer: {
    command: 'uv run bias serve --host 127.0.0.1 --port 8001',
    env: { BIAS_CORPUS_DIR: 'data/personas/jensen-huang', BIAS_PERSONA_SNAPSHOT: 'data/snapshots/jensen-huang-2026-11-12.json', BIAS_MOTION_CATALOG: '' },
    url: 'http://127.0.0.1:8001/api/health',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
