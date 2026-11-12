import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  testIgnore: '**/jensen-attention.spec.ts',
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  // Software WebGL on shared CI runners can exceed a minute across the full
  // mobile scenario. Keep each assertion bounded; allow the scenario to finish.
  timeout: process.env.CI ? 120_000 : 60_000,
  use: {
    // Full Chromium supports the mobile full-page captures used by the suite.
    channel: 'chromium',
    baseURL: 'http://127.0.0.1:8000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
      args: ['--use-gl=angle', '--use-angle=swiftshader'],
    },
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
  webServer: {
    command: 'uv run bias serve --host 127.0.0.1 --port 8000',
    // Preserve the pre-Jensen Settings.from_env defaults, including the original
    // Q&A corpus and dated snapshot. The summary seed is a different fixture.
    env: { BIAS_CORPUS_DIR: 'data/qna/elon-musk-v1', BIAS_PERSONA_SNAPSHOT: 'data/snapshots/elon-musk-2026-11-11.json', BIAS_MOTION_CATALOG: 'data/motion/elon-musk-ted-v1.json' },
    url: 'http://127.0.0.1:8000/api/health',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});

