import { expect, test } from '@playwright/test';

test('picture-in-picture moves the live canvas and leaves the question composer in Studio', async ({ page, isMobile }) => {
  test.skip(isMobile, 'Document Picture-in-Picture is a desktop presentation feature.');
  await page.goto('/');
  const open = page.getByRole('button', { name: 'Open avatar picture-in-picture' });
  test.skip(await open.count() === 0, 'This browser does not expose Document Picture-in-Picture.');
  const canvas = page.locator('.avatar-viewport canvas');
  await expect(canvas).toBeVisible();
  await canvas.evaluate((node) => node.setAttribute('data-test-existing-canvas', 'yes'));

  await open.click();
  await expect(page.getByRole('status').filter({ hasText: 'Avatar is in a separate window.' })).toBeVisible();
  await expect(canvas).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => {
    const pip = (window as Window & { documentPictureInPicture: { window: Window } }).documentPictureInPicture.window;
    return {
      sameCanvas: pip.document.querySelector('canvas')?.getAttribute('data-test-existing-canvas'),
      disclosure: pip.document.querySelector('.avatar-window-disclosure')?.textContent,
      composerCount: pip.document.querySelectorAll('textarea').length,
    };
  })).toEqual({ sameCanvas: 'yes', disclosure: 'AI simulation', composerCount: 0 });

  const keepsRendering = await page.evaluate(() => {
    const pip = (window as Window & { documentPictureInPicture: { window: Window } }).documentPictureInPicture.window;
    const movedCanvas = pip.document.querySelector('canvas');
    if (!movedCanvas) return false;
    return new Promise<boolean>((resolve) => {
      const timer = window.setTimeout(() => { frames.disconnect(); resolve(false); }, 3000);
      // The renderer writes this telemetry attribute on each rendered frame.
      const frames = new MutationObserver(() => { window.clearTimeout(timer); frames.disconnect(); resolve(true); });
      frames.observe(movedCanvas, { attributeFilter: ['data-attention-phase'] });
    });
  });
  expect(keepsRendering).toBe(true);

  await expect(page.getByText('Studio connected', { exact: true })).toBeVisible();
  const composer = page.getByRole('textbox', { name: 'Your message' });
  await composer.fill('What is your approach to building teams?');
  await composer.press('Enter');
  await expect(page.getByRole('log')).toContainText('Research persona');
  await expect(composer).toBeEnabled();

  await page.evaluate(() => {
    (window as Window & { documentPictureInPicture: { window: Window } }).documentPictureInPicture.window.close();
  });
  await expect(canvas).toHaveAttribute('data-test-existing-canvas', 'yes');
  await expect(open).toBeVisible();
});

test('separate avatar window works when document PiP is unavailable', async ({ page, isMobile }) => {
  test.skip(isMobile, 'Presentation pop-ups are exercised on desktop.');
  await page.addInitScript(() => Object.defineProperty(window, 'documentPictureInPicture', { value: undefined }));
  await page.goto('/');
  const canvas = page.locator('.avatar-viewport canvas');
  await expect(canvas).toBeVisible();
  await canvas.evaluate((node) => node.setAttribute('data-test-existing-canvas', 'yes'));

  const popupPromise = page.waitForEvent('popup');
  await page.getByRole('button', { name: 'Open avatar in a separate window' }).click();
  const popup = await popupPromise;
  await expect(popup).toHaveTitle('Presentation avatar');
  await expect(popup.locator('canvas')).toHaveAttribute('data-test-existing-canvas', 'yes');
  await expect(popup.getByText('AI simulation', { exact: true })).toBeVisible();
  await expect(popup.getByRole('textbox')).toHaveCount(0);

  await page.getByRole('button', { name: 'Return avatar to Studio' }).click();
  await expect(canvas).toHaveAttribute('data-test-existing-canvas', 'yes');
  await expect.poll(() => popup.isClosed()).toBe(true);
});
