import { expect, test } from '@playwright/test';

test('answers a question with traceable evidence and resets the conversation', async ({ page }, info) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByText('Studio connected', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: /Meet the mind/ })).toBeVisible();
  await page.getByRole('textbox', { name: 'Your message' }).fill('Why did Tesla begin with an expensive sports car?');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText(/Evidence preview — these are curated/)).toBeVisible();
  await page.getByRole('button', { name: /evidence records/ }).click();
  await expect(page.getByRole('heading', { name: 'Response evidence' })).toBeVisible();
  const original = page.getByRole('link', { name: 'Read original source' }).first();
  await expect(original).toHaveAttribute('href', /^https:\/\//);
  await page.getByRole('button', { name: 'Close evidence', exact: true }).click();
  await page.screenshot({ path: info.outputPath('conversation.png'), fullPage: true });
  await page.getByRole('button', { name: 'New conversation', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Where should we begin?' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('sources, comparison, settings, and mobile layout work', async ({ page }, info) => {
  await page.goto('/');
  await expect(page.getByText('Studio connected', { exact: true })).toBeVisible();
  const isMobile = info.project.name === 'mobile';
  await page.getByRole('navigation', { name: isMobile ? 'Mobile navigation' : 'Main navigation', exact: true }).getByRole('button', { name: isMobile ? 'Sources' : /Source library/ }).click();
  await expect(page.getByRole('heading', { name: 'Every idea has a source.' })).toBeVisible();
  await page.getByRole('button', { name: 'Held out', exact: true }).click();
  await expect(page.getByText('Held out for evaluation').first()).toBeVisible();
  await page.getByRole('navigation', { name: isMobile ? 'Mobile navigation' : 'Main navigation', exact: true }).getByRole('button', { name: isMobile ? 'Lab' : 'Research lab', exact: true }).click();
  await page.getByRole('button', { name: 'Run comparison' }).click();
  await expect(page.getByText('Source preview', { exact: true })).toHaveCount(3);
  await expect(page.getByRole('button', { name: 'Run comparison' })).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await expect(page.getByRole('dialog', { name: 'Studio settings' })).toBeVisible();
  await page.getByRole('button', { name: 'Close settings' }).click();
  await page.screenshot({ path: info.outputPath('research-lab.png'), fullPage: true });
});

