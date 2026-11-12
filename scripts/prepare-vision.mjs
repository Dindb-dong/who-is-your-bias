import { createHash } from 'node:crypto';
import { cp, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// Provision once at build/dev time. Live video and inference stay same-origin.
const version = '0.10.32';
const modelUrl = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';
const modelSha256 = 'b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f';
const modelBytes = 229746;
const root = fileURLToPath(new URL('../', import.meta.url));
const destination = join(root, 'apps/web/public/vision');
const packageRoot = dirname(createRequire(import.meta.url).resolve('@mediapipe/tasks-vision'));
const metadata = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8'));
if (metadata.version !== version) throw new Error(`Expected MediaPipe ${version}; run npm ci with the committed lockfile.`);

const valid = (bytes) => bytes.length === modelBytes && createHash('sha256').update(bytes).digest('hex') === modelSha256;
await mkdir(destination, { recursive: true });
const modelPath = join(destination, 'blaze_face_short_range.tflite');
let cached;
try { cached = await readFile(modelPath); } catch (error) { if (error.code !== 'ENOENT') throw error; }
if (!cached || !valid(cached)) {
  const response = await fetch(modelUrl, { signal: AbortSignal.timeout(45_000) });
  if (!response.ok) throw new Error(`Face detector download failed (${response.status}). Retry npm run prepare:vision.`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (!valid(bytes)) throw new Error('Face detector model failed the pinned size/SHA-256 check. No unverified model was installed.');
  const temporary = `${modelPath}.${process.pid}.tmp`;
  await writeFile(temporary, bytes);
  await rename(temporary, modelPath);
}
await cp(join(packageRoot, 'wasm'), join(destination, 'wasm'), { recursive: true });
await cp(join(packageRoot, 'vision_bundle.mjs'), join(destination, 'vision_bundle.mjs'));
await writeFile(join(destination, 'assets.json'), `${JSON.stringify({
  package: '@mediapipe/tasks-vision', version, packageLicense: metadata.license,
  model: { url: modelUrl, sha256: modelSha256, bytes: modelBytes },
}, null, 2)}\n`);
console.log(`Local face detector ready: MediaPipe ${version}, SHA-256 ${modelSha256.slice(0, 12)}…`);
