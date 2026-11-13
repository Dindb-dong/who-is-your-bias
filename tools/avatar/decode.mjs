/** Decode the pinned generic CC0 foundation for editable Blender authoring. */
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
import sharp from 'sharp';
const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node decode.mjs INPUT.glb OUTPUT.glb');
const bytes = await readFile(input);
if (createHash('sha256').update(bytes).digest('hex') !== 'aee550c9cb5b20c6afccfb038c42c2eb8d7aab2c8fb74ca977aefed56f633519') throw new Error('Unexpected generic template revision');
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(input);
for (const texture of doc.getRoot().listTextures()) {
  texture.setImage(await sharp(texture.getImage()).png().toBuffer()).setMimeType('image/png');
}
for (const extension of doc.getRoot().listExtensionsUsed()) {
  if (['EXT_meshopt_compression', 'EXT_texture_webp'].includes(extension.extensionName)) extension.dispose();
}
await io.write(output, doc);
console.log(JSON.stringify({ output, input_sha256: createHash('sha256').update(bytes).digest('hex') }));
