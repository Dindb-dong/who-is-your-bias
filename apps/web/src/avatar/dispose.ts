import { Material, Mesh, Object3D, Texture } from 'three';

/** Disposes shared GPU resources exactly once, including decoded GLB images. */
export function disposeObject(root: Object3D): void {
  const geometries = new Set<unknown>();
  const materials = new Set<Material>();
  const textures = new Set<Texture>();
  root.traverse((object) => {
    if (!(object instanceof Mesh)) return;
    if (!geometries.has(object.geometry)) {
      object.geometry.dispose();
      geometries.add(object.geometry);
    }
    const ownMaterials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of ownMaterials) materials.add(material);
  });
  for (const material of materials) {
    for (const value of Object.values(material)) if (value instanceof Texture) textures.add(value);
    material.dispose();
  }
  const images = new Set<unknown>();
  for (const texture of textures) {
    const image: unknown = texture.source?.data;
    if (image && !images.has(image) && typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) image.close();
    images.add(image);
    texture.dispose();
  }
}
