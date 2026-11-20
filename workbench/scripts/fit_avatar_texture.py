"""Bake a PR #20 face reference into the existing skinned avatar's UV atlas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import Delaunay

PROJECT = Path(__file__).resolve().parents[1]
WEB = PROJECT / "integrations/who-is-your-bias-pr19/apps/web"
if (PROJECT.parent / "apps/web/package.json").is_file():
    WEB = PROJECT.parent / "apps/web"
REFERENCES = (PROJECT / "data/local/integration/jensen-pr20/data/acquisition"
              / "jensen-huang/reference-images")
OUTPUT = PROJECT / "data/local/avatars/jensen-photo-fit-v4"
BASE = WEB / "public/models/jensen-source-fitted-v3.glb"
PHOTO_ID = "JH0015_face"
PR_HEAD = "fa336cfeb2149c2015393508d16c4742fb1eb1fb"
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_REGIONS = [
    [70, 63, 105, 66, 107, 133, 155, 145, 144, 163, 7, 33],
    [336, 296, 334, 293, 300, 263, 249, 390, 373, 374, 382, 362],
    [107, 336, 168, 6],
]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
        314, 17, 84, 181, 91, 146]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_glb(payload):
    magic, version, length, json_length, json_tag = struct.unpack_from("<4sIII4s", payload)
    assert (magic, version, length, json_tag) == (b"glTF", 2, len(payload), b"JSON")
    offset = 20 + json_length
    bin_length, bin_tag = struct.unpack_from("<I4s", payload, offset)
    assert bin_tag == b"BIN\x00" and offset + 8 + bin_length == length
    return json.loads(payload[20:offset]), payload[offset + 8:]


def pack_glb(model, binary):
    encoded = json.dumps(model, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    return (struct.pack("<4sIII4s", b"glTF", 2, 28 + len(encoded) + len(binary), len(encoded), b"JSON")
            + encoded + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


def patch_model(model, binary, skin_view, texture):
    original_length = len(binary)
    binary += texture
    model["bufferViews"][skin_view].update(byteOffset=original_length, byteLength=len(texture))
    model["buffers"][0]["byteLength"] = len(binary)
    for material in model["materials"]:
        pbr = material["pbrMetallicRoughness"]
        if material["name"] == "JensenAuthoredDarkJacket":
            pbr["roughnessFactor"] = 0.56
        if material["name"] == "JensenAuthoredTitaniumGlasses":
            pbr["roughnessFactor"] = 0.38
    return pack_glb(model, binary)


def landmarks(detector, rgb):
    height, width = rgb.shape[:2]
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if len(result.face_landmarks) != 1:
        raise ValueError("Expected one visible face in each input")
    return np.float32([(p.x * width - 0.5, p.y * height - 0.5) for p in result.face_landmarks[0]])


def warp_reference(rgb, source, target, size):
    # Dense face landmarks define a piecewise affine map to the model's frontal view.
    tessellation = Delaunay(target[:468])
    y, x = np.mgrid[:size, :size].astype(np.float32)
    pixels = np.stack((x, y), axis=-1)
    cell = tessellation.find_simplex(pixels)
    valid = cell >= 0
    active = cell[valid]
    affine = tessellation.transform[active]
    relative = pixels[valid] - affine[:, 2]
    weights = np.einsum("nij,nj->ni", affine[:, :2], relative)
    weights = np.column_stack((weights, 1 - weights.sum(axis=1)))
    mapping = np.zeros_like(pixels)
    mapping[valid] = np.einsum("ni,nij->nj", weights, source[tessellation.simplices[active]])
    warped = cv2.remap(rgb, mapping, None, cv2.INTER_LINEAR)
    oval = np.zeros((size, size), np.uint8)
    cv2.fillPoly(oval, [np.rint(target[FACE_OVAL]).astype(np.int32)], 1)
    scale = np.linalg.norm(target[10] - target[152]) / 400
    feather = cv2.distanceTransform(oval, cv2.DIST_L2, 5)
    alpha = np.clip(feather / (24 * scale), 0, 1) * valid
    # Keep the existing brow, eyelid and lip seams aligned with the animated mesh.
    excluded = np.ones_like(oval)
    for indices in [*EYE_REGIONS, LIPS]:
        cv2.fillConvexPoly(excluded, cv2.convexHull(np.rint(target[indices]).astype(np.int32)), 0)
    clearance = cv2.distanceTransform(excluded, cv2.DIST_L2, 5)
    alpha *= np.clip((clearance - 6 * scale) / (18 * scale), 0, 1)
    return warped, alpha.astype(np.float32)


def triangle_pixels(triangle, width, height):
    lo = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), (width - 1, height - 1))
    if np.any(hi < lo):
        return None
    edge1, edge2 = triangle[1:] - triangle[0]
    det = edge1[0] * edge2[1] - edge1[1] * edge2[0]
    if abs(det) < 1e-7:
        return None
    y, x = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
    dx, dy = x - triangle[0, 0], y - triangle[0, 1]
    w1 = (edge2[1] * dx - edge2[0] * dy) / det
    w2 = (-edge1[1] * dx + edge1[0] * dy) / det
    w0 = 1 - w1 - w2
    inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    return y[inside], x[inside], np.stack((w0[inside], w1[inside], w2[inside]), axis=-1)


def map_to_atlas(geometry, mask, width, height):
    size = geometry["size"]
    verts = np.asarray(geometry["screen"])
    triangles = verts[np.asarray(geometry["index"]).reshape(-1, 3)]
    zbuffer = np.full((size, size), np.inf, np.float32)
    active = []
    face_y, face_x = np.nonzero(mask > 0)
    low, high = np.array([face_x.min(), face_y.min()]), np.array([face_x.max(), face_y.max()])
    for tri in triangles:
        screen = tri[:, :2]
        if np.any(screen.max(axis=0) < low) or np.any(screen.min(axis=0) > high):
            continue
        pixels = triangle_pixels(screen, size, size)
        if pixels is None:
            continue
        y, x, weight = pixels
        zbuffer[y, x] = np.minimum(zbuffer[y, x], weight @ tri[:, 2])
        active.append(tri)
    uv_to_screen = np.zeros((height, width, 2), np.float32)
    atlas_depth = np.full((height, width), np.inf, np.float32)
    for tri in active:
        uv = tri[:, 3:] * (width, height) - 0.5
        pixels = triangle_pixels(uv, width, height)
        if pixels is None:
            continue
        y, x, weight = pixels
        screen = weight @ tri[:, :3]
        ix, iy = np.rint(screen[:, :2]).astype(int).T
        inside = (ix >= 0) & (iy >= 0) & (ix < size) & (iy < size)
        y, x, screen, ix, iy = y[inside], x[inside], screen[inside], ix[inside], iy[inside]
        z = screen[:, 2]
        visible = ((mask[iy, ix] > 0) & (z <= zbuffer[iy, ix] + 0.00032)
                   & (z < atlas_depth[y, x]))
        y, x, screen = y[visible], x[visible], screen[visible]
        uv_to_screen[y, x] = screen[:, :2]
        atlas_depth[y, x] = screen[:, 2]
    return uv_to_screen, np.isfinite(atlas_depth)


def blend_texture(base, photo, alpha):
    selected = alpha > 0.72
    if np.count_nonzero(selected) < 5000:
        raise ValueError("Face projection did not cover enough atlas texels")
    photo = photo.astype(np.float32)
    base = base.astype(np.float32)
    photo_lab = cv2.cvtColor(photo / 255, cv2.COLOR_RGB2LAB)
    base_lab = cv2.cvtColor(base / 255, cv2.COLOR_RGB2LAB)
    photo_lab += np.median(base_lab[selected], axis=0) - np.median(photo_lab[selected], axis=0)
    # Fade photographic light and colour into the baked base; preserve more of the skin detail.
    soft = cv2.GaussianBlur(alpha, (0, 0), 9)
    base_light = cv2.GaussianBlur(base_lab[:, :, 0] * alpha, (0, 0), 9) / np.maximum(soft, 1e-5)
    photo_light = cv2.GaussianBlur(photo_lab[:, :, 0] * alpha, (0, 0), 9) / np.maximum(soft, 1e-5)
    delta = photo_lab - base_lab
    delta[:, :, 0] -= 0.33 * (photo_light - base_light)
    weight = alpha[:, :, None] * np.float32([0.76, 0.55, 0.55])
    blended = cv2.cvtColor(base_lab + weight * delta, cv2.COLOR_LAB2RGB)
    return np.where(alpha[:, :, None] > 0, np.rint(blended * 255), base).astype(np.uint8)


def build(prefix, task):
    raw = BASE.read_bytes()
    model, binary = parse_glb(raw)
    skin = next(m for m in model["materials"] if m["name"] == "Skin")
    texture = model["textures"][skin["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    skin_view = model["images"][texture["extensions"]["EXT_texture_webp"]["source"]]["bufferView"]
    view = model["bufferViews"][skin_view]
    start = view.get("byteOffset", 0)
    base = np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB"))
    with (REFERENCES / "metadata.csv").open(encoding="utf-8-sig") as source_csv:
        record = next(r for r in csv.DictReader(source_csv) if r["image_id"] == PHOTO_ID)
    if record["license"] != "CC BY-SA 4.0":
        raise ValueError("Verify texture source attribution and adaptation terms")
    photo_path = REFERENCES / record["file_path"]
    photo = np.asarray(ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB"))
    neutral = np.asarray(Image.open(f"{prefix}-neutral.png").convert("RGB"))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task)), num_faces=1)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        source, target = landmarks(detector, photo), landmarks(detector, neutral)
    geometry = json.loads(Path(f"{prefix}-geometry.json").read_text())
    warped, alpha = warp_reference(photo, source, target, geometry["size"])
    height, width = base.shape[:2]
    mapping, visibility = map_to_atlas(geometry, alpha, width, height)
    photo_uv = cv2.remap(warped, mapping, None, cv2.INTER_LINEAR)
    alpha_uv = cv2.remap(alpha, mapping, None, cv2.INTER_LINEAR) * visibility
    # A narrow fade hides UV chart edges without blurring neighbouring atlas islands.
    alpha_uv *= np.clip(cv2.distanceTransform(visibility.astype(np.uint8), cv2.DIST_L2, 5) / 4, 0, 1)
    texture_out = blend_texture(base, photo_uv, alpha_uv)
    compressed = io.BytesIO()
    Image.fromarray(texture_out).save(compressed, format="WEBP", quality=96, method=6)
    glb = patch_model(model, binary, skin_view, compressed.getvalue())
    asset = OUTPUT / "jensen-photo-fit-v4.glb"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(glb)
    attribution = (BASE.with_suffix(".LICENSE.md")).read_text()
    added = (f"# Jensen photo-fit v4\n\n"
             f"Adapts the v3 face texture with a licensed frontal reference from [PR #20]"
             f"(https://github.com/Dindb-dong/who-is-your-bias/pull/20) · "
             f"[{record['title']}]({record['source_url']}) · "
             f"**{record['author']}**, [CC BY-SA 4.0]({record['license_url']}).\n\n"
             "Changes: landmark-guided photograph projection and color blending in the facial UV atlas, "
             "and reduced specular sharpness on the jacket and glasses. The existing mesh, articulation "
             "and base texture attribution are retained. The derived asset is CC BY-SA 4.0.\n\n"
             "## Upstream attribution\n\n")
    license_path = OUTPUT / "jensen-photo-fit-v4.LICENSE.md"
    license_path.write_text(added + attribution)
    manifest = {
        "version": "jensen-photo-fit-v4", "sha256": sha(glb), "base_sha256": sha(raw),
        "license_sha256": sha(license_path.read_bytes()),
        "reference_pr_head": PR_HEAD, "reference_image": {
            **{key: record[key] for key in ("image_id", "title", "author", "source_url", "license", "license_url")},
            "sha256": sha(photo_path.read_bytes()),
        },
        "face_texels_updated": int(np.count_nonzero(np.any(base != texture_out, axis=-1))),
        "atlas_size": [width, height], "geometry": "source-fitted-v3",
    }
    (OUTPUT / "appearance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capture_prefix", type=Path)
    cli.add_argument("--task", type=Path, default=PROJECT / ".tools/avatar-python/face_landmarker.task")
    args = cli.parse_args()
    build(args.capture_prefix, args.task)
