"""Blender: bake licensed multi-view face references into the fitted skin UVs.

This creates an animated-mesh texture, not a portrait billboard. Only the two
explicitly credited CC BY-SA Stanford photographs may contribute image pixels.
Reference lighting and occlusion are retained limitations, not recovered albedo.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.avatar.surface_fit import anatomical_frame

ALLOWED = {
    'stanford-2026-008': '3d4f737f1936504c646e46c398972f90dfc60cdff69b8bdd1241bb3b2c873c45',
    'stanford-2026-010': 'f08aa7f24227eaa6e3a964dcf676cdfe0ab85f917822ba3f668ef2641caf7838',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_pixels(image):
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    return values.reshape(image.size[1], image.size[0], 4)


def sample(image, uv):
    height, width, _ = image.shape
    xy = np.clip(uv, 0, 1) * np.array([width - 1, height - 1])
    left = np.floor(xy).astype(int); right = np.minimum(left + 1, [width - 1, height - 1])
    fraction = xy - left
    bottom = image[left[:, 1], left[:, 0], :3] * (1 - fraction[:, :1]) + image[left[:, 1], right[:, 0], :3] * fraction[:, :1]
    top = image[right[:, 1], left[:, 0], :3] * (1 - fraction[:, :1]) + image[right[:, 1], right[:, 0], :3] * fraction[:, :1]
    return bottom * (1 - fraction[:, 1:]) + top * fraction[:, 1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['master', 'template_binding', 'template_landmarks', 'measurements', 'reference_root', 'plan', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise ValueError('Choose an unused texture candidate output directory')
    plan = json.loads(args.plan.read_text())
    if plan.get('persona_id') != 'jensen-huang' or plan.get('license') != 'CC-BY-SA-4.0' or len(plan.get('sources', [])) != 2:
        raise ValueError('Expected the two-photo attributed Stanford texture plan')
    sources = []
    for source in plan['sources']:
        path = (args.reference_root / source['filename']).resolve()
        if not path.is_relative_to(args.reference_root.resolve()) or source['sha256'] != ALLOWED.get(source['id']) or digest(path) != source['sha256']:
            raise ValueError('Texture reference differs from the licensed frozen bytes')
        sources.append((source, path))
    args.output.mkdir(parents=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.master.resolve()))
    binding = json.loads(args.template_binding.read_text())
    landmarks = json.loads(args.template_landmarks.read_text())['landmarks']
    measured = {row['id']: row for row in json.loads(args.measurements.read_text())['accepted']}
    skin = bpy.data.objects[binding['skin']]; rig = bpy.data.objects[binding['rig']]
    base = skin.data.shape_keys.key_blocks[0].data
    world = np.array([skin.matrix_world @ v.co for v in base])
    visible = [v for v in binding['vertices'] if v['point'][2] > binding['top'] - .38]
    indices = []
    for point in landmarks[:468]:
        candidates = [(v, math.hypot(v['image'][0] - point[0], v['image'][1] - point[1])) for v in visible]
        closest = min(d for _, d in candidates)
        indices.append(min((v for v, d in candidates if d <= closest + .002), key=lambda v: v['depth'])['index'])
    anchor_points = world[indices]
    origin, axes, height = anatomical_frame(anchor_points)
    coordinates = (world - origin) @ axes / height
    unique = sorted(set(indices)); groups = [[i for i, vertex in enumerate(indices) if vertex == index] for index in unique]
    anchors = coordinates[unique]
    affine_inputs = np.column_stack([anchors, np.ones(len(anchors))])
    distance = np.sum((anchors[:, None] - anchors[None, :]) ** 2, axis=2)
    kernel = np.exp(-distance / (2 * .12 ** 2)) + np.eye(len(anchors)) * .005
    projections = []
    for source, path in sources:
        row = measured[source['id']]
        if row['sha256'] != source['sha256']:
            raise ValueError('Texture landmarks do not belong to the exact source image')
        photo_uv = np.asarray([[p[0], 1 - p[1]] for p in row['landmarks'][:468]])
        goals = np.array([np.mean(photo_uv[group], axis=0) for group in groups])
        affine = np.linalg.lstsq(affine_inputs, goals, rcond=None)[0]
        coefficients = np.linalg.solve(kernel, goals - affine_inputs @ affine)
        mapped = np.empty((len(world), 2))
        for start in range(0, len(world), 512):
            points = coordinates[start:start + 512]
            distances = np.sum((points[:, None] - anchors[None, :]) ** 2, axis=2)
            mapped[start:start + len(points)] = np.column_stack([points, np.ones(len(points))]) @ affine + np.exp(-distances / (2 * .12 ** 2)) @ coefficients
        image = bpy.data.images.load(str(path), check_existing=False)
        factor = min(1, 2048 / max(image.size)); image.scale(round(image.size[0] * factor), round(image.size[1] * factor))
        pixels = image_pixels(image); bpy.data.images.remove(image)
        width, image_height = row['size']
        source_points = np.asarray([[p[0] * width, -p[1] * image_height, -p[2] * width] for p in row['landmarks'][:468]])
        _, source_axes, _ = anatomical_frame(source_points)
        viewing = np.array([0, 0, 1]) @ source_axes
        projections.append((mapped, pixels, viewing))
    material = skin.data.materials[0]
    shader = material.node_tree.nodes.get('Principled BSDF')
    old_node = next(node for node in material.node_tree.nodes if node.type == 'TEX_IMAGE' and node.image and any(link.to_socket == shader.inputs['Base Color'] for link in node.outputs['Color'].links))
    old_image = old_node.image
    canvas = image_pixels(old_image).copy(); texture_height, texture_width, _ = canvas.shape
    # Match the generic neck/body to a robust cheek color from the licensed input.
    cheek_indexes = [50, 101, 118, 123, 280, 330, 347, 352]
    desired = []
    for source_index, (source, _) in enumerate(sources):
        row = measured[source['id']]
        desired.extend(sample(projections[source_index][1], np.array([[row['landmarks'][i][0], 1-row['landmarks'][i][1]] for i in cheek_indexes])))
    target_color = np.median(desired, axis=0)
    skin.data.calc_loop_triangles(); uv_data = skin.data.uv_layers.active.data
    # A neutral CC0 skin-color estimate; this color match does not infer physical
    # albedo or remove lighting from the source photographs.
    skin_pixels = canvas[:, :, :3].reshape(-1, 3)
    valid = skin_pixels[(skin_pixels[:, 0] > .3) & (skin_pixels[:, 0] > skin_pixels[:, 1] * 1.03)]
    multiplier = np.clip(target_color / np.maximum(np.median(valid, axis=0), .05), .55, 1.25)
    canvas[:, :, :3] = np.clip(canvas[:, :, :3] * multiplier, 0, 1)
    covered = np.zeros((texture_height, texture_width), dtype=bool)
    for triangle in skin.data.loop_triangles:
        vertex = list(triangle.vertices); local = coordinates[vertex]; center = np.mean(local, axis=0)
        if center[2] < -.60 or center[2] > .65 or abs(center[0]) > .65 or center[1] < -.12:
            continue
        normal = np.cross(local[1] - local[0], local[2] - local[0]); normal /= max(1e-12, np.linalg.norm(normal))
        # The anatomical frame may have either determinant; orient the outward
        # triangle normal using the mesh's actual world-space normal.
        mesh_normal = np.asarray(triangle.normal) @ np.asarray(skin.matrix_world.to_3x3()).T @ axes
        if np.dot(normal, mesh_normal) < 0: normal *= -1
        if normal[1] < .05: continue
        uv = np.array([uv_data[index].uv for index in triangle.loops]) * [texture_width - 1, texture_height - 1]
        lower = np.maximum(0, np.floor(np.min(uv, axis=0)).astype(int)); upper = np.minimum([texture_width - 1, texture_height - 1], np.ceil(np.max(uv, axis=0)).astype(int))
        if np.any(upper < lower): continue
        x, y = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1)); samples = np.column_stack([x.ravel(), y.ravel()])
        matrix = np.vstack([uv.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-8: continue
        barycentric = np.column_stack([samples, np.ones(len(samples))]) @ np.linalg.inv(matrix).T
        inside = np.min(barycentric, axis=1) >= -1e-5
        samples = samples[inside]; weights = barycentric[inside]
        if not len(samples): continue
        location = weights @ local
        mask = np.minimum.reduce([np.clip((.59-np.abs(location[:, 0]))/.12, 0, 1), np.clip((location[:, 2]+.56)/.08, 0, 1), np.clip((.60-location[:, 2])/.13, 0, 1), np.clip((location[:, 1]+.1)/.16, 0, 1)])
        colors = np.zeros((len(samples), 3)); total = 0.
        for mapped, pixels, viewing in projections:
            contribution = max(.03, float(np.dot(normal, viewing))) ** 4
            colors += sample(pixels, weights @ mapped[vertex]) * contribution; total += contribution
        colors /= total
        sx, sy = samples[:, 0], samples[:, 1]
        canvas[sy, sx, :3] = canvas[sy, sx, :3] * (1-mask[:, None]) + colors * mask[:, None]
        covered[sy, sx] |= mask > .1
    texture = bpy.data.images.new('JensenStanfordFaceCCBYSA40', width=texture_width, height=texture_height, alpha=True)
    texture.colorspace_settings.name = 'sRGB'; texture.pixels.foreach_set(canvas.ravel()); texture.update()
    texture.filepath_raw = str((args.output / 'jensen-stanford-face-cc-by-sa-4.png').resolve()); texture.file_format = 'PNG'; texture.save(); texture.pack()
    old_node.image = texture
    # The generic foundation has blue irises. An explicit authored brown tint
    # changes only chromatic iris texels; white sclera and alpha stay unchanged.
    iris_texels = 0
    for eye_material in bpy.data.materials:
        if eye_material.name != 'Eyes' or not eye_material.use_nodes: continue
        nodes = [node for node in eye_material.node_tree.nodes if node.type == 'TEX_IMAGE' and node.image]
        if not nodes: continue
        eye_image = nodes[0].image; pixels = image_pixels(eye_image).copy(); rgb = pixels[:, :, :3]
        iris = (rgb[:, :, 2] - rgb[:, :, 0] > .015) & (np.max(rgb, axis=2) < .92) & (pixels[:, :, 3] > .1)
        iris_texels += int(np.sum(iris))
        luminance = np.mean(rgb[iris], axis=1)
        rgb[iris] = luminance[:, None] * np.array([.50, .32, .18])
        tinted = bpy.data.images.new('JensenAuthoredBrownIris', width=eye_image.size[0], height=eye_image.size[1], alpha=True)
        tinted.colorspace_settings.name = 'sRGB'; tinted.pixels.foreach_set(pixels.ravel()); tinted.update()
        tinted.filepath_raw = str((args.output / 'jensen-authored-brown-iris.png').resolve()); tinted.file_format = 'PNG'; tinted.save(); tinted.pack()
        for node in nodes:
            if node.image == eye_image: node.image = tinted
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    # Retain gray hair cards but shorten the generic raised hairstyle. This is
    # authored grooming, not a measured reconstruction of individual hair strands.
    forehead = float(anchor_points[10, 2])
    for obj in meshes:
        if 'short01' not in obj.name: continue
        inverse = obj.matrix_world.inverted()
        collections = [key.data for key in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else [obj.data.vertices]
        offsets = []
        for vertex in collections[0]:
            point = obj.matrix_world @ vertex.co
            if point.z > forehead: point.z = forehead + (point.z - forehead) * .60
            offsets.append(inverse @ point - vertex.co)
        for vertices in collections:
            for vertex, offset in zip(vertices, offsets): vertex.co += offset
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True)
    for obj in meshes: obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    glb = args.output / 'jensen-source-fitted-candidate-v3.glb'; master = args.output / 'jensen-source-fitted-candidate-v3.blend'
    bpy.ops.export_scene.gltf(filepath=str(glb.resolve()), export_format='GLB', use_selection=True, export_animations=False, export_skins=True, export_morph=True, export_morph_normal=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(master.resolve()))
    bpy.context.scene.render.filepath = str((args.output / 'candidate-neutral.png').resolve()); bpy.ops.render.render(write_still=True)
    report = {'schema_version': 'avatar-reference-texture-bake.v1', 'persona_id': 'jensen-huang', 'license': 'CC-BY-SA-4.0',
              'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/', 'attribution': plan['attribution'], 'sources': plan['sources'],
              'changes': 'Two licensed face photographs warped by 468 source-bound landmarks, blended by surface viewing angle, and baked into the fitted animated skin UVs; gray hair shape authored.',
              'master_input_sha256': digest(args.master), 'measurement_sha256': digest(args.measurements), 'recipe_sha256': digest(args.plan),
              'texture_pixels_with_source_contribution': int(np.sum(covered)), 'texture_size': [texture_width, texture_height], 'skin_color_multiplier': multiplier.tolist(),
              'authored_iris_texels': iris_texels, 'authored_hair_compression': .60,
              'glb': {'filename': glb.name, 'sha256': digest(glb), 'bytes': glb.stat().st_size},
              'editable_master': {'filename': master.name, 'sha256': digest(master), 'bytes': master.stat().st_size},
              'promotion': 'candidate_only', 'likeness_review': 'pending_visual_review',
              'limitations': ['Photographic illumination, glasses and occlusions can remain baked into the surface.', 'The photograph bake is not intrinsic albedo or photogrammetric reconstruction.', 'The two texture views share one event family; other event families contribute geometry only.', 'Likeness and animation quality require explicit visual review.']}
    (args.output / 'texture-bake-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'covered_pixels': report['texture_pixels_with_source_contribution'], 'bytes': glb.stat().st_size}))


if __name__ == '__main__': main()
