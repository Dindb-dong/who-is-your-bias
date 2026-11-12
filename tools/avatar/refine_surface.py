"""Blender: refine an editable coarse fit using event-balanced dense landmarks.

Preserves topology, skin weights and all existing facial expression deltas.
Inferred source depth receives 35% weight; surface displacements are bounded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.avatar.surface_fit import anatomical_frame, measured_surface, SmoothSurfaceFit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coarse_master', type=Path)
    parser.add_argument('template_binding', type=Path)
    parser.add_argument('template_landmarks', type=Path)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise ValueError('Use a new directory for each immutable surface candidate')
    args.output.mkdir(parents=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.coarse_master.resolve()))
    binding = json.loads(args.template_binding.read_text())
    detected = json.loads(args.template_landmarks.read_text())
    measurements = json.loads(args.measurements.read_text())
    target_normalized, uncertainty = measured_surface(measurements)
    skin = bpy.data.objects[binding['skin']]
    rig = bpy.data.objects[binding['rig']]
    base = skin.data.shape_keys.key_blocks[0].data if skin.data.shape_keys else skin.data.vertices
    candidates = [v for v in binding['vertices'] if v['point'][2] > binding['top'] - .38]
    anchors = []
    for point in detected['landmarks'][:468]:
        distances = [(v, math.hypot(v['image'][0] - point[0], v['image'][1] - point[1])) for v in candidates]
        closest = min(d for _, d in distances)
        vertex = min((v for v, d in distances if d <= closest + .002), key=lambda v: v['depth'])
        anchors.append(vertex['index'])
    points = np.array([skin.matrix_world @ base[index].co for index in anchors])
    origin, axes, height = anatomical_frame(points)
    normalized = (points - origin) @ axes / height
    target_normalized[:, 1] = normalized[:, 1] * .65 + target_normalized[:, 1] * .35
    targets = origin + target_normalized @ axes.T * height
    # Duplicate mesh anchors can correspond to adjacent image landmarks. Average
    # their desired displacements instead of making the linear system singular.
    unique = sorted(set(anchors))
    grouped = [[i for i, vertex in enumerate(anchors) if vertex == index] for index in unique]
    centers = np.array([points[indices[0]] for indices in grouped])
    goals = np.array([np.mean(targets[indices], axis=0) for indices in grouped])
    spread = np.array([np.mean(uncertainty[indices]) for indices in grouped])
    fitter = SmoothSurfaceFit(centers, goals, sigma=height * .12, maximum=height * .12, uncertainty=spread)
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    moved = {}
    maximum = 0.
    for obj in meshes:
        collections = [key.data for key in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else [obj.data.vertices]
        world = np.array([obj.matrix_world @ vertex.co for vertex in collections[0]])
        offsets = fitter.displacement(world)
        maximum = max(maximum, float(np.max(np.linalg.norm(offsets, axis=1))))
        moved[obj.name] = int(np.sum(np.linalg.norm(offsets, axis=1) > 1e-6))
        inverse_rotation = np.array(obj.matrix_world.inverted().to_3x3())
        local_offsets = offsets @ inverse_rotation.T
        # Equal identity offsets across all keys preserve each authored expression
        # delta exactly, while keeping the neutral face and mouth in one surface.
        for vertices in collections:
            for vertex, offset in zip(vertices, local_offsets):
                vertex.co += Vector(offset)
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True); bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    inverse = rig.matrix_world.inverted()
    for bone in rig.data.edit_bones:
        points_bone = np.array([rig.matrix_world @ bone.head, rig.matrix_world @ bone.tail])
        transformed = points_bone + fitter.displacement(points_bone)
        bone.head = inverse @ Vector(transformed[0]); bone.tail = inverse @ Vector(transformed[1])
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True)
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    glb = args.output / 'jensen-landmark-candidate-v2.glb'
    master = args.output / 'jensen-landmark-candidate-v2.blend'
    bpy.ops.export_scene.gltf(filepath=str(glb.resolve()), export_format='GLB', use_selection=True, export_animations=False, export_skins=True, export_morph=True, export_morph_normal=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(master.resolve()))
    bpy.context.scene.render.filepath = str((args.output / 'candidate-neutral.png').resolve())
    bpy.ops.render.render(write_still=True)
    fitted = points + fitter.displacement(points)
    report = {'schema_version': 'avatar-surface-fit.v1', 'persona_id': 'jensen-huang',
              'inputs': {name: {'path': str(getattr(args, name)), 'sha256': digest(getattr(args, name))} for name in ['coarse_master', 'template_binding', 'template_landmarks', 'measurements']},
              'method': 'event-balanced dense landmark Gaussian surface interpolation',
              'landmarks': 468, 'unique_mesh_anchors': len(unique), 'event_families': measurements['event_families'],
              'inferred_depth_weight': .35, 'sigma_face_height': .12, 'displacement_cap_face_height': .12,
              'maximum_displacement': maximum, 'moved_vertices_by_mesh': moved,
              'landmark_rmse_before': float(np.sqrt(np.mean(np.sum((points - targets) ** 2, axis=1)))),
              'landmark_rmse_after': float(np.sqrt(np.mean(np.sum((fitted - targets) ** 2, axis=1)))),
              'glb': {'filename': glb.name, 'sha256': digest(glb), 'bytes': glb.stat().st_size},
              'editable_master': {'filename': master.name, 'sha256': digest(master), 'bytes': master.stat().st_size},
              'likeness_review': 'ungraded', 'promotion': 'candidate_only',
              'limitations': ['Inferred source depth is not calibrated scan geometry.', 'Source expressions, glasses occlusion and detector priors affect the fitted surface.', 'Texture, hair and clothing remain generic or authored; no source pixels are baked.', 'Residual reduction is an engineering fit metric, not human likeness acceptance.']}
    (args.output / 'surface-fit-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'anchors': len(unique), 'rmse_before': report['landmark_rmse_before'], 'rmse_after': report['landmark_rmse_after']}))


if __name__ == '__main__':
    main()
