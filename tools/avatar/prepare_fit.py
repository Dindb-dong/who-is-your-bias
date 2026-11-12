"""Blender: create a neutral camera/mesh binding for a new coarse identity fit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


def aim(obj, at):
    obj.rotation_euler = (Vector(at) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise ValueError('Use an unused directory for the immutable template preparation')
    args.output.mkdir(parents=True)
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    rig = next(obj for obj in bpy.data.objects if obj.type == 'ARMATURE')
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    skin = max(meshes, key=lambda obj: len(obj.data.vertices))
    skin.name = 'JensenCandidateSkin'
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 12
    scene.cycles.use_denoising = False
    scene.render.threads_mode = 'FIXED'; scene.render.threads = 3
    scene.render.resolution_x = 768; scene.render.resolution_y = 896; scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'; scene.world.color = (.12, .12, .12)
    scene.view_settings.view_transform = 'AgX'
    world = [skin.matrix_world @ v.co for v in skin.data.vertices]
    top = max(p.z for p in world)
    target_z = top - .115
    bpy.ops.object.camera_add(location=(0, 1.1, target_z))
    camera = bpy.context.object; camera.name = 'IdentityFitCamera'; camera.data.lens = 68
    aim(camera, (0, 0, target_z)); scene.camera = camera
    for position, energy, size in [((2, 3, 3.2), 600, 3), ((-2, 2, 2), 350, 3), ((0, -2, 2.8), 350, 2)]:
        bpy.ops.object.light_add(type='AREA', location=position); light = bpy.context.object
        light.data.energy = energy; light.data.size = size; aim(light, (0, 0, target_z))
    bpy.context.view_layer.update()
    projections = []
    for v, point in zip(skin.data.vertices, world):
        p = world_to_camera_view(scene, camera, point)
        projections.append({'index': v.index, 'point': list(point), 'image': [p.x, 1-p.y], 'depth': p.z})
    metadata = {'schema_version': 'avatar-template-binding.v1', 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'skin': skin.name, 'rig': rig.name, 'top': top, 'camera_target_z': target_z, 'vertices': projections,
                'basis': 'Blender Z-up, face +Y; camera front', 'foundation': 'committed generic CC0 anatomical proxy; not a celebrity model'}
    (args.output/'template-binding.json').write_text(json.dumps(metadata, allow_nan=False)+'\n')
    scene.render.filepath = str((args.output/'template.png').resolve())
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output/'template.blend').resolve()))
    bpy.ops.render.render(write_still=True)
    print(json.dumps({'template': str(args.output), 'vertices': len(projections), 'bones': len(rig.data.bones)}))


if __name__ == '__main__':
    main()
