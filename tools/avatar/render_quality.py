"""Blender CPU visual qualification of a fitted mesh and its actual hand skin.

These are offline renders, not a claim that browser/WebGL checks passed.
The stress pose is authored; an optional measured frame remains geometry_preview.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector
from bpy_extras.object_utils import world_to_camera_view

SIDES = [('left', 'L', -1), ('right', 'R', 1)]
FINGERS = ['ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal', 'IndexProximal', 'IndexIntermediate', 'IndexDistal', 'MiddleProximal', 'MiddleIntermediate', 'MiddleDistal', 'RingProximal', 'RingIntermediate', 'RingDistal', 'LittleProximal', 'LittleIntermediate', 'LittleDistal']
JOINTS = {'hips': 'root', 'spine': 'spine03', 'chest': 'spine01', 'neck': 'neck01', 'head': 'head', 'jaw': 'jaw', 'leftEye': 'eye.L', 'rightEye': 'eye.R'}
for side, suffix, _ in SIDES:
    JOINTS.update({side+'Shoulder': 'clavicle.'+suffix, side+'UpperArm': 'upperarm01.'+suffix,
                   side+'Forearm': 'lowerarm01.'+suffix, side+'Hand': 'wrist.'+suffix})
    JOINTS.update({side+name: f'finger{index//3+1}-{index%3+1}.{suffix}' for index, name in enumerate(FINGERS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('master', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--clip', type=Path); parser.add_argument('--at-ms', type=float, default=1800)
    parser.add_argument('--allow-geometry-preview', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists(): raise ValueError('Choose a new visual qualification directory')
    args.output.mkdir(parents=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.master.resolve()))
    rig = next(obj for obj in bpy.data.objects if obj.type == 'ARMATURE')
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.hide_render and not any(c.name == 'glTF_not_exported' for c in obj.users_collection)]
    skin = max(meshes, key=lambda obj: len(obj.data.vertices))
    scene = bpy.context.scene; camera = scene.camera
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 12
    scene.render.threads_mode = 'FIXED'; scene.render.threads = 3
    scene.render.image_settings.file_format = 'PNG'
    all_points = [obj.matrix_world @ v.co for obj in meshes for v in (obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices)]
    bottom = min(p.z for p in all_points); top = max(p.z for p in all_points); unit = (top-bottom)/2.4
    def bone(control): return rig.pose.bones[JOINTS[control]]
    def point(control): return rig.matrix_world @ bone(control).head
    def aim_arm(control, child, direction):
        target = bone(control); current = point(child)-point(control)
        change = current.normalized().rotation_difference(Vector(direction).normalized())
        orientation = (rig.matrix_world @ target.matrix).to_quaternion()
        target.rotation_mode = 'QUATERNION'; target.rotation_quaternion @= orientation.inverted() @ change @ orientation
        bpy.context.view_layer.update()
    for side, _, sign in SIDES:
        aim_arm(side+'UpperArm', side+'Forearm', (sign*.16, .025, -1))
        aim_arm(side+'Forearm', side+'Hand', (sign*.10, .14, -1))
    neutral = {control: bone(control).rotation_quaternion.copy() for control in JOINTS}
    orientations = {control: (rig.matrix_world @ bone(control).matrix).to_quaternion() for control in JOINTS}
    canonical_to_blender = Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_quaternion()
    def rotate(control, xyzw):
        q = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2]))
        q = canonical_to_blender @ q @ canonical_to_blender.inverted()
        local = orientations[control].inverted() @ q @ orientations[control]
        bone(control).rotation_quaternion = neutral[control] @ local
    def reset():
        for control, q in neutral.items(): bone(control).rotation_quaternion = q
        for obj in meshes:
            if obj.data.shape_keys:
                for key in obj.data.shape_keys.key_blocks[1:]: key.value = 0
        bpy.context.view_layer.update()
    hand_vertex_ids = [v.index for v in skin.data.vertices if any(skin.vertex_groups[group.group].name.startswith(('finger', 'wrist')) and group.weight > .05 for group in v.groups)]
    if len(hand_vertex_ids) < 100: raise ValueError('No usable skinned hand surface was found')
    def hand_bounds():
        evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = evaluated.to_mesh()
        points = [world_to_camera_view(scene, camera, evaluated.matrix_world @ mesh.vertices[i].co) for i in hand_vertex_ids]
        evaluated.to_mesh_clear()
        bounds = {'x': [min(p.x for p in points), max(p.x for p in points)], 'y': [min(p.y for p in points), max(p.y for p in points)], 'minimum_depth': min(p.z for p in points)}
        bounds['inside_frame'] = all(.01 < v < .99 for axis in ['x', 'y'] for v in bounds[axis]) and bounds['minimum_depth'] > 0
        return bounds
    renders = []
    def render(name, width, height, *, portrait=False, angle=0, hands=False, kind='neutral'):
        scene.render.resolution_x = width; scene.render.resolution_y = height; scene.render.resolution_percentage = 100
        if portrait:
            target = Vector((0, 0, top-.13)); distance = 1.1
            camera.location = (math.sin(angle)*distance, math.cos(angle)*distance, target.z)
            camera.data.sensor_fit = 'AUTO'; camera.data.lens = 68
        else:
            aspect = width/height; tangent = math.tan(math.radians(18))
            target = Vector((0, 0, bottom+1.65*unit)); distance = 2.3*unit
            points = [(Vector((x*unit,0,bottom+z*unit)),.06*unit) for x,z in [(-.4,1.05),(.4,1.05),(0,2.4)]]
            for control in JOINTS:
                if control not in ['head','chest'] and (not control.startswith(('left','right')) or control.endswith('Eye')): continue
                radius = .27 if control == 'head' else .22 if control == 'chest' else .12 if any(part in control for part in ['Shoulder','UpperArm','Forearm']) else .08
                points.append((point(control),radius*unit))
            for position,radius in points:
                distance=max(distance,position.y+radius+(abs(position.x)+radius)/(tangent*max(.2,aspect)*.90),position.y+radius+(abs(position.z-target.z)+radius)/(tangent*.90))
            camera.location = (0, distance, target.z)
            camera.data.sensor_fit = 'VERTICAL'; camera.data.lens = camera.data.sensor_height/(2*tangent)
        camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        bounds = hand_bounds() if hands else None
        if bounds and not bounds['inside_frame']: raise ValueError(f'{name} clips the actual skinned hand geometry: {bounds}')
        path = args.output/(name+'.png'); scene.render.filepath = str(path.resolve()); bpy.ops.render.render(write_still=True)
        renders.append({'image': path.name, 'width': width, 'height': height, 'pose_kind': kind, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'hand_bounds_normalized': bounds})
        (args.output/'render-progress.json').write_text(json.dumps(renders, indent=2)+'\n')
    reset()
    for name, angle in [('portrait-front', 0), ('portrait-three-quarter', math.radians(35)), ('portrait-profile', math.radians(80))]:
        render(name, 768, 896, portrait=True, angle=angle)
    clip_report = None
    if args.clip:
        clip = json.loads(args.clip.read_text())
        if clip.get('schema_version') != 'avatar-clip.v1' or (clip.get('review', {}).get('status') != 'speaker_reviewed' and not args.allow_geometry_preview):
            raise ValueError('Measured-frame rendering requires a valid clip and explicit preview opt-in')
        frame = min(clip['frames'], key=lambda frame: abs(frame['time_ms']-args.at_ms))
        for control, xyzw in frame.get('bones', {}).items():
            if control in JOINTS: rotate(control, xyzw)
        for obj in meshes:
            if obj.data.shape_keys:
                for name, value in frame.get('face', {}).items():
                    if name in obj.data.shape_keys.key_blocks: obj.data.shape_keys.key_blocks[name].value = value
        clip_report = {'sha256': hashlib.sha256(args.clip.read_bytes()).hexdigest(), 'source': clip['source'], 'review': clip['review'], 'rendered_frame_ms': frame['time_ms']}
        render('desktop-observed-frame', 1200, 900, hands=True, kind='measured_geometry_preview')
    reset()
    for side, _, sign in SIDES:
        aim_arm(side+'UpperArm', side+'Forearm', (sign*.70, .25, -.60))
        aim_arm(side+'Forearm', side+'Hand', (-sign*.12, .68, .55))
        for index, name in enumerate(FINGERS):
            target = bone(side+name); target.rotation_mode = 'QUATERNION'
            target.rotation_quaternion @= Quaternion((1, 0, 0), .45 if index < 3 else .65)
    render('desktop-hand-stress', 1200, 900, hands=True, kind='authored_articulation_stress')
    render('mobile-hand-stress', 720, 1000, hands=True, kind='authored_articulation_stress')
    report = {'schema_version': 'avatar-offline-visual-qa.v1', 'master_sha256': hashlib.sha256(args.master.read_bytes()).hexdigest(),
              'engine': 'Blender Cycles CPU', 'samples': 12, 'threads': 3, 'renders': renders, 'hand_skin_vertices_checked': len(hand_vertex_ids),
              'measured_frame': clip_report, 'browser_rendering': 'not_tested_by_this_command',
              'human_likeness': 'requires_visual_review', 'framing_method': 'current mapped head/arm/hand joint envelope with explicit skin margins; fixed front camera per render',
              'notes': ['Portrait views use the same asset and lighting.', 'Stress poses are authored mechanical checks, not observed personal gestures.', 'Mobile framing uses the actual pose envelope; full extensions can still require a wider view.']}
    (args.output/'visual-qa-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'renders': len(renders), 'hand_vertices': len(hand_vertex_ids)}))


if __name__ == '__main__': main()
