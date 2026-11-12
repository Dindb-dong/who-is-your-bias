"""Blender: explicit authored appearance corrections after the measured face bake.

The saved v3b candidate remains the immutable input. Iris, hair and brow edits
are authored reference-informed choices, not measured identity-fit results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(image):
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    return values.reshape(image.size[1], image.size[0], 4)


def replace_image(material, values, name, output):
    node = next(n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image)
    image = bpy.data.images.new(name, width=values.shape[1], height=values.shape[0], alpha=True)
    image.colorspace_settings.name = 'sRGB'; image.pixels.foreach_set(values.ravel()); image.update()
    image.filepath_raw = str((output/(name+'.png')).resolve()); image.file_format = 'PNG'; image.save(); image.pack()
    node.image = image
    shader = material.node_tree.nodes.get('Principled BSDF')
    material.node_tree.links.new(node.outputs['Color'], shader.inputs['Base Color'])
    return shader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('master', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists(): raise ValueError('Choose a new appearance candidate directory')
    # This correction is specifically calibrated to the saved v3b texture atlas.
    if digest(args.master) != '9f08cd7abebca4a610683acc730bad9c69c6874996d82a9ffd5a1f237867b0aa':
        raise ValueError('Appearance correction requires the exact preserved v3b master')
    args.output.mkdir(parents=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.master.resolve()))
    eyes = bpy.data.materials['Eyes']; eye_image = next(n.image for n in eyes.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image)
    eye = pixels(eye_image).copy(); height, width, _ = eye.shape
    yy, xx = np.mgrid[0:height, 0:width]; uv = np.stack([xx/max(1,width-1), yy/max(1,height-1)], axis=-1)
    # The CC0 grey-eye atlas has two circular irises. Cover the complete iris,
    # including neutral gray fibres that the former blue-only selector missed.
    distance = np.minimum.reduce([np.linalg.norm(uv-center, axis=-1) for center in [(0.295, .300), (.704, .705)]])
    mask = np.clip((.122-distance)/.007, 0, 1) * (eye[:,:,3] > .1)
    luminance = np.mean(eye[:,:,:3], axis=-1)
    target = np.clip(luminance, .02, .80)[:,:,None] * np.array([.20, .105, .045])
    # Keep very dark pupil centres; sclera and all transparency are unchanged.
    target[luminance < .035] = eye[:,:,:3][luminance < .035]
    eye[:,:,:3] = eye[:,:,:3]*(1-mask[:,:,None]) + target*mask[:,:,None]
    eye_shader = replace_image(eyes, eye, 'JensenDarkBrownIris', args.output)
    eye_shader.inputs['Roughness'].default_value = .32
    eye_shader.inputs['Specular IOR Level'].default_value = .30

    hair = bpy.data.materials['Hair']; hair_image = next(n.image for n in hair.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image)
    hair_pixels = pixels(hair_image).copy(); gray = np.mean(hair_pixels[:,:,:3], axis=-1)
    visible = hair_pixels[:,:,3] > .35
    lo, hi = np.quantile(gray[visible], [.05, .95])
    strands = np.clip((gray-lo)/max(.001, hi-lo), 0, 1)
    # Reconnect the original strand detail as dark/gray salt-and-pepper color.
    # The old flat .48 linear base color washed out to white under the key light.
    shade = .065 + .32*strands**1.1
    hair_pixels[:,:,:3] = shade[:,:,None]*np.array([.98, 1, .97])
    hair_shader = replace_image(hair, hair_pixels, 'JensenSaltPepperHair', args.output)
    hair_shader.inputs['Roughness'].default_value = .83
    hair_shader.inputs['Specular IOR Level'].default_value = .22

    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not any(c.name == 'glTF_not_exported' for c in obj.users_collection)]
    skin = max(meshes, key=lambda obj: len(obj.data.vertices))
    base = skin.data.shape_keys.key_blocks[0].data
    surface = BVHTree.FromPolygons([skin.matrix_world @ v.co for v in base], [tuple(p.vertices) for p in skin.data.polygons])
    trimmed = 0; maximum_shift = 0
    for obj in meshes:
        if not any(material and material.name == 'Hair' for material in obj.data.materials): continue
        data = obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices
        inverse = obj.matrix_world.inverted()
        for index, vertex in enumerate(data):
            world = obj.matrix_world @ vertex.co; nearest = surface.find_nearest(world)
            if nearest[0] is None: continue
            point, normal, _, _ = nearest; clearance = (world-point).dot(normal)
            shift = normal * (max(0, clearance-.002)*.35)
            if shift.length < 1e-9: continue
            local_shift = inverse.to_3x3() @ shift
            for key in obj.data.shape_keys.key_blocks if obj.data.shape_keys else []: key.data[index].co -= local_shift
            if not obj.data.shape_keys: vertex.co -= local_shift
            trimmed += 1; maximum_shift = max(maximum_shift, shift.length)

    brows = []
    for obj in meshes:
        if 'eyebrow001' not in obj.name: continue
        obj.hide_render = False; obj.hide_set(False); brows.append(obj.name)
        basis = obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices
        points = [obj.matrix_world @ v.co for v in basis]
        centers = {side: np.median([p.z for p in points if (p.x >= 0) == side]) for side in [True, False]}
        for index, (vertex, point) in enumerate(zip(basis, points)):
            target_z = centers[point.x >= 0]+(point.z-centers[point.x >= 0])*.48
            local_shift = obj.matrix_world.inverted().to_3x3() @ type(point)((0,0,target_z-point.z))
            for key in obj.data.shape_keys.key_blocks if obj.data.shape_keys else []: key.data[index].co += local_shift
            if not obj.data.shape_keys: vertex.co += local_shift
    brow_material = bpy.data.materials['Brows']
    brow_image = next(n.image for n in brow_material.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image)
    brow_pixels = pixels(brow_image).copy(); brow_gray = np.mean(brow_pixels[:,:,:3], axis=-1)
    brow_pixels[:,:,:3] = (.10+.22*brow_gray)[:,:,None]*np.array([.95, 1, .94])
    brow_shader = replace_image(brow_material, brow_pixels, 'JensenSubtleGrayBrows', args.output)
    brow_shader.inputs['Roughness'].default_value = .84

    rig = next(obj for obj in bpy.data.objects if obj.type == 'ARMATURE')
    bpy.context.view_layer.update(); bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True)
    for obj in meshes:
        if not obj.hide_render: obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    glb=args.output/'jensen-source-fitted-candidate-v3.glb'; master=args.output/'jensen-source-fitted-candidate-v3.blend'
    bpy.ops.export_scene.gltf(filepath=str(glb.resolve()), export_format='GLB', use_selection=True, export_animations=False, export_skins=True, export_morph=True, export_morph_normal=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(master.resolve()))
    bpy.context.scene.render.filepath=str((args.output/'candidate-neutral.png').resolve()); bpy.ops.render.render(write_still=True)
    report={'schema_version':'avatar-authored-appearance.v1','persona_id':'jensen-huang','input_master_sha256':digest(args.master),
            'license':'CC-BY-SA-4.0','source_texture_unchanged':True,'basis':'authored reference-informed edits; not measured identity fitting',
            'iris_pixels':int(np.sum(mask>.1)),'iris_atlas_centers':[[.295,.300],[.704,.705]],'iris_radius':.122,
            'hair_texture_range':[.065,.385],'hair_trimmed_vertices':trimmed,'maximum_hair_shift':maximum_shift,'hair_surface_clearance':.002,
            'authored_brow_height_scale':.48,'restored_brow_meshes':brows,
            'glb':{'filename':glb.name,'sha256':digest(glb),'bytes':glb.stat().st_size},
            'editable_master':{'filename':master.name,'sha256':digest(master),'bytes':master.stat().st_size},
            'promotion':'candidate_only','likeness_review':'pending_visual_review',
            'limitations':['Hair and brows remain authored generic geometry with source-informed appearance.','Reference-photo lighting remains baked into the facial texture.','No identity-recognition or human likeness threshold has passed.']}
    (args.output/'appearance-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'bytes':glb.stat().st_size}))


if __name__=='__main__': main()
