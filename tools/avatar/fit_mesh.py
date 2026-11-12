"""Blender: bounded coarse ratio fit, authored accessories, editable candidate.

The six deformation parameters fit measured facial proportions. Hair, glasses,
materials and garment details remain authored. The result is not a scan.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.avatar.identity import RATIO_PAIRS, ratios


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template', type=Path); parser.add_argument('template_landmarks', type=Path)
    parser.add_argument('measurements', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=80); parser.add_argument('--resume', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if not 1 <= args.steps <= 500:
        raise ValueError('Coarse fitting budget must be 1–500 iterations')
    if (args.output/'jensen-landmark-candidate-v1.glb').exists():
        raise ValueError('A completed candidate already exists; choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str((args.template/'template.blend').resolve()))
    binding = json.loads((args.template/'template-binding.json').read_text())
    detected = json.loads(args.template_landmarks.read_text())
    data = json.loads(args.measurements.read_text())
    if data.get('persona_id') != 'jensen-huang' or data.get('schema_version') != 'avatar-identity-measurements.v1':
        raise ValueError('Use verified Jensen measurements')
    target = data['target_ratios']
    skin = bpy.data.objects[binding['skin']]; rig = bpy.data.objects[binding['rig']]
    needed = sorted({i for pairs in RATIO_PAIRS.values() for pair in pairs for i in pair})
    anchors = {}
    for index in needed:
        x, y = detected['landmarks'][index][:2]
        candidates = [(v, math.hypot(v['image'][0]-x, v['image'][1]-y)) for v in binding['vertices'] if v['point'][2] > binding['top']-.38]
        closest = min(distance for _, distance in candidates)
        # A contour may project to both sides of the head. Choose the frontmost
        # candidate inside a small pixel neighborhood, not a hidden rear vertex.
        selected = min((v for v, d in candidates if d <= closest + .003), key=lambda v: v['depth'])
        anchors[index] = selected
    original = {index: tuple(row['point']) for index, row in anchors.items()}
    baseline = ratios(original)
    cx = (original[234][0]+original[454][0])/2
    eye_z = (original[33][2]+original[263][2])/2
    mouth_z = (original[61][2]+original[291][2])/2
    chin_z = original[152][2]; nose_z = original[2][2]
    face_h = math.dist(original[10], original[152]); face_w = math.dist(original[234], original[454])
    front_y = original[2][1]
    def deform(point, params):
        x,y,z = point; px=x-cx
        head = max(0., min(1., (z - (chin_z-.20*face_h))/(.20*face_h)))
        front = math.exp(-((y-front_y)/(.65*face_w))**2)
        jaw = math.exp(-((z-(chin_z+.12*face_h))/(.25*face_h))**2)*front
        nose = math.exp(-(px/(.19*face_w))**2-((z-nose_z)/(.18*face_h))**2)*front
        mouth = math.exp(-(px/(.35*face_w))**2-((z-mouth_z)/(.13*face_h))**2)*front
        eyes = math.exp(-((z-eye_z)/(.13*face_h))**2)*front
        x += px*head*(params[0]+params[1]*jaw+params[2]*nose+params[3]*mouth+params[4]*eyes)
        z -= params[5] * max(0., nose_z-z)*head*front
        return (x,y,z)
    def measured(params):
        return ratios({index: deform(point,params) for index,point in original.items()})
    def loss(params):
        current=measured(params)
        return sum(((current[k]-target[k])/target[k])**2 for k in target) + .001*sum(x*x for x in params)
    metadata={'measurement_sha256':digest(args.measurements),'template_binding_sha256':digest(args.template/'template-binding.json'),
              'template_landmarks_sha256':digest(args.template_landmarks),'source_photo_ids':[row['id'] for row in data['accepted']]}
    params=[0.]*6; start=0; history=[]; step=.08
    if args.resume:
        state=json.loads(args.resume.read_text())
        if state['inputs'] != metadata:raise ValueError('Resume inputs differ')
        params=state['parameters'];start=state['iteration'];history=state['history'];step=state['step_size']
    for iteration in range(start,args.steps):
        improved=False
        for axis in range(6):
            options=[params.copy()]
            for sign in [-1,1]:
                candidate=params.copy();candidate[axis]=max(-.30,min(.30,candidate[axis]+sign*step));options.append(candidate)
            best=min(options,key=loss)
            if loss(best)<loss(params)-1e-12:improved=True;params=best
        if not improved:step=max(.0001,step*.75)
        history.append({'iteration':iteration+1,'objective':loss(params),'step_size':step})
        if (iteration+1)%10==0 or iteration+1==args.steps:
            atomic(args.output/f'fit-step-{iteration+1:04d}.json',{'schema_version':'avatar-ratio-fit-state.v1','inputs':metadata,
                   'iteration':iteration+1,'parameters':params,'history':history,'step_size':step})
    meshes=[obj for obj in bpy.data.objects if obj.type=='MESH']
    # Apply the same deformation to each base and expression vertex. Fitting a
    # neutral mesh while leaving expression shapes behind would tear the face.
    for obj in meshes:
        transform=obj.matrix_world; inverse=transform.inverted()
        collections=[key.data for key in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else [obj.data.vertices]
        for vertices in collections:
            for vertex in vertices:vertex.co=inverse@Vector(deform(transform@vertex.co,params))
    # Preserve skin bind consistency after changing the rest head/eye locations.
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT');inverse=rig.matrix_world.inverted()
    for bone in rig.data.edit_bones:
        bone.head=inverse@Vector(deform(rig.matrix_world@bone.head,params))
        bone.tail=inverse@Vector(deform(rig.matrix_world@bone.tail,params))
    bpy.ops.object.mode_set(mode='OBJECT')
    def plain_material(name,color,roughness,metallic=0):
        material=bpy.data.materials.new(name);material.use_nodes=True
        shader=material.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(*color,1)
        shader.inputs['Roughness'].default_value=roughness;shader.inputs['Metallic'].default_value=metallic
        return material
    # Color/material choices are authored; no source photograph pixels are baked.
    hair_material=plain_material('JensenAuthoredSilverHair',(.52,.53,.51),.68)
    jacket=plain_material('JensenAuthoredDarkJacket',(.012,.014,.018),.30)
    for obj in meshes:
        names=' '.join(mat.name.lower() for mat in obj.data.materials if mat)
        if 'hair' in names:
            for mat in obj.data.materials:
                shader=mat.node_tree.nodes.get('Principled BSDF') if mat.use_nodes else None
                if shader:
                    for link in list(shader.inputs['Base Color'].links):mat.node_tree.links.remove(link)
                    shader.inputs['Base Color'].default_value=(.48,.50,.49,1)
        if 'clothes' in names:obj.data.materials.clear();obj.data.materials.append(jacket)
    glasses=plain_material('JensenAuthoredTitaniumGlasses',(.065,.055,.038),.28,.65)
    def tube(name,points,radius,material):
        curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.resolution_u=2
        spline=curve.splines.new('POLY');spline.points.add(len(points)-1)
        for point,co in zip(spline.points,points):point.co=(*co,1)
        curve.bevel_depth=radius;curve.bevel_resolution=3
        obj=bpy.data.objects.new(name,curve);bpy.context.collection.objects.link(obj);curve.materials.append(material)
        bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
        bpy.ops.object.convert(target='MESH');obj=bpy.context.object
        matrix=obj.matrix_world.copy();obj.parent=rig;obj.parent_type='BONE';obj.parent_bone='head';obj.matrix_world=matrix
        meshes.append(obj);return obj
    # Use fitted outer-eye anchors for scale. Slightly in front of the face;
    # the rigid frames follow the head and remain independent of eyelid morphs.
    eyes_span=math.dist(deform(original[33],params),deform(original[263],params))
    lens_w=eyes_span*.42;lens_h=face_h*.105;centers=[-eyes_span*.275,eyes_span*.275]
    glass_y=max(deform(original[33],params)[1],deform(original[263],params)[1])+face_h*.07
    glass_z=eye_z
    for side,center in zip(['Right','Left'],centers):
        points=[]
        for i in range(65):
            a=i/64*math.tau;c=math.cos(a);s=math.sin(a)
            points.append((cx+center+math.copysign(abs(c)**.65,c)*lens_w*.5,glass_y,glass_z+math.copysign(abs(s)**.65,s)*lens_h*.5))
        tube('AuthoredGlasses'+side,points,face_h*.0035,glasses)
        x=cx+center+(lens_w*.5 if center>0 else -lens_w*.5)
        tube('AuthoredTemple'+side,[(x,glass_y,glass_z),(x*1.05,glass_y-.06,glass_z-.004)],face_h*.0028,glasses)
    tube('AuthoredGlassesBridge',[(cx-lens_w*.12,glass_y,glass_z+.004),(cx,glass_y+.002,glass_z+.007),(cx+lens_w*.12,glass_y,glass_z+.004)],face_h*.0035,glasses)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True)
    for obj in meshes:obj.select_set(True)
    bpy.context.view_layer.objects.active=rig
    glb=args.output/'jensen-landmark-candidate-v1.glb'
    bpy.ops.export_scene.gltf(filepath=str(glb.resolve()),export_format='GLB',use_selection=True,export_animations=False,export_skins=True,export_morph=True,export_morph_normal=True)
    master=args.output/'jensen-landmark-candidate-v1.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(master.resolve()))
    bpy.context.scene.render.filepath=str((args.output/'candidate-neutral.png').resolve());bpy.ops.render.render(write_still=True)
    report={'schema_version':'avatar-identity-fit.v1','persona_id':'jensen-huang','asset_kind':'coarse_landmark_fitted_candidate',
            'inputs':metadata,'method':'six-parameter bounded regional deformation fitted to event-balanced photo/video landmark ratios',
            'iterations':args.steps,'parameter_names':['head_width','jaw_width','nose_width','mouth_width','eye_spacing','lower_face_length'],
            'parameters':params,'target_ratios':target,'baseline_ratios':baseline,'fitted_ratios':measured(params),
            'baseline_objective':loss([0.]*6),'final_objective':loss(params),'history':history,'anchors':{str(i):row['index'] for i,row in anchors.items()},
            'glb':{'path':glb.name,'sha256':digest(glb),'bytes':glb.stat().st_size},'editable_master':{'path':master.name,'sha256':digest(master),'bytes':master.stat().st_size},
            'authored_regions':['silver hair material','titanium-style glasses','dark jacket material'],
            'likeness_review':'ungraded','promotion':'candidate_only',
            'limitations':['Weak-perspective inferred face geometry is not photogrammetry or a metric scan.',
                           'Six coarse parameters cannot reproduce fine facial identity, wrinkles or facial texture.',
                           'Glasses, hair and garment details require visual refinement.',
                           'Landmark residual reduction does not demonstrate human likeness or animation quality.']}
    atomic(args.output/'identity-fit-report.json',report)
    print(json.dumps({'output':str(args.output),'initial':report['baseline_objective'],'final':report['final_objective'],'bytes':glb.stat().st_size}))


if __name__=='__main__':main()
