"""Run with Blender 4.3.2: author a CC0 humanoid and native expression targets.

The parameters below are authored design choices, not measurements of a person.
MPFB remains an external, separately licensed build dependency.
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
from tools.avatar.fetch import fetch_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/local/avatar-inputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local/avatar-build"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    root = args.inputs.resolve()
    verification = fetch_inputs(root, verify_only=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / "mpfb2-6ff1eb358f6c1959ced77e79e6dd2406d9a71f49/src/mpfb"
    extension_directory = output / "extensions"
    extension_directory.mkdir(exist_ok=True)
    link = extension_directory / "mpfb"
    if not link.exists():
        link.symlink_to(source, target_is_directory=True)
    if link.resolve() != source:
        raise ValueError("Build extension points to a different MPFB checkout")
    bpy.ops.preferences.extension_repo_add(name="Bias Build", use_custom_directory=True,
        custom_directory=str(extension_directory), type="LOCAL")
    entry = next(r for r in bpy.context.preferences.extensions.repos if r.directory == str(extension_directory))
    module = "bl_ext." + entry.module + ".mpfb"
    bpy.ops.preferences.addon_enable(module=module)
    mpfb = sys.modules[module]
    services = mpfb.MPFB_CONTEXTUAL_INFORMATION["SERVICES"]
    human, target = services["HumanService"], services["TargetService"]
    export, face = services["ExportService"], services["FaceService"]
    assets = root / "system-assets"
    mpfb.set_preference("mpfb_second_root", str(assets))
    services["LocationService"].update_second_root()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    macro = target.get_default_macro_info_dict()
    macro.update(gender=1., age=.64, muscle=.55, weight=.66, height=.56)
    # These are upstream mesh-slider keys, not inferred demographic annotations.
    macro["race"] = {"asian": 0., "african": 0., "caucasian": 1.}
    body = human.create_human(macro_detail_dict=macro)
    detail = {
        "head/head-scale-horiz-incr": .13, "head/head-square": .18, "head/head-fat-incr": .08,
        "chin/chin-width-incr": .18, "chin/chin-height-decr": .10, "chin/chin-prominent-incr": .05,
        "nose/nose-width1-decr": .15, "nose/nose-scale-depth-incr": .13, "nose/nose-point-down": .05,
        "mouth/mouth-scale-horiz-incr": .08, "mouth/mouth-upperlip-volume-decr": .12,
        "neck/neck-scale-horiz-incr": .10,
        "eyes/l-eye-height2-decr": .12, "eyes/r-eye-height2-decr": .12,
    }
    for key, value in detail.items():
        target.load_target(body, str(source / "data/targets" / (key + ".target.gz")), weight=value)
    target.bake_targets(body)
    body.name = "ResearchHumanSkin"
    rig = human.add_builtin_rig(body, "default")
    rig.name = "ResearchHumanRig"
    children = {}
    for kind, fragment in [
        ("Eyes", "eyes/high-poly/high-poly.mhclo"),
        ("Eyebrows", "eyebrows/eyebrow001/eyebrow001.mhclo"),
        ("Teeth", "teeth/teeth_base/teeth_base.mhclo"),
        ("Tongue", "tongue/tongue01/tongue01.mhclo"),
        ("Hair", "hair/short01/short01.mhclo"),
        ("Clothes", "clothes/male_casualsuit01/male_casualsuit01.mhclo"),
    ]:
        children[kind] = human.add_mhclo_asset(str(assets / fragment), body,
            asset_type=kind, subdiv_levels=0, material_type="GAMEENGINE")

    def material(obj, name, texture, roughness, alpha=False):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat.diffuse_color = (.5, .5, .5, 1.)
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = 0
        texture_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        texture_node.image = bpy.data.images.load(str(assets / texture), check_existing=True)
        mat.node_tree.links.new(texture_node.outputs["Color"], bsdf.inputs["Base Color"])
        if alpha:
            mat.node_tree.links.new(texture_node.outputs["Alpha"], bsdf.inputs["Alpha"])
            mat.surface_render_method = "DITHERED"
            mat.use_backface_culling = False
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    material(body, "Skin", "skins/middleage_caucasian_male/middleage_lightskinned_male_diffuse.png", .58)
    material(children["Eyes"], "Eyes", "eyes/materials/grey_eye.png", .22, True)
    material(children["Eyebrows"], "Brows", "eyebrows/eyebrow001/eyebrow001.png", .74, True)
    material(children["Teeth"], "Teeth", "teeth/teeth_base/teeth.png", .40)
    material(children["Tongue"], "Tongue", "tongue/tongue01/tongue01_diffuse.png", .45)
    material(children["Hair"], "Hair", "hair/short01/short01_diffuse.png", .66, True)
    material(children["Clothes"], "Clothes", "clothes/male_casualsuit01/male_casualsuit01_diffuse.png", .90)

    extras = root / "extra-targets-7eaba3453134385bb5ea9811ef0b33b85b4b556d/assets"
    for directory in [extras / "faceunits01/targets/faceunits", extras / "visemes02/targets"]:
        for path in sorted(directory.rglob("*.target")):
            target.load_target(body, str(path), name=path.stem)
    face.interpolate_targets(body)
    export.bake_modifiers_remove_helpers(body, bake_masks=True, remove_helpers=True)
    meshes = [body, *children.values()]
    for mesh in meshes:
        for polygon in mesh.data.polygons:
            polygon.use_smooth = True
        if mesh.data.shape_keys:
            basis = mesh.data.shape_keys.key_blocks[0]
            for key in list(mesh.data.shape_keys.key_blocks)[1:]:
                if key.name != "viseme_sil" and all((p.co - q.co).length < 1e-7 for p, q in zip(key.data, basis.data)):
                    mesh.shape_key_remove(key)
        for modifier in list(mesh.modifiers):
            if modifier.type == "SUBSURF":
                mesh.modifiers.remove(modifier)

    # Preserve the authored rest skeleton; the runtime supplies its relaxed pose.
    rig.rotation_euler[2] = math.pi
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = rig
    out = output / "research-human-v2.glb"
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_selection=True,
        export_animations=False, export_skins=True, export_morph=True,
        export_morph_normal=True, export_apply=False)
    report = {
        "blender": bpy.app.version_string, "inputs": verification,
        "vertices_before_uv_split": sum(len(m.data.vertices) for m in meshes),
        "triangles": sum(sum(len(p.vertices) - 2 for p in m.data.polygons) for m in meshes),
        "bones": len(rig.data.bones),
        "morph_union": sorted({key.name for mesh in meshes if mesh.data.shape_keys for key in list(mesh.data.shape_keys.key_blocks)[1:]}),
        "asset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "asset_bytes": out.stat().st_size,
        "authored_macro": macro, "authored_detail": detail,
        "likeness_review": "not_performed", "asset_kind": "authored_anatomical_proxy",
    }
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.resolution_x, scene.render.resolution_y = 720, 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (.06, .07, .06)
    scene.view_settings.view_transform = "AgX"

    def aim(obj, at):
        obj.rotation_euler = (Vector(at) - obj.location).to_track_quat("-Z", "Y").to_euler()

    bpy.ops.object.camera_add(location=(.07, 2.0, 1.65))
    camera = bpy.context.object
    camera.name = "ReviewCamera"
    camera.data.lens = 85
    aim(camera, (0, 0, 1.55))
    scene.camera = camera
    for position, energy, size in [((2.8, 3.0, 3.5), 650, 3.), ((-2, 2.5, 2.), 300, 2.8), ((.2, -2.2, 2.6), 500, 2.2)]:
        bpy.ops.object.light_add(type="AREA", location=position)
        light = bpy.context.object
        light.data.energy, light.data.shape, light.data.size = energy, "DISK", size
        aim(light, (0, 0, 1.4))
    scene.render.filepath = str(output / "neutral.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "research-human-v2.blend"))
    if args.render:
        bpy.ops.render.render(write_still=True)
    print(json.dumps({"asset": str(out), "bytes": report["asset_bytes"], "native_targets": len(report["morph_union"])}))


if __name__ == "__main__":
    main()
