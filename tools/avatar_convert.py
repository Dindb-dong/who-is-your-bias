"""Convert a lawfully acquired local FBX/Blend/glTF asset without changing its source.

    python tools/avatar_convert.py candidate.fbx candidate.glb --report conversion.json

Blender's Python auto-execution is disabled. The receipt records exact input/output
bytes and actual export settings; it does not assess ownership or visual likeness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def glb_inventory(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20 or len(data) > 64 * 1024 * 1024:
        raise ValueError("Converted GLB must be complete and at most 64 MiB.")
    magic, version, length, json_length, json_type = struct.unpack_from("<5I", data)
    if (magic, version, length, json_type) != (0x46546C67, 2, len(data), 0x4E4F534A):
        raise ValueError("Conversion did not produce a valid GLB header.")
    if json_length % 4 or 20 + json_length > len(data):
        raise ValueError("Invalid converted JSON chunk.")
    doc = json.loads(data[20:20 + json_length])
    resources = doc.get("images", []) + doc.get("buffers", [])
    if any(resource.get("uri") and not resource["uri"].startswith("data:") for resource in resources):
        raise ValueError("Converted GLB retains an external texture/buffer reference.")
    return {
        "mesh_count": len(doc.get("meshes", [])),
        "skin_count": len(doc.get("skins", [])),
        "morph_names": sorted({name for mesh in doc.get("meshes", []) for name in mesh.get("extras", {}).get("targetNames", [])}),
        "joint_names": sorted({doc["nodes"][index].get("name", "") for skin in doc.get("skins", []) for index in skin.get("joints", [])}),
        "material_count": len(doc.get("materials", [])),
        "embedded_images": len(doc.get("images", [])),
        "animation_count": len(doc.get("animations", [])),
    }


def source_resources(source: Path) -> list[dict]:
    """Bind external glTF buffers/images before import; never download resources."""
    if source.suffix.lower() != ".gltf":
        return []
    document = json.loads(source.read_text())
    records = []
    for item in document.get("buffers", []) + document.get("images", []):
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        parsed = urlsplit(uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("glTF conversion accepts local package resources only; embed remote resources first.")
        member = (source.parent / unquote(parsed.path)).resolve()
        if not member.is_relative_to(source.parent) or not member.is_file():
            raise ValueError("Keep glTF buffers and textures within the selected source package folder.")
        records.append({"path": str(member.relative_to(source.parent)), "sha256": digest(member), "bytes": member.stat().st_size})
    return records


def worker(source: Path, destination: Path, report: Path) -> None:
    import bpy  # type: ignore[import-not-found]

    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    if source.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
    else:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        if source.suffix.lower() == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(source), use_anim=True)
        else:
            bpy.ops.import_scene.gltf(filepath=str(source))
    objects = list(bpy.context.scene.objects)
    morph_names = sorted({key.name for obj in objects if obj.type == "MESH" and obj.data.shape_keys
                          for key in list(obj.data.shape_keys.key_blocks)[1:]})
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    before = {
        "scene": bpy.context.scene.name,
        "mesh_count": sum(obj.type == "MESH" for obj in objects),
        "armatures": len(armatures),
        "joint_names": sorted({bone.name for obj in armatures for bone in obj.data.bones}),
        "morph_names": morph_names,
        "material_count": len({slot.material.name for obj in objects if obj.type == "MESH"
                               for slot in obj.material_slots if slot.material}),
    }
    input_images = []
    for image in bpy.data.images:
        if image.name in {"Render Result", "Viewer Node"}:
            continue
        packed = image.packed_file
        external = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
        input_images.append({"name": image.name, "packed": bool(packed),
                             "sha256": hashlib.sha256(packed.data).hexdigest() if packed else
                             digest(external) if external and external.is_file() else None,
                             "available": bool(packed or (external and external.is_file()))})
    if not before["mesh_count"]:
        raise ValueError("The active scene contains no mesh to convert.")
    settings = {
        "export_format": "GLB", "use_active_scene": True, "use_selection": False,
        "export_animations": True, "export_skins": True, "export_morph": True,
        "export_morph_normal": True, "export_morph_animation": True,
        "export_rest_position_armature": True, "export_apply": False,
        "export_materials": "EXPORT", "export_image_format": "AUTO",
        "export_yup": True, "export_cameras": False, "export_lights": False,
    }
    supported = {prop.identifier for prop in bpy.ops.export_scene.gltf.get_rna_type().properties}
    missing = set(settings) - supported
    if missing:
        raise ValueError(f"This Blender exporter is missing required settings: {sorted(missing)}")
    outcome = bpy.ops.export_scene.gltf(filepath=str(destination), **settings)
    if "FINISHED" not in outcome:
        raise ValueError(f"GLB export did not finish: {outcome}")
    after = glb_inventory(destination)
    missing_targets = sorted(set(morph_names) - set(after["morph_names"]))
    if missing_targets:
        raise ValueError(f"Export lost these shape keys: {missing_targets}")
    if armatures and not after["skin_count"]:
        raise ValueError("The source contains an armature but no skinned mesh survived export.")
    report.write_text(json.dumps({
        "blender_version": bpy.app.version_string, "python_auto_execution": False,
        "settings": settings, "before": before, "after": after, "input_images": input_images,
    }, indent=2) + "\n")


def convert(source: Path, destination: Path, report: Path, *, blender: str = "blender", timeout: int = 300) -> dict:
    source = source.resolve(); destination = destination.resolve(); report = report.resolve()
    if not source.is_file() or source.suffix.lower() not in {".fbx", ".blend", ".gltf", ".glb"}:
        raise ValueError("Choose an existing FBX, Blend, glTF or GLB file.")
    if destination.suffix.lower() != ".glb" or len({source, destination, report}) != 3:
        raise ValueError("Use distinct input, .glb output and report paths.")
    if destination.exists() or report.exists():
        raise ValueError("Choose new output/report paths; existing files are never overwritten.")
    binary = shutil.which(blender)
    if not binary:
        raise ValueError("Blender is not installed; pass --blender with its executable path.")
    source_sha = digest(source); resource_hashes = source_resources(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bias-avatar-convert-", dir=destination.parent) as directory:
        temporary = Path(directory)
        output = temporary / "candidate.glb"; receipt = temporary / "conversion.json"
        command = [binary, "--background", "--factory-startup", "--disable-autoexec", "--python", str(Path(__file__).resolve()),
                   "--", "--worker", str(source), str(output), str(receipt)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        if process.returncode or not output.is_file() or not receipt.is_file():
            details = (process.stdout + "\n" + process.stderr)[-6000:]
            raise ValueError(f"Blender conversion failed.\n{details}")
        metadata = json.loads(receipt.read_text())
        if source_sha != digest(source) or resource_hashes != source_resources(source):
            raise ValueError("A source file changed during conversion; repeat with an immutable package.")
        result = {
            "schema_version": "avatar-conversion.v1", "input": {"filename": source.name, "sha256": source_sha, "bytes": source.stat().st_size,
                                                                   "external_resources": resource_hashes},
            "output": {"filename": destination.name, "sha256": digest(output), "bytes": output.stat().st_size},
            "elapsed_seconds": round(time.monotonic() - started, 3), **metadata,
            "review": {"materials": "ungraded", "deformation": "requires_browser_audition", "likeness": "ungraded"},
            "limitations": ["Only the active Blender scene is exported.", "DCC shaders may differ from glTF PBR materials.",
                            "Shape-name preservation does not establish dynamic facial quality.",
                            "Run the GLB qualification CLI and inspect the browser result before selection."],
        }
        # Hard-link creation fails if a concurrent process created the destination.
        os.link(output, destination)
        try:
            with report.open("x") as stream:
                json.dump(result, stream, indent=2); stream.write("\n")
        except BaseException:
            destination.unlink()
            raise
        return result


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if args and args[0] == "--worker":
        if len(args) != 4:
            raise ValueError("Invalid Blender worker arguments.")
        worker(*(Path(arg) for arg in args[1:])); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True); parser.add_argument("--blender", default="blender")
    parser.add_argument("--timeout", type=int, default=300)
    options = parser.parse_args(args)
    result = convert(options.input, options.output, options.report, blender=options.blender, timeout=options.timeout)
    print(json.dumps({"output": str(options.output), "sha256": result["output"]["sha256"], "report": str(options.report)}))


if __name__ == "__main__":
    main()
