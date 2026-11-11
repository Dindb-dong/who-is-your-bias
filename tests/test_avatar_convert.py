"""Validate actual conversion and prevent losing or overwriting source assets."""
import json
from pathlib import Path
import shutil
import struct

import pytest

from tools.avatar_convert import convert, glb_inventory


def tiny_glb(path: Path) -> bytes:
    binary = struct.pack("<18f", 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "Head", "mesh": 0}],
           "meshes": [{"name": "Face", "extras": {"targetNames": ["jawOpen"]},
                       "primitives": [{"attributes": {"POSITION": 0}, "targets": [{"POSITION": 1}]}]}],
           "buffers": [{"byteLength": len(binary)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 36}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 2, 0]},
                         {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [0, 0.2, 0]}]}
    text = json.dumps(doc).encode(); text += b" " * (-len(text) % 4)
    contents = struct.pack("<5I", 0x46546C67, 2, 28 + len(text) + len(binary), len(text), 0x4E4F534A) + text
    contents += struct.pack("<2I", len(binary), 0x004E4942) + binary
    path.write_bytes(contents); return contents


def test_conversion_refuses_existing_outputs_and_source_aliases(tmp_path):
    source = tmp_path / "source.glb"; original = tiny_glb(source)
    with pytest.raises(ValueError, match="distinct"):
        convert(source, source, tmp_path / "report.json")
    output = tmp_path / "existing.glb"; output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="never overwritten"):
        convert(source, output, tmp_path / "report.json")
    assert source.read_bytes() == original
    assert output.read_bytes() == b"keep"


def test_conversion_inspection_does_not_accept_truncated_bytes(tmp_path):
    source = tmp_path / "truncated.glb"; contents = tiny_glb(source)
    source.write_bytes(contents[:-4])
    with pytest.raises(ValueError, match="header"):
        glb_inventory(source)


@pytest.mark.skipif(not shutil.which("blender"), reason="Requires the actual Blender glTF importer/exporter")
def test_actual_blender_roundtrip_preserves_shape_and_source(tmp_path):
    source = tmp_path / "source.glb"; original = tiny_glb(source)
    output = tmp_path / "converted.glb"; receipt = tmp_path / "conversion.json"
    result = convert(source, output, receipt, timeout=90)
    assert source.read_bytes() == original
    assert output.read_bytes().startswith(b"glTF")
    assert result["after"]["morph_names"] == ["jawOpen"]
    assert result["settings"]["export_morph"] is True
    assert result["python_auto_execution"] is False
    assert result["review"]["likeness"] == "ungraded"
    assert json.loads(receipt.read_text()) == result
