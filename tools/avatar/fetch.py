"""Fetch only the declared CC0 graphical inputs and the external MPFB build tool."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import urllib.request
import zlib

MANIFEST = Path(__file__).with_name("inputs.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def child(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name:
        raise ValueError("Input manifest contains an unsafe relative path")
    destination = (root / path).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError("Input path escapes the destination")
    return destination


def verify(data: bytes, row: dict) -> None:
    if len(data) != row["size"] or digest(data) != row["sha256"]:
        raise ValueError(f"Pinned input differs: {row.get('path', row.get('name'))}")


def unpack_member(blob: bytes, row: dict) -> bytes:
    if len(blob) < 30:
        raise ValueError("ZIP member header is incomplete")
    header = struct.unpack("<4s5H3L2H", blob[:30])
    if header[0] != b"PK\x03\x04" or header[2] & 1 or header[3] != row["compression"]:
        raise ValueError("ZIP member header differs")
    if blob[30:30 + header[9]].decode("utf-8") != row["path"]:
        raise ValueError("ZIP member path differs")
    offset = 30 + header[9] + header[10]
    compressed = blob[offset:offset + row["compressed"]]
    if len(compressed) != row["compressed"]:
        raise ValueError("ZIP member data is incomplete")
    if row["compression"] == 8:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(compressed, row["size"] + 1)
        if not decoder.eof or decoder.unconsumed_tail:
            raise ValueError("ZIP member decompression exceeded the declared size")
    elif row["compression"] == 0:
        data = compressed
    else:
        raise ValueError("Unsupported pinned ZIP member compression")
    if zlib.crc32(data) != row["crc32"]:
        raise ValueError("ZIP member CRC differs")
    verify(data, row)
    return data


def fetch_inputs(output: Path, *, verify_only: bool = False) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    output = output.resolve()
    if not verify_only:
        output.mkdir(parents=True, exist_ok=True)
    for archive in manifest["archives"]:
        destination = child(output, archive["name"] + ".tar.gz")
        if not destination.is_file():
            if verify_only:
                raise ValueError(f"Missing input archive: {destination}")
            with urllib.request.urlopen(archive["url"], timeout=60) as response:
                data = response.read(archive["size"] + 1)
            verify(data, archive)
            destination.write_bytes(data)
        verify(destination.read_bytes(), archive)
        directory = child(output, archive["directory"])
        with tarfile.open(destination, "r:gz") as bundle:
            members = bundle.getmembers()
            for member in members:
                resolved = child(output, member.name)
                if not resolved.is_relative_to(directory):
                    raise ValueError("Archive contains an unexpected top-level directory")
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Build archives must contain only files and directories")
            if not directory.exists():
                if verify_only:
                    raise ValueError(f"Missing extracted input: {directory}")
                bundle.extractall(output, filter="data")
            # Also detect modifications to previously extracted tool/asset files.
            for member in members:
                if member.isfile():
                    original = bundle.extractfile(member).read()
                    path = child(output, member.name)
                    if not path.is_file() or digest(path.read_bytes()) != digest(original):
                        raise ValueError(f"Extracted input differs: {member.name}")
    pack = manifest["system_assets"]
    for row in pack["files"]:
        destination = child(output / "system-assets", row["path"])
        if destination.is_file():
            verify(destination.read_bytes(), row)
            continue
        if verify_only:
            raise ValueError(f"Missing system asset: {destination}")
        start = row["offset"]
        end = min(pack["archive_size"] - 1, start + row["compressed"] + 1024)
        request = urllib.request.Request(pack["source_url"], headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = f"bytes {start}-{end}/{pack['archive_size']}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Publisher did not return the declared byte range")
            data = unpack_member(response.read(end - start + 2), row)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return {"input_manifest_sha256": digest(MANIFEST.read_bytes()), "archives_verified": len(manifest["archives"]),
            "system_files_verified": len(pack["files"]), "full_system_archive_downloaded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/local/avatar-inputs"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(fetch_inputs(args.output, verify_only=args.verify_only), indent=2))


if __name__ == "__main__":
    main()
