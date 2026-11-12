"""Lossless byte chunks for durable full-video storage and verified recovery."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _chunk_stream(path: Path, name: str, *, archive: bool):
    if archive and path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as container:
            members = container.getmembers()
            if len(members) != 1 or members[0].name != name or not members[0].isfile():
                raise ValueError("Chunk tar must contain exactly its registered regular file")
            with container.extractfile(members[0]) as stream:
                yield stream
    else:
        with (gzip.open(path, "rb") if archive else path.open("rb")) as stream:
            yield stream


def split(source: Path, output: Path, *, chunk_bytes: int = 200 * 1024 * 1024) -> dict:
    if output.exists() or chunk_bytes < 1:
        raise ValueError("Choose a new chunk directory and positive chunk size")
    output.mkdir(parents=True)
    parts, whole, offset = [], hashlib.sha256(), 0
    with source.open("rb") as stream:
        while data := stream.read(chunk_bytes):
            name = f"{source.name}.part{len(parts) + 1:03d}.bin"
            (output / name).write_bytes(data)
            parts.append({"index": len(parts), "file": name, "offset": offset,
                          "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            whole.update(data)
            offset += len(data)
    manifest = {"schema_version": "persona-media-byte-chunks.v1", "original_file": source.name,
                "bytes": offset, "sha256": whole.hexdigest(), "chunk_bytes": chunk_bytes,
                "encoding": "raw_contiguous_bytes", "parts": parts}
    (output / "reconstruction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(output)
    return manifest


def verify(directory: Path, *, output: Path | None = None, gzip_archives: bool = False) -> dict:
    manifest = json.loads((directory / "reconstruction.json").read_text())
    if manifest.get("schema_version") != "persona-media-byte-chunks.v1" or not manifest.get("parts"):
        raise ValueError("Expected a nonempty byte-chunk manifest")
    if output is not None and (output.exists() or output.with_name(output.name + ".part").exists()):
        raise ValueError("Reconstruction output already exists")
    whole, offset = hashlib.sha256(), 0
    partial = output.with_name(output.name + ".part") if output is not None else None
    stream = partial.open("xb") if partial is not None else None
    try:
        for index, item in enumerate(manifest["parts"]):
            name = item["file"]
            path = directory / (name + ".gz" if gzip_archives else name)
            if gzip_archives and not path.exists():
                path = directory / (name + ".tar.gz")
            if (Path(name).name != name or path.is_symlink() or item["index"] != index
                    or item["offset"] != offset):
                raise ValueError("Chunk path, order or offset changed")
            digest, size = hashlib.sha256(), 0
            with _chunk_stream(path, name, archive=gzip_archives) as chunk:
                while block := chunk.read(4 * 1024 * 1024):
                    digest.update(block)
                    whole.update(block)
                    size += len(block)
                    if stream is not None:
                        stream.write(block)
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError("Chunk bytes or hash mismatch")
            offset += size
        if offset != manifest["bytes"] or whole.hexdigest() != manifest["sha256"]:
            raise ValueError("Reconstructed full-file bytes or hash mismatch")
    except BaseException:
        if stream is not None:
            stream.close()
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise
    if stream is not None:
        stream.close()
        partial.rename(output)
    return {"bytes": offset, "sha256": whole.hexdigest(), "parts": len(manifest["parts"]), "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("split")
    make.add_argument("source", type=Path)
    make.add_argument("directory", type=Path)
    make.add_argument("--chunk-bytes", type=int, default=200 * 1024 * 1024)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("--output", type=Path)
    check.add_argument("--gzip", action="store_true", help="Read .bin.gz or single-member .bin.tar.gz without intermediate raw chunks")
    args = parser.parse_args()
    if args.command == "split":
        result = split(args.source, args.directory, chunk_bytes=args.chunk_bytes)
        print(json.dumps({key: result[key] for key in ("original_file", "bytes", "sha256")}))
    else:
        print(json.dumps(verify(args.directory, output=args.output, gzip_archives=args.gzip)))


if __name__ == "__main__":
    main()
