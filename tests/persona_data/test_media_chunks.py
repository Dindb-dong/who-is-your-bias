import json

import pytest

from tools.persona_data.media_chunks import split, verify


def test_lossless_recovery_and_changed_bytes_rejection(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(bytes(range(256)) * 10)
    chunks = tmp_path / "chunks"
    manifest = split(source, chunks, chunk_bytes=501)
    restored = tmp_path / "restored.mp4"
    assert verify(chunks, output=restored)["verified"]
    assert restored.read_bytes() == source.read_bytes()
    part = chunks / manifest["parts"][2]["file"]
    part.write_bytes(part.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Chunk bytes or hash"):
        verify(chunks, output=tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()
    assert not (tmp_path / "bad.mp4.part").exists()


def test_changed_order_is_not_a_complete_video(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"abcd1234")
    chunks = tmp_path / "chunks"
    manifest = split(source, chunks, chunk_bytes=4)
    manifest["parts"].reverse()
    (chunks / "reconstruction.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="order or offset"):
        verify(chunks)
