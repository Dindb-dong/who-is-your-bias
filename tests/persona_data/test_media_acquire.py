import pytest

from tools.persona_data.media_acquire import hls_manifest


def test_hls_manifest_requires_complete_unencrypted_vod():
    complete = b"#EXTM3U\n#EXTINF:4.0,\na.ts\n#EXTINF:2.5,\nb.ts\n#EXT-X-ENDLIST\n"
    assert hls_manifest(complete) == (6.5, 2)
    with pytest.raises(ValueError, match="finite"):
        hls_manifest(complete.replace(b"#EXT-X-ENDLIST", b""))
    with pytest.raises(ValueError, match="Encrypted"):
        hls_manifest(complete + b'#EXT-X-KEY:METHOD=AES-128,URI="key"\n')
