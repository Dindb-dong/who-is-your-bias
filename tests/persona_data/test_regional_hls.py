import hashlib

import pytest

from tools.persona_data.regional_hls import parse_registered_playlist


def registered(raw):
    return {"url": "https://publisher.example/vod/replay.m3u8",
            "publisher_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_fragments": 2, "expected_duration_seconds": 12.5}


VALID = b"#EXTM3U\n#EXTINF:10.0,\n00001.ts\n#EXTINF:2.5,\n00002.ts\n#EXT-X-ENDLIST\n"


def test_accepts_complete_registered_playlist_in_order():
    assert parse_registered_playlist(VALID, registered(VALID)) == [
        "https://publisher.example/vod/00001.ts", "https://publisher.example/vod/00002.ts"]


@pytest.mark.parametrize("raw", [
    VALID.replace(b"#EXT-X-ENDLIST\n", b""),
    VALID.replace(b"00002.ts", b"00001.ts"),
    VALID.replace(b"00002.ts", b"https://different.example/00002.ts"),
    VALID.replace(b"00002.ts", b"http://publisher.example/00002.ts"),
    VALID.replace(b"#EXTINF:2.5,", b"#EXT-X-DISCONTINUITY\n#EXTINF:2.5,"),
    VALID.replace(b"#EXTINF:2.5,", b'#EXT-X-KEY:METHOD=AES-128,URI="key"\n#EXTINF:2.5,'),
])
def test_rejects_incomplete_duplicated_changed_origin_or_nonplain_media(raw):
    with pytest.raises(ValueError):
        parse_registered_playlist(raw, registered(raw))


@pytest.mark.parametrize("field,value", [
    ("publisher_manifest_sha256", "0" * 64),
    ("expected_fragments", 3),
    ("expected_duration_seconds", 13.5),
])
def test_rejects_registration_drift(field, value):
    entry = registered(VALID)
    entry[field] = value
    with pytest.raises(ValueError):
        parse_registered_playlist(VALID, entry)
