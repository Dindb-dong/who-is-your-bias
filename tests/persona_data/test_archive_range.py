import hashlib
import json
import subprocess

import httpx
import pytest

from tools.persona_data.archive_range import download_range


def entry():
    return {"source_url": "https://publisher.example/ceremony.iso", "source_byte_start": 3,
            "source_byte_count": 7, "source_total_bytes": 12, "split": "train",
            "excluded_from_frozen_runs": True}


def test_exact_range_and_fresh_bytes(tmp_path):
    source = b"abcdefghijkl"
    requests = []

    def respond(request):
        requests.append(request.headers["range"])
        start, end = map(int, request.headers["range"][6:].split("-"))
        return httpx.Response(206, headers={"content-range": f"bytes {start}-{end}/12",
                                           "etag": '"same"'}, content=source[start:end + 1])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = download_range(entry(), tmp_path / "address.vob", client=client, chunk_bytes=4)
    assert requests == ["bytes=3-6", "bytes=7-9"]
    assert (tmp_path / "address.vob").read_bytes() == b"defghij"
    assert result["sha256"] == hashlib.sha256(b"defghij").hexdigest()
    assert result["full_source_archive_acquired"] is False


@pytest.mark.parametrize("status,headers,body", [
    (200, {}, b"abcdefghijkl"),
    (206, {"content-range": "bytes 0-6/12"}, b"abcdefg"),
    (206, {"content-range": "bytes 3-9/12"}, b"short"),
])
def test_rejects_unhonored_or_incomplete_ranges(tmp_path, status, headers, body):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
            status, headers=headers, content=body))) as client:
        with pytest.raises(ValueError):
            download_range(entry(), tmp_path / "address.vob", client=client)
    assert not (tmp_path / "address.vob").exists()


def test_source_change_never_finalizes_partial(tmp_path):
    def respond(request):
        start, end = map(int, request.headers["range"][6:].split("-"))
        return httpx.Response(206, headers={"content-range": f"bytes {start}-{end}/12",
                                           "etag": str(start)}, content=b"a" * (end - start + 1))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="identity changed"):
            download_range(entry(), tmp_path / "address.vob", client=client, chunk_bytes=4)
    state = json.loads((tmp_path / "address.vob.state.json").read_text())
    assert state["bytes"] == 4
    assert not (tmp_path / "address.vob").exists()


def test_resume_rejects_changed_bytes(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            download_range(entry(), tmp_path / "address.vob", client=client)
        state_path = tmp_path / "address.vob.state.json"
        state = json.loads(state_path.read_text())
        state.update(bytes=1, sha256=hashlib.sha256(b"a").hexdigest())
        state_path.write_text(json.dumps(state))
        (tmp_path / "address.vob.part").write_bytes(b"b")
        with pytest.raises(ValueError, match="Resume bytes"):
            download_range(entry(), tmp_path / "address.vob", client=client, resume=True)


def test_resume_reuses_only_verified_prefix_and_requests_remaining_range(tmp_path):
    requests = []

    def respond(request):
        requests.append(request.headers["range"])
        if len(requests) == 2:
            return httpx.Response(503)
        start, end = map(int, request.headers["range"][6:].split("-"))
        return httpx.Response(206, headers={"content-range": f"bytes {start}-{end}/12"},
                              content=b"abcdefghijkl"[start:end + 1])

    output = tmp_path / "address.vob"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            download_range(entry(), output, client=client, chunk_bytes=4)
        with (tmp_path / "address.vob.part").open("ab") as stream:
            stream.write(b"uncheckpointed interrupted append")
        result = download_range(entry(), output, client=client, resume=True, chunk_bytes=4)
    assert requests == ["bytes=3-6", "bytes=7-9", "bytes=7-9"]
    assert output.read_bytes() == b"defghij"
    assert result["bytes"] == 7


def test_ignored_range_body_is_never_consumed(tmp_path):
    class NeverRead(httpx.SyncByteStream):
        def __iter__(self):
            raise AssertionError("A complete archive body must not be consumed")
            yield b""

    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, stream=NeverRead()))) as client:
        with pytest.raises(ValueError, match="body not consumed"):
            download_range(entry(), tmp_path / "address.vob", client=client)


@pytest.mark.parametrize("decoder_stderr,expected", [
    ("", "passed"),
    ("Error submitting packet to decoder: Invalid data found when processing input\n",
     "completed_with_decoder_warnings"),
])
def test_successful_decoder_exit_preserves_warnings_in_saved_receipt(
        tmp_path, monkeypatch, decoder_stderr, expected):
    from tools.persona_data import archive_range

    original_client = httpx.Client
    transport = httpx.MockTransport(lambda _: httpx.Response(
        206, headers={"content-range": "bytes 3-9/12"}, content=b"defghij"))
    monkeypatch.setattr(archive_range.httpx, "Client",
                        lambda **kwargs: original_client(transport=transport))
    monkeypatch.setattr(archive_range.subprocess, "check_output", lambda _: json.dumps({
        "format": {"duration": "10", "start_time": "3"},
        "streams": [{"codec_type": "video"}],
    }).encode())
    monkeypatch.setattr(archive_range.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stderr=decoder_stderr))

    output = tmp_path / "address.vob"
    archive_range.acquire(entry(), output)
    saved = json.loads((tmp_path / "address.vob.receipt.json").read_text())
    assert saved["validation"]["tail_decode"] == expected
    assert saved["validation"]["tail_decoder_exit_code"] == 0
    assert saved["validation"]["tail_decoder_stderr"] == decoder_stderr
    assert saved["full_source_archive_acquired"] is False
    assert "semantic_completeness_requires_review" in saved["status"]
