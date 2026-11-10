import json
from pathlib import Path

import httpx
import pytest

from tools.research.collect import collect_source, validate_public_url
from tools.research.inventory import canonical_url, inventory
from tools.research.__main__ import main


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_collection_preserves_unreviewed_provenance_and_content_hash(tmp_path):
    source = {"id": "interview-1", "url": "https://publisher.example/interview"}
    with client(lambda request: httpx.Response(200, text="A publisher page", headers={"content-type": "text/html"})) as transport:
        result = collect_source(source, tmp_path, client=transport)
    assert result["indexed"] is False
    assert result["target_speaker_verified"] is False
    assert result["review_status"] == "unreviewed"
    assert (tmp_path / source["id"] / result["artifact"]).read_text() == "A publisher page"
    manifest = next((tmp_path / source["id"]).glob("*.manifest.json"))
    assert json.loads(manifest.read_text())["sha256"] == result["sha256"]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://publisher.example/", "https://user:password@publisher.example/", "https://127.0.0.1/", "https://169.254.169.254/", "https://localhost/", "https://host.internal/", "https://publisher.example:8000/"])
def test_nonpublic_or_credential_urls_rejected(url):
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_redirect_to_localhost_is_not_followed(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://127.0.0.1/private"})
    with client(handler) as transport, pytest.raises(ValueError):
        collect_source({"id": "source", "url": "https://publisher.example/"}, tmp_path, client=transport)
    assert len(calls) == 1


def test_recording_and_oversize_downloads_do_not_create_artifacts(tmp_path):
    source = {"id": "source", "url": "https://publisher.example/"}
    for mime, maximum in [("audio/mpeg", 100), ("text/html", 2)]:
        with client(lambda request: httpx.Response(200, content=b"123456", headers={"content-type": mime})) as transport, pytest.raises(ValueError):
            collect_source(source, tmp_path, client=transport, maximum_bytes=maximum)
    assert not list(tmp_path.iterdir())


def test_existing_symlink_source_directory_is_rejected(tmp_path):
    output = tmp_path / "output"; output.mkdir()
    other = tmp_path / "other"; other.mkdir()
    (output / "source").symlink_to(other, target_is_directory=True)
    with client(lambda request: httpx.Response(200, text="data", headers={"content-type": "text/plain"})) as transport, pytest.raises(ValueError):
        collect_source({"id": "source", "url": "https://publisher.example/"}, output, client=transport)
    assert not list(other.iterdir())


def test_canonical_urls_remove_tracking_without_collapsing_document_ids():
    assert canonical_url("https://publisher.example/talk/?utm_source=x#part") == "https://publisher.example/talk"
    assert canonical_url("https://publisher.example/watch?v=one") != canonical_url("https://publisher.example/watch?v=two")


def test_inventory_detects_event_family_split_leak(corpus):
    records = list(corpus.sources.values())
    first = next(source for source in records if source.split == "train")
    heldout = next(source for source in records if source.split == "test")
    corpus.sources[first.id] = first.model_copy(update={"source_event_id": "same-interview"})
    corpus.sources[heldout.id] = heldout.model_copy(update={"source_event_id": "same-interview"})
    report = inventory(corpus)
    assert report["source_event_split_leaks"][0]["source_event_id"] == "same-interview"
    assert len(report["source_event_split_leaks"][0]["sources"]) == 2


def test_inventory_preserves_all_documents_and_typed_media_links(corpus):
    source = corpus.sources["rocket-interview"]
    urls = ["https://example.org/part-one", "https://example.org/part-two", "https://example.org/part-three"]
    corpus.sources[source.id] = source.model_copy(update={
        "transcript_url": urls[0], "transcript_urls": urls,
        "audio_episode_url": "https://example.org/episode",
        "media_links": [{"url": "https://example.org/video", "kind": "video_page"}],
    })
    record = next(item for item in inventory(corpus)["sources"] if item["id"] == source.id)
    assert record["acquisition_urls"] == urls
    assert record["audio_url"] is None
    assert record["audio_episode_url"].endswith("/episode")
    assert record["media_links"][0]["kind"] == "video_page"


def test_identical_documents_keep_each_url_provenance(tmp_path):
    with client(lambda request: httpx.Response(200, text="Same content", headers={"content-type": "text/plain"})) as transport:
        for part in ("one", "two"):
            collect_source({"id": "source", "url": "https://example.org/" + part}, tmp_path, client=transport)
    assert len(list((tmp_path / "source").glob("*.txt"))) == 1
    records = [json.loads(path.read_text()) for path in (tmp_path / "source").glob("*.manifest.json")]
    assert {item["requested_url"] for item in records} == {"https://example.org/one", "https://example.org/two"}


def test_inventory_cli_emits_split_violation_report(corpus, tmp_path):
    source = corpus.sources["rocket-interview"]
    corpus.sources["heldout"] = corpus.sources["heldout"].model_copy(update={"url": source.url})
    (tmp_path / "profile.json").write_text(corpus.persona.model_dump_json())
    (tmp_path / "sources.json").write_text(json.dumps([source.model_dump(mode="json") for source in corpus.sources.values()]))
    (tmp_path / "evidence.jsonl").write_text("\n".join(record.model_dump_json() for record in corpus.evidence.values()) + "\n")
    report_path = tmp_path / "inventory.json"
    assert main(["--corpus", str(tmp_path), "inventory", "--output", str(report_path)]) == 1
    report = json.loads(report_path.read_text())
    assert report["url_split_leaks"][0]["sources"] == ["heldout", "rocket-interview"]
