import hashlib
import json

import pytest

from tools.demo_latency.publish import public_turn, publish


def test_publication_omits_source_passage_and_preserves_generated_answer():
    turn = {"text": "Generated answer.", "citations": [{"evidence_id": "source-1", "url": "https://example.com/source", "text": "Original publisher passage."}]}
    public = public_turn(turn)
    assert public["text"] == turn["text"]
    assert "text" not in public["citations"][0]
    assert public["citations"][0]["text_sha256"] == hashlib.sha256(b"Original publisher passage.").hexdigest()
    assert turn["citations"][0]["text"] == "Original publisher passage."


def test_failed_report_cannot_be_packaged_as_success(tmp_path):
    path = tmp_path / "failed.json"
    path.write_text(json.dumps({"schema_version": "demo-http-latency.v1", "completed": False}))
    with pytest.raises(ValueError, match="completed actual"):
        publish(path, [], tmp_path / "public")
    assert not (tmp_path / "public").exists()
