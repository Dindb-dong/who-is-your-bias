import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import batch, dataset
from pipeline.common import PipelineError, read_csv


class BatchTests(unittest.TestCase):
    def test_retry_skips_done_items_and_preserves_pending_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [(batch.URL + vid, vid) for vid in ("abcdefghijk", "01234567890")]
            path = batch.new_job(root, entries + [("https://youtu.be/abcdefghijk", "")])

            def captions_for(vid, language):
                if vid == "01234567890":
                    raise PipelineError("no captions")
                return [{"start": 0, "end": 1, "text": "Interviewer: Why?"},
                        {"start": 1, "end": 2, "text": "Jensen: An example."}]

            with patch.object(batch, "aside_segments", side_effect=captions_for), patch(
                    "pipeline.media.download_youtube", side_effect=PipelineError("unavailable")):
                report = batch.run_job(root, path)
            self.assertEqual([item["status"] for item in report["items"]], ["done", "failed"])
            path_for_source = dataset.source_dir(root, "yt-abcdefghijk")
            self.assertEqual(path_for_source.name, "abcdefghijk [yt-abcdefghijk]")
            rows = read_csv(path_for_source / "review.csv")
            self.assertEqual(rows[0]["turn_id"], rows[1]["turn_id"])
            self.assertTrue(all(row["reviewed"] == "False" for row in rows))

            with patch.object(batch, "aside_segments", return_value=[
                {"start": 0, "end": 1, "text": "A different example."}
            ]) as fetch:
                report = batch.run_job(root, path)
            fetch.assert_called_once_with("01234567890", "en")
            self.assertEqual([item["status"] for item in report["items"]], ["done", "done"])

    def test_llm_can_suggest_roles_but_cannot_change_source_or_approve(self):
        rows = [{"segment_id": "seg-00001", "start": 1, "end": 2, "text": "Original",
                 "speaker": "unknown", "reviewed": False, "voice_clean": False}]
        answer = {"rows": [{"segment_id": "seg-00001", "speaker": "target",
                            "text": "Changed", "start": 0, "reviewed": True, "voice_clean": True}]}
        response = {"choices": [{"message": {"content": json.dumps(answer)}}]}
        with patch.dict("os.environ", {"LLM_BASE_URL": "https://example.com", "LLM_MODEL": "fake"}), patch(
                "pipeline.experiments.post_json", return_value=json.dumps(response)):
            batch.llm_labels(rows)
        self.assertEqual(rows[0]["speaker"], "target")
        self.assertEqual((rows[0]["text"], rows[0]["start"]), ("Original", 1))
        self.assertFalse(rows[0]["reviewed"])
        self.assertFalse(rows[0]["voice_clean"])
