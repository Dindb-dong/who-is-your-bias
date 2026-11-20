import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import captions, dataset, demo, experiments, handoff, media
from pipeline.common import PipelineError, read_csv, read_json, read_jsonl, write_json, write_jsonl


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def sample(self):
        result = demo.create(self.root)
        return {k: Path(v) if k != "note" else v for k, v in result.items()}

    def test_full_demo_and_portable_handoff(self):
        r = self.sample()
        self.assertEqual(len(read_jsonl(r["dataset"] / "train.jsonl")), 2)
        self.assertEqual(len(read_jsonl(r["dataset"] / "eval.jsonl")), 1)
        requests = read_jsonl(r["experiment"] / "requests.jsonl")
        self.assertEqual([x["example_count"] for x in requests], [0, 1, 1])
        answer = read_jsonl(r["dataset"] / "eval.jsonl")[0]["answer_verbatim"]
        self.assertTrue(all(answer not in json.dumps(q["messages"]) for q in requests))
        self.assertEqual(read_json(r["handoff"] / "avatar_turn.json")["status"], "awaiting_audio")
        self.assertTrue((r["handoff"] / "reference" / "reference.wav").exists())
        self.assertAlmostEqual(media.audio_info(r["reference"].parent / "reference.wav")["duration"], 5)

    def test_unreviewed_group_cannot_enter_dataset(self):
        self.sample()
        rows = read_csv(dataset.source_dir(self.root, "demo-1") / "review.csv")
        rows[1]["reviewed"], rows[1]["voice_clean"] = False, False
        dataset.save_review(self.root, "demo-1", rows)
        result = dataset.build(self.root)
        self.assertEqual(len(read_jsonl(result / "train.jsonl")), 1)
        self.assertEqual(read_json(result / "build_report.json")["skipped"][0]["reason"], "unreviewed")

    def test_same_recording_split_rejected(self):
        self.sample()
        with self.assertRaisesRegex(PipelineError, "같은 촬영분"):
            dataset.register(self.root, "new", "synthetic-recording-1", split="eval",
                             subtitle_path=self.root / "example_inputs" / "demo-3.srt")

    def test_cross_split_duplicate_answer_rejected(self):
        self.sample()
        one = read_csv(dataset.source_dir(self.root, "demo-1") / "review.csv")
        three = read_csv(dataset.source_dir(self.root, "demo-3") / "review.csv")
        three[1]["text"] = one[1]["text"]
        dataset.save_review(self.root, "demo-3", three)
        with self.assertRaisesRegex(PipelineError, "같은 답변"):
            dataset.build(self.root)

    def test_pending_and_evaluation_audio_reference_rejected(self):
        self.sample()
        with self.assertRaisesRegex(PipelineError, "평가 촬영분"):
            dataset.reference(self.root, "demo-3", "seg-00002")
        with self.assertRaisesRegex(PipelineError, "단독 음성"):
            dataset.reference(self.root, "demo-1", "seg-00001")

    def test_invalid_and_out_of_media_timestamps_rejected(self):
        self.sample()
        path = dataset.source_dir(self.root, "demo-1")
        original = read_csv(path / "review.csv")
        for start, end in [(2, 1), (-1, 2), (float("nan"), 3), (2, 100)]:
            rows = [dict(r) for r in original]
            rows[1].update(start=start, end=end)
            with self.assertRaises(PipelineError):
                dataset.save_review(self.root, "demo-1", rows)

    def test_registration_does_not_overwrite_or_leave_broken_source(self):
        self.sample()
        with self.assertRaises(PipelineError):
            dataset.register(self.root, "demo-1", "other", subtitle_path=self.root / "missing.srt")
        malformed = self.root / "bad.srt"
        malformed.write_text("no timing here")
        with self.assertRaises(PipelineError):
            dataset.register(self.root, "bad", "bad-recording", subtitle_path=malformed)
        self.assertFalse((self.root / "sources" / "bad").exists())

    def test_caption_bom_tags_and_rolling_cues_preserved(self):
        file = self.root / "test.vtt"
        file.write_text("\ufeffWEBVTT\n\n00:01.000 --> 00:02.000 align:start\n<v Guest>Hello &amp; welcome</v>\n\n00:01.900 --> 00:03.000\nHello &amp; welcome again\n")
        rows = captions.parse(file)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "Hello & welcome")
        self.assertIn("<v Guest>", rows[0]["raw_text"])

    def test_unknown_speaker_cannot_be_approved(self):
        self.sample()
        rows = read_csv(dataset.source_dir(self.root, "demo-1") / "review.csv")
        rows[1]["speaker"] = "unknown"
        with self.assertRaisesRegex(PipelineError, "화자"):
            dataset.save_review(self.root, "demo-1", rows)

    def test_style_monologue_not_used_as_qa(self):
        self.sample()
        rows = read_csv(dataset.source_dir(self.root, "demo-1") / "review.csv")
        rows[0]["kind"] = "exclude"
        rows[1]["kind"] = "style"
        dataset.save_review(self.root, "demo-1", rows)
        out = dataset.build(self.root)
        self.assertEqual(len(read_jsonl(out / "train.jsonl")), 1)
        self.assertEqual(len(read_jsonl(out / "style.jsonl")), 1)

    def test_situation_changes_retrieval_without_changing_k(self):
        r = self.sample()
        rows = read_jsonl(r["dataset"] / "train.jsonl")
        selected = experiments.retrieve(rows, rows[0]["question"], "reflection", "C", 1)
        self.assertEqual(selected[0]["source_id"], "demo-2")

    def test_evaluation_evidence_cannot_enter_persona(self):
        r = self.sample()
        card = experiments.persona_card("synthetic-persona", "Example")
        card.update(style_notes=["Makes suggestions"], evidence_ids=["demo-3__q1"])
        write_json(self.root / "persona.json", card)
        with self.assertRaisesRegex(PipelineError, "evidence_ids"):
            experiments.prepare(r["dataset"], self.root / "persona.json")

    def test_llm_adapter_success_and_error_are_persisted(self):
        r = self.sample()
        response = json.dumps({"choices": [{"message": {"content": "Synthetic answer"}}], "usage": {"total_tokens": 12}}).encode()
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://example.invalid/v1", "LLM_MODEL": "test"}), patch("pipeline.experiments.post_json", return_value=response) as call:
            out = experiments.run_llm(r["experiment"])
            self.assertEqual(call.call_count, 3)
            self.assertEqual(read_json(out / "run.json")["completed"], 3)
            self.assertNotIn("condition", read_csv(out / "blind_ratings.csv")[0])
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://example.invalid/v1", "LLM_MODEL": "test"}), patch("pipeline.experiments.post_json", side_effect=PipelineError("HTTP 429")) as call:
            out = experiments.run_llm(r["experiment"])
            self.assertEqual(call.call_count, 1)
            self.assertEqual(read_json(out / "run.json")["completed"], 0)

    def test_tts_uses_configured_voice_not_reference_upload(self):
        r = self.sample()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "not-a-real-key", "ELEVENLABS_VOICE_ID": "voice-demo"}), patch("pipeline.handoff.post_json", return_value=b"x" * 100) as call, patch("pipeline.handoff.attach_audio", return_value="test.wav"):
            handoff.elevenlabs(r["handoff"])
            url, payload, headers = call.call_args.args
            self.assertIn("voice-demo", url)
            self.assertEqual(set(payload), {"text", "model_id"})
            self.assertFalse(read_json(r["handoff"] / "tts_result.json")["reference_uploaded"])

    def test_local_mp4_to_review_and_reference(self):
        self.sample()
        input_audio = self.root / "example_inputs" / "demo-1.wav"
        video = self.root / "video.mp4"
        subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i",
                        "color=c=navy:s=320x180:r=10:d=8", "-i", str(input_audio), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)], check=True, capture_output=True)
        dest = self.root / "video-case"
        dataset.register(dest, "video", "recording", target="synthetic-persona", media_path=video,
                         subtitle_path=self.root / "example_inputs" / "demo-1.srt", synthetic=True)
        rows = read_csv(dataset.source_dir(dest, "video") / "review.csv")
        for i, row in enumerate(rows):
            row.update(speaker="target" if i else "interviewer", turn_id="q1", reviewed=True, voice_clean=bool(i))
        dataset.save_review(dest, "video", rows)
        out = dataset.build(dest)
        ref = dataset.reference(dest, "video", "seg-00002")
        self.assertEqual(len(read_jsonl(out / "train.jsonl")), 1)
        self.assertAlmostEqual(media.audio_info(ref.parent / "reference.wav")["duration"], 5)

    def test_youtube_playlist_is_rejected_before_network(self):
        for url in ["https://www.youtube.com/playlist?list=abc", "https://www.youtube.com/@channel", "http://youtu.be/abcdefghijk"]:
            with self.assertRaises(PipelineError):
                media.download_youtube(url, self.root / "download")

    def test_transcription_creates_pending_review_and_prevents_overwrite(self):
        self.sample()
        root = self.root / "asr-case"
        dataset.register(root, "speech", "speech-recording", media_path=self.root / "example_inputs" / "demo-1.wav")
        segments = [{"segment_id": "seg-00001", "start": 0.0, "end": 2.0, "text": "Test speech"}]
        with patch("pipeline.media.transcribe", return_value=(segments, {"engine": "test"})):
            result = dataset.transcribe_source(root, "speech")
            self.assertEqual(read_csv(result)[0]["speaker"], "unknown")
            self.assertEqual(read_csv(result)[0]["reviewed"], "False")
            with self.assertRaises(PipelineError):
                dataset.transcribe_source(root, "speech")


if __name__ == "__main__":
    unittest.main()
