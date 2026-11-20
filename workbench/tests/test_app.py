import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import demo
from pipeline.common import read_json, read_jsonl

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "웹 화면 의존성 미설치")
class AppTests(unittest.TestCase):
    def test_review_build_experiment_handoff_via_ui(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"YAICON_DATA_DIR": tmp}):
            root = Path(tmp) / "demo"
            demo.create(root)
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            app.sidebar.radio[1].set_value("1. 자료 입력").run()
            self.assertFalse(app.exception)
            app.sidebar.radio[0].set_value("가상 예제").run()
            app.sidebar.radio[1].set_value("2. 전사·검수").run()
            next(b for b in app.button if b.label == "검수표 저장").click().run()
            self.assertFalse(app.error)
            app.sidebar.radio[1].set_value("3. 데이터셋").run()
            next(b for b in app.button if b.label == "데이터셋 만들기").click().run()
            self.assertFalse(app.error)
            app.sidebar.radio[1].set_value("4. 답변 실험").run()
            next(t for t in app.text_area if t.label == "실험 질문").set_value("How can I learn from failure?")
            next(t for t in app.text_input if t.label == "질문의 상황").set_value("advice")
            next(b for b in app.button if b.label == "A/B/C 실험 입력 만들기").click().run()
            self.assertFalse(app.error)
            self.assertFalse(app.exception)
            experiments = sorted((root / "experiments").glob("abc-*"))
            self.assertEqual(len(experiments), 2)
            generated = [read_jsonl(p / "requests.jsonl") for p in experiments]
            self.assertTrue(any(r[0]["question"] == "How can I learn from failure?" for r in generated))
            app.sidebar.radio[1].set_value("5. 음성·인계").run()
            next(t for t in app.text_area if t.label.startswith("말할 답변")).set_value("This is a UI test.")
            next(b for b in app.button if b.label == "동욱에게 넘길 묶음 만들기").click().run()
            self.assertFalse(app.error)
            self.assertFalse(app.exception)
            packages = list((root / "handoff").glob("*/avatar_turn.json"))
            self.assertTrue(any(read_json(p)["speech_text"] == "This is a UI test." for p in packages))


if __name__ == "__main__":
    unittest.main()
