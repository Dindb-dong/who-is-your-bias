import fcntl
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.adaptation.supervise import run_supervised
from tools.adaptation.checkpoints import atomic_json


def test_child_exit_is_recorded_and_lock_is_released(tmp_path):
    state = run_supervised([sys.executable, "-c", "print('actual child')"], tmp_path / "state", "one", tmp_path / "one.log", timeout=3, interval=.02)
    assert state["invocation_state"] == "exited"
    assert state["child_returncode"] == 0 and state["child_stopped"]
    assert (tmp_path / "one.log").read_text().strip() == "actual child"
    assert json.loads((tmp_path / "state/one.json").read_text())["child_stopped"]
    second = run_supervised([sys.executable, "-c", "pass"], tmp_path / "state", "two", tmp_path / "two.log", timeout=3, interval=.02)
    assert second["child_returncode"] == 0


def test_existing_lock_prevents_any_second_child(tmp_path):
    directory = tmp_path / "state"
    directory.mkdir()
    with (directory / "active.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="shared lock"):
            run_supervised([sys.executable, "-c", "raise Exception('must not launch')"], directory, "blocked", tmp_path / "blocked.log")
    assert not (tmp_path / "blocked.log").exists()


def test_file_stop_request_stops_owned_child_without_tool_handle(tmp_path):
    directory = tmp_path / "state"
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_supervised, [sys.executable, "-c", "import time; time.sleep(10)"], directory, "stoptest", tmp_path / "stop.log", timeout=5, interval=.02)
        deadline = time.monotonic() + 3
        while not (directory / "stoptest.json").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (directory / "stoptest.json").exists()
        atomic_json(directory / "stop-stoptest.json", {"name": "stoptest", "reason": "test owned-process shutdown"})
        state = future.result(timeout=3)
    assert state["invocation_state"] == "stopped_by_request"
    assert state["child_stopped"] and state["child_returncode"] is not None


def test_deadline_stops_child_and_preserves_log(tmp_path):
    state = run_supervised([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "state", "deadline", tmp_path / "deadline.log", timeout=.1, interval=.02)
    assert state["invocation_state"] == "timed_out" and state["child_stopped"]
    assert (tmp_path / "deadline.log").exists()


def test_existing_log_cannot_be_overwritten(tmp_path):
    log = tmp_path / "saved.log"
    log.write_text("saved evidence")
    with pytest.raises(ValueError, match="new name/log"):
        run_supervised([sys.executable, "-c", "pass"], tmp_path / "state", "saved", log)
    assert log.read_text() == "saved evidence"
