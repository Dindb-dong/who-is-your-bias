from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote

from . import media
from .common import PipelineError, digest, identifier, now, read_json, run_id, write_json
from .experiments import post_json


def package(root, text, language="en", target="jensen-huang", reference_path=None, display_text=None):
    if not text.strip():
        raise PipelineError("말할 답변을 입력하세요.")
    identifier(target)
    ref = None
    if reference_path:
        reference_path = Path(reference_path)
        ref = read_json(reference_path)
        if ref.get("target_id") != target or ref.get("split") != "train" or ref.get("reviewed") is not True or not ref.get("voice_clean"):
            raise PipelineError("인계 대상과 참조 음성의 인물·검수·분리 정보를 확인하세요.")
        source_audio = reference_path.parent / ref["reference_audio"]
        if not source_audio.is_file() or digest(source_audio) != ref["audio_sha256"]:
            raise PipelineError("참조 음성이 없거나 검수 이후 변경되었습니다.")
    turn_id = run_id("turn")
    output = Path(root) / "handoff" / turn_id
    output.mkdir(parents=True)
    reference_info = None
    if ref:
        (output / "reference").mkdir()
        shutil.copy2(source_audio, output / "reference" / "reference.wav")
        write_json(output / "reference" / "reference.json", ref)
        reference_info = {"audio_file": "reference/reference.wav", "text": ref["reference_text"],
                          "language": ref["language"], "source_id": ref["source_id"], "segment_id": ref["segment_id"]}
    write_json(output / "tts_request.json", {"schema_version": "1.0", "turn_id": turn_id, "target_id": target,
               "speech_text": text.strip(), "language": language, "reference": reference_info,
               "provider": "undecided", "note": "팀 공통 입력 형식입니다. 각 모델의 API에는 어댑터로 매핑합니다."})
    write_json(output / "avatar_turn.json", {"schema_version": "1.0", "turn_id": turn_id, "target_id": target,
               "display_text": display_text if display_text is not None else text.strip(), "speech_text": text.strip(),
               "language": language, "audio_file": None, "audio_info": None, "face_video": None,
               "gesture_hints": [], "status": "awaiting_audio", "is_ai_simulation": True})
    write_json(output / "manifest.json", {"created_at": now(), "turn_id": turn_id,
               "files": ["tts_request.json", "avatar_turn.json"] + (["reference/reference.wav", "reference/reference.json"] if ref else [])})
    return output


def attach_audio(package_path, audio_path):
    package_path = Path(package_path)
    turn = read_json(package_path / "avatar_turn.json")
    if turn.get("audio_file"):
        raise PipelineError("이미 음성이 연결되어 있습니다. 비교하려면 새 인계 묶음을 만드세요.")
    info = media.to_wav(audio_path, package_path / "speech.wav", sample_rate=24000)
    turn.update(audio_file="speech.wav", audio_info=info, status="ready_for_video")
    write_json(package_path / "avatar_turn.json", turn)
    manifest = read_json(package_path / "manifest.json")
    manifest["files"].append("speech.wav")
    manifest["speech_sha256"] = digest(package_path / "speech.wav")
    write_json(package_path / "manifest.json", manifest)
    return package_path / "speech.wav"


def elevenlabs(package_path):
    package_path = Path(package_path)
    turn = read_json(package_path / "avatar_turn.json")
    if turn.get("audio_file") or (package_path / "provider_output.mp3").exists():
        raise PipelineError("생성된 음성이 있습니다. 반복 과금을 막기 위해 새 묶음에서 실행하세요.")
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    model = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9_-]+", voice):
        raise PipelineError(".env에 ELEVENLABS_API_KEY와 기존 ELEVENLABS_VOICE_ID를 설정하세요.")
    media.ffmpeg()  # Fail before a paid API call if decoding is unavailable.
    audio = post_json(f"https://api.elevenlabs.io/v1/text-to-speech/{quote(voice)}?output_format=mp3_44100_128",
                      {"text": turn["speech_text"], "model_id": model}, {"xi-api-key": key}, timeout=180)
    if not audio or len(audio) < 64:
        raise PipelineError("음성 API가 비어 있는 파일을 반환했습니다.")
    (package_path / "provider_output.mp3").write_bytes(audio)
    write_json(package_path / "tts_result.json", {"provider": "elevenlabs", "voice_id": voice,
               "model_id": model, "created_at": now(), "reference_uploaded": False})
    return attach_audio(package_path, package_path / "provider_output.mp3")
