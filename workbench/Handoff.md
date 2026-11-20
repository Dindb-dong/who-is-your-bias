# 동욱에게 전달하는 형식

한 번 답하는 단위마다 `data/local/handoff/turn-...` 폴더가 생긴다. **폴더 전체를 전달한다.** 경로는 폴더 안의 상대 경로여서 다른 컴퓨터에서도 연결할 수 있다.

## 파일과 쓰임새

| 파일 | 용도 |
| --- | --- |
| avatar_turn.json | 화면 자막, 발화 텍스트, 음성 경로, 턴 ID, 처리 상태 |
| tts_request.json | 어느 TTS를 사용하든 필요한 답변과 선택적 참조 음성·전사 |
| reference/reference.wav | 검수한 원본 참조 음성. 재생할 새 답변 음성과 다름 |
| reference/reference.json | 정확한 참조 전사, 출처·구간·샘플레이트·해시 |
| speech.wav | 생성한 답변 음성. TTS 실행 또는 외부 생성 음성 연결 후 생김 |
| manifest.json | 파일 목록과 생성 시점, 생성 음성 해시 |

## 영상 담당자가 읽을 파일

```json
{
  "schema_version": "1.0",
  "turn_id": "turn-example",
  "target_id": "jensen-huang",
  "display_text": "화면에 표시할 답변",
  "speech_text": "음성으로 읽을 답변",
  "language": "ko",
  "audio_file": "speech.wav",
  "audio_info": {
    "sample_rate": 24000,
    "channels": 1,
    "sample_width": 2,
    "duration": 4.2
  },
  "face_video": null,
  "gesture_hints": [],
  "status": "ready_for_video",
  "is_ai_simulation": true
}
```

음성이 아직 없으면 `audio_file=null`, `audio_info=null`, `status=awaiting_audio`다. 위 예시의 길이는 설명용이며 실제 파일에서는 측정값을 기록한다. `face_video`와 제스처는 동욱 쪽에서 결정한다. 단어별 정렬과 제스처 시점은 현재 생성하지 않는다.

## 음성 모델별 연결

- **ElevenLabs**: 이미 사용할 수 있는 `voice_id`와 답변 텍스트를 전달한다. 이 프로젝트의 어댑터는 새 음성을 등록하거나 참조 파일을 업로드하지 않는다. 선택한 voice_id가 어떤 목소리인지 따로 확인해야 한다. MP3 응답을 받은 뒤 24 kHz mono PCM WAV로 변환한다. [공식 API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
- **Qwen3-TTS Base**: 팀의 `speech_text`를 `text`, 참조 WAV를 `ref_audio`, 정확한 참조 전사를 `ref_text`로 매핑하는 입력을 준비한다. 통합 시연의 로컬 추론은 `scripts/local_voice.py`에서 같은 입력을 사용한다. [공식 사용 예시](https://github.com/QwenLM/Qwen3-TTS#voice-clone)
- **CosyVoice**: 답변, 참조 WAV, 참조 전사를 보존한다. 팀 카톡 후보는 CosyVoice 2이며, 선택할 실제 버전의 입력 함수와 샘플레이트를 확인한 뒤 연결한다. 최신 저장소의 기능을 2 버전에 그대로 적용한다고 가정하지 않는다. [공식 저장소](https://github.com/FunAudioLLM/CosyVoice)

`tts_request.json`은 팀 공통 형식이다. 파일 자체를 모델 API에 그대로 보내는 것은 아니다. Qwen/CosyVoice에서 생성한 파일은 **생성 음성 연결** 또는 `attach-audio` 명령으로 붙일 수 있다.

## 동욱과 맞출 항목

1. `avatar_turn.json`과 WAV 파일을 받는 방식으로 첫 연결을 할 수 있는가.
2. 현재 임시 규격 24 kHz mono PCM WAV를 지원하는가. 다르면 출력 변환 규격을 바꾼다.
3. 얼굴 영상은 어느 쪽이 준비하고, 어떤 영상 ID를 사용할 것인가.
4. 제스처는 음성을 보고 자동 생성할지, 의미 태그를 받을지. 실제 음성 길이를 얻은 후 타이밍을 맞춘다.
5. 사전 제작 시연부터 할지, 현장 질문의 스트리밍까지 할지. 현재 구현은 파일 단위 배치 처리다.

첫 연동은 짧은 한 턴으로 자막·음성·입모양·턴 전환을 확인한다. 얼굴/제스처 평가와 페르소나 답변 평가를 따로 기록한다.

## Microsoft Build 문답을 연결한 로컬 시연

초기 실험은 `integrations/who-is-your-bias/` 사본에서 진행했다. 공유 브랜치의 실행 방법은 [작업실 README](README.md)를 참고한다. 당시 기준 커밋은 [3038629](https://github.com/Dindb-dong/who-is-your-bias/commit/3038629f3c4aeefff568684a6a7ba10df2b61db8)다.

Finder에서 `아바타시연.command`를 더블클릭하고 [아바타 시연](http://127.0.0.1:8101)을 연다. 첫 추천 질문인 **What did Jensen Huang say about RTX Spark and personal AI at Microsoft Build?**를 선택한다. 응답 아래 **evidence records**에서 인터뷰 링크를 열 수 있다. **Avatar settings → Connected voice → Close settings**를 선택하고 응답 아래 **Listen**을 누르면 Mac 기본 음성으로 읽는다.

시연 시작 시 `scripts/avatar_demo.py`가 작업실 데이터셋에서 검수 근거가 있는 Microsoft Build 문답을 골라 동욱이의 기존 NVIDIA 자료에 더한다. 원본 출처, 시각, 문답 ID를 유지하고 정확한 말뭉치 해시에 연결된 스냅샷을 만든다. 결과는 `data/local/avatar_runs/`의 실행별 폴더에 저장한다. 로컬 말뭉치 생성과 음성 재생은 동욱이의 GitHub 저장소에 변경을 반영하지 않는다.

시연의 답변은 출처에서 가져온 원문 발췌다. 음성은 Mac 기본 음성이고, 아바타는 동욱이 저장소의 시제품을 사용한다. 변환기는 현재 Microsoft Build 파일럿을 지원하며, 검수표를 수정했다면 새 검수 근거를 기록해야 한다.

다른 Mac에서 준비하려면 이 저장소 루트에서 `python3 -m venv .venv`와 `.venv/bin/python -m pip install .`을 실행한다. 저장소 요구 버전을 만족하는 Node.js로 `npm ci`와 `npm run build`를 실행한다. 음성에는 macOS의 Samantha와 Yuna가 필요하다.

## 로컬 페르소나 후속 실행 — 2026-11-13

새 실행은 `로컬페르소나시연.command`, 주소 8105. LLM 8103, voice 8104. `.tools/local-runtime.json`과 `.tools/local-models.json`에 로컬 모델 revision·경로·결과 위치 기록. upstream 수정 없이 scripts/local_persona.py와 run_local_persona.py를 사용. 연구/결과 문서를 참고. 실제 답변은 생성되지만 근거 인용 누락과 일반적인 말투가 남아 품질 완료 아님. 사용자가 요구한 로컬 모델·기존 음성 연결은 실제 시험 성공.
