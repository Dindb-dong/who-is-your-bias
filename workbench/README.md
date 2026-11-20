# YAICON 데이터 작업실

Who’s your bias?의 영상·자막 수집, 문답 검수, RAG 실험, 음성 인계 코드를 모아둔 폴더다. 아바타 렌더링과 실시간 대화 서버는 이 저장소 루트에서 실행한다.

## 데이터 작업실 실행

Mac에서 이 폴더의 `시작.command`를 실행하고 [작업실](http://127.0.0.1:8501)을 연다. 가상 예제는 화면에서 만들 수 있다. 터미널에서는 이 폴더 기준으로 다음 명령을 실행한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pipeline demo
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

자동 전사가 필요하면 같은 환경에 `requirements-asr.txt`를 설치한다. [사용 가이드](사용가이드.md)는 영상 입력부터 검수와 A/B/C 실험까지, [데이터 형식](Data.md)과 [인계 형식](Handoff.md)은 내보내는 파일을 설명한다.

## 아바타와 연결

기본 아바타와 공개 근거를 확인하려면 [저장소 루트의 실행 가이드](../README.md)를 따른다. 팀 자료를 쓰는 Mac 시연은 루트에 Python 환경과 웹 빌드를 준비하고 `workbench/로컬페르소나시연.command`를 실행한다.

- 작업실에서 검수 데이터셋을 먼저 만든다. 팀에서 관리하는 원문 자료, 검수된 인물 카드와 참조 음성은 해당 작업의 로컬 입력으로 준비한다. PR #19 원문 자료를 병합하려면 `data/local/integration/pr19-corpus.json`이 복원한 자료의 `relative_path`를 가리켜야 한다.
- 기존 Mac 시연은 `.tools/local-models.json`에 `llm`과 `voice`의 `repo`, `revision`, `path`를 기록하고 각각 `.tools/venv-llm`, `.tools/venv-voice`에 실행 환경을 둔다. Apple Silicon에서 사용한 패키지 버전은 `requirements-local-*.lock.txt`에 있다. Whisper와 아바타/동작 자료의 준비 방식은 [루트 시연 가이드](../docs/jensen-studio-quickstart.md)를 참고한다.
- 외부 LLM·TTS를 선택하면 `.env.live.example`을 `.env.live`로 복사해 개인 설정을 채운다. 수집·실험 화면의 API 설정은 `.env.example`을 사용한다.

## 음성 벤치마크

- [Kaggle Qwen](benchmarks/kaggle-qwen.ipynb)은 `scripts/gpu_voice_trial.py`로 만든 로컬 `gpu-voice-trial.tar.gz`를 입력으로 받는다.
- [Kaggle F5‑TTS](benchmarks/kaggle-f5.ipynb)는 `reference.wav`, 대응 전사 `reference.txt`, `text`와 `language`를 가진 `benchmark.json`이 필요하다. 각 노트북에서 `input_path`를 자신의 입력 폴더로 바꾼다.
- Mac에서는 `.tools/venv-voice-mlx` 같은 별도 환경에 `mlx-audio==0.6.6`을 설치하고 `python scripts/benchmark_mlx_voice.py --help`로 입력을 확인한다. 이미 내려받은 Qwen3‑TTS Base 경로와 동일한 참조·답변으로 합성 시간을 비교할 수 있다.

개인 API 키와 실제 자료는 Git에서 제외된다. 연구용 참조 자료, 모델 가중치, 녹음, 생성 WAV 및 출력물이 필요하면 팀이 사용하는 비공개 전달 경로에서 개별로 준비한다.
