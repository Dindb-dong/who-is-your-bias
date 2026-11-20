#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
repo="$PWD/integrations/who-is-your-bias"
if [[ -f ../src/bias/api.py ]]; then
  repo="$(cd .. && pwd -P)"
fi
python="$repo/.venv/bin/python"
if [[ ! -x "$python" || ! -f "$repo/apps/web/dist/index.html" ]]; then
  print -u2 '동욱이 저장소의 Python 환경과 웹 빌드가 필요합니다. Handoff.md를 참고하세요.'
  exit 1
fi
for port in 8101 8102; do
  if /usr/sbin/lsof -t -iTCP:$port -sTCP:LISTEN >/dev/null; then
    print -u2 "포트 $port 사용 중. 이미 실행했다면 http://127.0.0.1:8101 을 여세요."
    exit 1
  fi
done
export PYTHONPATH="$PWD:$repo/src:$repo"
output=$("$python" -m scripts.avatar_demo)
"$python" -m uvicorn scripts.avatar_demo:create_tts --factory --host 127.0.0.1 --port 8102 &
voice_pid=$!
trap 'kill "$voice_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM
BIAS_CORPUS_DIR="$output/corpus" BIAS_PERSONA_SNAPSHOT="$output/snapshot.json" \
BIAS_WEB_DIR="$repo/apps/web/dist" \
BIAS_LLM_BASE_URL="" BIAS_LLM_MODEL="" \
BIAS_TTS_BASE_URL="http://127.0.0.1:8102/v1" BIAS_TTS_MODEL="macos-say" BIAS_TTS_PERSONA_ID="generic" \
BIAS_TTS_VOICE="default" BIAS_TTS_LANGUAGE_HINT="true" \
"$python" -m bias.cli serve --port 8101
