#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
studio="$PWD/integrations/who-is-your-bias"
if [[ -f "$PWD/integrations/who-is-your-bias-pr19/apps/web/dist/index.html" ]]; then
    studio="$PWD/integrations/who-is-your-bias-pr19"
fi
if [[ -f "$PWD/integrations/who-is-your-bias-gesture/apps/web/src/avatar/gaussian/trackedMouth.ts" && -d "$PWD/integrations/who-is-your-bias-gesture/node_modules" ]]; then
    studio="$PWD/integrations/who-is-your-bias-gesture"
fi
python="$PWD/integrations/who-is-your-bias/.venv/bin/python"
if [[ -f ../src/bias/api.py ]]; then
    studio="$(cd .. && pwd -P)"
    python="$studio/.venv/bin/python"
fi
if [[ ! -x "$python" ]]; then
    print -u2 '저장소 루트의 Python 환경이 필요합니다. workbench/README.md를 참고하세요.'
    exit 1
fi
export PYTHONPATH="$PWD:$studio/src:$studio"
if [[ -z "${FFMPEG_BINARY:-}" && -x "$PWD/.venv/bin/python" ]]; then
    export FFMPEG_BINARY="$("$PWD/.venv/bin/python" -c 'from pipeline.media import ffmpeg; print(ffmpeg())')"
fi
exec "$python" -m scripts.run_local_persona "$@"
