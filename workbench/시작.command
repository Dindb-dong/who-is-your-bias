#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import streamlit, imageio_ffmpeg, yt_dlp' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m pipeline --root "${YAICON_DATA_DIR:-data}/local" organize-sources
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 2048 --browser.gatherUsageStats false
