from __future__ import annotations
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
body = {
    "model": "Qwen3-4B-Q4_K_M", "messages": [
        {"role": "user", "content": "Explain retrieval augmented generation in one sentence."}],
    "max_tokens": 96, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "min_p": 0.0, "presence_penalty": 1.5, "seed": 20261120,
    "stream": True,
}
start = time.perf_counter()
chunks = []; first = None
request = urllib.request.Request("http://127.0.0.1:8129/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    for line in response:
        text = line.decode().strip()
        if not text.startswith("data: ") or text == "data: [DONE]":
            continue
        chunk = json.loads(text[6:]); chunks.append(chunk)
        if first is None and any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
            first = time.perf_counter() - start
elapsed = time.perf_counter() - start
answer = "".join(c.get("delta", {}).get("content", "") for ch in chunks for c in ch.get("choices", []))
record = {"request": body, "response_text": answer, "chunks": chunks,
          "first_token_seconds": first, "elapsed_seconds": elapsed,
          "purpose": "Neutral startup smoke only; not benchmark or persona evaluation"}
(ROOT / "startup-smoke.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k not in {"chunks", "request"}}, indent=2))
