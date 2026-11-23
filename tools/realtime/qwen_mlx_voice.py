"""Local Qwen speech experiment with one continuous +1 semitone filter."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
from threading import Thread, Event

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


SAMPLE_RATE = 24_000
MAX_FRAMES = SAMPLE_RATE * 80


class QwenMlxStream:
    def __init__(self, model: Path, reference: Path, ffmpeg: str):
        self.ffmpeg = ffmpeg
        paths = reference / "reference.wav", reference / "reference.txt"
        proof = json.loads((reference / "reference.json").read_text())
        for path, kind in zip(paths, ("sha256", "transcript_sha256"), strict=True):
            if hashlib.sha256(path.read_bytes()).hexdigest() != proof[kind]:
                raise ValueError("The Jensen voice reference has changed.")
        self.model = load_model(str(model.resolve(strict=True)))
        if self.model.sample_rate != SAMPLE_RATE:
            raise ValueError("Unexpected model sample rate.")
        self.reference = load_audio(str(paths[0]), sample_rate=SAMPLE_RATE)
        self.transcript = paths[1].read_text().strip()
        mx.eval(self.reference)

    def read(self, text: str, on_pcm: Callable[[bytes], None]):
        # Keep pitch and tempo state across the entire utterance.
        rate = round(SAMPLE_RATE * 2 ** (1 / 12))
        filters = f"asetrate={rate},aresample={SAMPLE_RATE},atempo={SAMPLE_RATE / rate:.10f}"
        command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-probesize", "32", "-analyzeduration", "0",
                   "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0", "-af", filters,
                   "-f", "s16le", "-flush_packets", "1", "pipe:1"]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as pitch:
            stopped = Event()
            errors: list[Exception] = []

            def deliver():
                carry = b""
                try:
                    while chunk := pitch.stdout.read1(4096):
                        chunk = carry + chunk
                        whole = len(chunk) // 2 * 2
                        if whole:
                            on_pcm(chunk[:whole])
                        carry = chunk[whole:]
                    if carry:
                        raise ValueError("Incomplete PCM frame.")
                except Exception as error:
                    errors.append(error)
                    stopped.set()
                    pitch.terminate()

            drain = Thread(target=deliver)
            drain.start()
            frames = 0
            try:
                mx.random.seed(42)
                for part in self.model.generate(" ".join(text.split()), ref_audio=self.reference, ref_text=self.transcript,
                        lang_code="English", temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05,
                        max_tokens=1000, stream=True, streaming_interval=0.48, verbose=False):
                    if stopped.is_set():
                        break
                    wav = np.asarray(part.audio.astype(mx.float32)).reshape(-1)
                    frames += wav.size
                    if frames > MAX_FRAMES or not np.isfinite(wav).all():
                        raise ValueError("Invalid generated PCM.")
                    pitch.stdin.write(np.rint(np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
                    pitch.stdin.flush()
            finally:
                pitch.stdin.close()
                drain.join()
            if errors:
                raise errors[0]
            if not frames or pitch.wait() != 0:
                raise ValueError("Qwen returned no usable audio.")


def serve(engine: QwenMlxStream):
    class SpeechHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            self.close_connection = True
            if self.path != "/speech":
                self.send_error(404)
                return
            self.connection.settimeout(15)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 24000 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Invalid request")
                text = json.loads(self.rfile.read(size))["text"]
                if not isinstance(text, str) or not 0 < len(text.strip()) <= 3000:
                    raise ValueError("Invalid text")
            except (ValueError, KeyError, TypeError, OSError):
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()

            def send_pcm(pcm):
                self.wfile.write(f"{len(pcm):x}\r\n".encode() + pcm + b"\r\n")

            try:
                engine.read(text, send_pcm)
                self.wfile.write(b"0\r\n\r\n")
            except (OSError, ValueError, RuntimeError) as error:
                self.log_error("Audio stream stopped: %s", type(error).__name__)

    engine.read("Today we will talk about accelerated computing.", lambda _: None)
    print("Qwen MLX ready at 127.0.0.1:8108", flush=True)
    with HTTPServer(("127.0.0.1", 8108), SpeechHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen3-TTS Base snapshot")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory with reviewed reference.wav, reference.txt and reference.json")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg binary")
    args = parser.parse_args()
    serve(QwenMlxStream(args.model, args.reference_dir, args.ffmpeg))
