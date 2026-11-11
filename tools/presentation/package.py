"""Package only a completed speech + MuseTalk run into portable playback media."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .common import check_timing, digest, probe, read_plan, save, wav_info


def package(*, plan_path: Path, speech_dir: Path, render_dir: Path, portrait: Path, output: Path) -> dict:
    from PIL import Image, ImageDraw, ImageFont
    plan = read_plan(plan_path)
    speech = json.loads((speech_dir / "speech.json").read_text())
    render = json.loads((render_dir / "render.json").read_text())
    if speech.get("status") != "completed" or render.get("status") != "completed":
        raise ValueError("Only completed speech and render runs can become a presentation")
    if speech["plan_sha256"] != digest(plan_path):
        raise ValueError("The script changed after speech synthesis")
    audio = speech_dir / "speech.wav"
    video = render_dir / "silent.mp4"
    if render["audio_sha256"] != digest(audio) or speech["audio"]["sha256"] != digest(audio):
        raise ValueError("Speech/render audio fingerprint mismatch")
    if render["silent_video_sha256"] != digest(video) or render["portrait_sha256"] != digest(portrait):
        raise ValueError("Render source or video fingerprint mismatch")
    if render["backend"] != "musetalk-1.5-fixed-crop":
        raise ValueError("This package requires the real MuseTalk backend")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory to preserve prior presentations")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("speech.wav", "captions.srt", "captions.vtt", "speech.json"):
        shutil.copy2(speech_dir / name, output / name)
    for segment in speech["segments"]:
        source = speech_dir / segment["path"]
        if source.parent != speech_dir or digest(source) != segment["sha256"]:
            raise ValueError("Segment audio is missing or changed after synthesis")
        shutil.copy2(source, output / source.name)
    shutil.copy2(render_dir / "render.json", output / "render.json")
    shutil.copy2(plan_path, output / "plan.json")
    shutil.copy2(portrait, output / "portrait.png")
    canvas = Image.new("RGB", (1280, 720), "#101827")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font, body_font, small_font = [ImageFont.truetype(font_path, size) for size in (43, 27, 19)]
    draw.text((550, 100), "WHO IS YOUR BIAS?", fill="#f5f5f4", font=title_font)
    draw.text((550, 165), "YAICON · PREPARED RESEARCH DEMO", fill="#7dd3fc", font=small_font)
    for index, (label, content) in enumerate([
        ("A", "Profile only"), ("B", "Actual interview examples"), ("C", "Topic + situation retrieval")]):
        y = 246 + index * 81
        draw.rounded_rectangle((550, y, 595, y + 44), radius=10, fill="#1d4ed8")
        draw.text((562, y + 5), label, fill="white", font=body_font)
        draw.text((615, y + 5), content, fill="#dbeafe", font=body_font)
    draw.text((550, 530), "Independent human review is pending.", fill="#94a3b8", font=small_font)
    draw.rectangle((0, 0, 1280, 53), fill="#0b1220")
    draw.text((25, 15), plan["disclosure"], fill="#fbbf24", font=small_font)
    canvas.save(output / "slide.png")
    # Paths are arguments, never interpolated into a shell. Subtitle filename
    # is a fixed safe basename under cwd, avoiding FFmpeg filter-path escaping.
    command = ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "25", "-i", "slide.png",
        "-i", str(video.resolve()), "-i", "speech.wav", "-filter_complex",
        "[1:v]scale=480:600[face];[0:v][face]overlay=30:60:shortest=1,subtitles=captions.srt:force_style='FontName=DejaVu Sans,FontSize=18,MarginV=12,Outline=2'[out]",
        "-map", "[out]", "-map", "2:a:0", "-c:v", "libx264", "-threads", "2", "-crf", "19", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        "-shortest", "presentation.mp4"]
    subprocess.run(command, cwd=output, check=True, capture_output=True, timeout=180)
    media = probe(output / "presentation.mp4")
    timing = check_timing(media, wav_info(audio)["seconds"], render["fps"])
    # Also decode every video/audio packet; a readable header alone is insufficient.
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output / "presentation.mp4"), "-f", "null", "-"],
        check=True, capture_output=True, timeout=60)
    record = {"schema_version": "presentation-package.v1", "status": "completed", "backend": render["backend"],
        "disclosure": plan["disclosure"], "script_kind": plan.get("script_kind"),
        "speech_model": {"model": speech["model"], "revision": speech["revision"]},
        "video_models": render["models"], "timing": timing, "full_decode_passed": True,
        "generated_frames": render["generated_frames"], "human_listening_review": "unrated",
        "human_visual_sync_review": "unrated", "likeness_review": "unrated",
        "limitations": ["Fixed artistic portrait; no learned personal head/body motion.",
            "Geometric face blending differs from the upstream face-parsing pipeline.",
            "Container integrity does not establish perceptual lip-sync, voice quality, or likeness.",
            "This prepared narration explains the experiment; it is not an attributed source quotation or live LLM answer."]}
    record["artifacts"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}
    save(output / "manifest.json", record)
    (output / "index.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Who is your bias? — Prepared presentation</title><style>body{margin:0;background:#101827;color:#e2e8f0;font:16px system-ui;padding:24px}main{max-width:1040px;margin:auto}video{width:100%;border-radius:12px;background:#000}h1{font-size:26px}.notice{color:#fbbf24}a{color:#7dd3fc}p{line-height:1.6}</style><main>
<h1>Who is your bias?</h1><p class="notice">AI research simulation — not the real person</p>
<video controls preload="metadata" poster="slide.png"><source src="presentation.mp4" type="video/mp4"><track kind="captions" src="captions.vtt" srclang="en" label="English"></video>
<p>Prepared Qwen3-TTS speech and actual MuseTalk 1.5 neural lip-sync on an artistic portrait. The captions are also burned into the video. Human voice and visual synchronization ratings remain ungraded.</p>
<p><a href="speech.wav">Generated speech</a> · <a href="captions.vtt">Captions</a> · <a href="manifest.json">Measured run metadata</a> · <a href="plan.json">Prepared script and sources</a></p></main></html>""", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package(plan_path=args.plan, speech_dir=args.speech_dir, render_dir=args.render_dir,
        portrait=args.portrait, output=args.output)["status"])


if __name__ == "__main__":
    main()
