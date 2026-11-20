"""Package and run the existing full-answer Qwen voice on a private CUDA workspace."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import venv
import wave
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
PORT = 8010
VOICE_REPO = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
REVISION = '5d83992436eae1d760afd27aff78a71d676296fc'
PROJECT = ROOT.parent
EVIDENCE = PROJECT / 'data/local/avatar_runs/elevenlabs-20261119'


def pack(output: Path) -> None:
    studio = (PROJECT.parent if (PROJECT.parent / 'services/voice/engine.py').is_file()
              else PROJECT / 'integrations/who-is-your-bias-gesture')
    reference = PROJECT / 'data/local/persona/jensen-huang/voice-reference'
    review = json.loads((reference / 'reference.json').read_text())
    files = {name: (reference / name).read_bytes() for name in ('reference.wav', 'reference.txt')}
    for name, expected in [('reference.wav', review['sha256']), ('reference.txt', review['transcript_sha256'])]:
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError(f'The existing reference changed: {name}')
    exercise = json.loads((EVIDENCE / 'real-api-result.json').read_text())
    warm = json.loads((EVIDENCE / 'qwen-same-answer-warm-result.json').read_text())
    files['benchmark.json'] = json.dumps({'text': exercise['answer'], 'language': 'English',
                                        'mac_warm_synthesis_ms': warm['server_timing']['synthesis_ms']}).encode()
    files['local_voice.py'] = (ROOT / 'local_voice.py').read_bytes()
    files['gpu_voice_trial.py'] = Path(__file__).read_bytes()
    for name in ('__init__.py', 'app.py', 'engine.py', 'provenance.py', 'requirements.txt'):
        file = f'services/voice/{name}'
        files[file] = (studio / file).read_bytes()
    manifest = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    files['manifest.json'] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as buffer, tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, contents in files.items():
            info = tarfile.TarInfo(name)
            info.mode, info.size = 0o600, len(contents)
            archive.addfile(info, io.BytesIO(contents))


def check() -> None:
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Package hash mismatch: {name}')


def serve() -> None:
    check()
    # Use the CUDA-enabled torch/torchaudio supplied by the workspace image.
    subprocess.run([sys.executable, '-c', 'import torch, torchaudio; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.cuda.get_device_name(0), flush=True)'], check=True)
    environment = ROOT / '.venv'
    if not (environment / 'bin/python').exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = str(environment / 'bin/python')
    pins = subprocess.check_output([sys.executable, '-c', "from importlib.metadata import version; print('torch=='+version('torch')); print('torchaudio=='+version('torchaudio'))"], text=True)
    constraints = ROOT / 'cuda-constraints.txt'
    constraints.write_text(pins)
    env = dict(os.environ, BIAS_VOICE_MODEL=VOICE_REPO, BIAS_VOICE_REVISION=REVISION,
               BIAS_VOICE_CACHE_DIR=str(ROOT / 'model-cache'), BIAS_VOICE_LOCAL_FILES_ONLY='true',
               BIAS_VOICE_DEVICE='cuda:0', BIAS_VOICE_DTYPE='bfloat16', BIAS_VOICE_ATTENTION='sdpa',
               BIAS_VOICE_REFERENCE_WAV=str(ROOT / 'reference.wav'),
               BIAS_VOICE_REFERENCE_TEXT_FILE=str(ROOT / 'reference.txt'),
               BIAS_VOICE_MAX_INPUT_CHARS='3000', BIAS_VOICE_MAX_NEW_TOKENS='1000',
               BIAS_VOICE_MAX_AUDIO_SECONDS='80', BIAS_VOICE_THREADS='4',
               HF_HUB_DISABLE_TELEMETRY='1', ORT_DISABLE_TELEMETRY='1')
    subprocess.run([python, '-m', 'pip', 'install', '-c', str(constraints), '-r', str(ROOT / 'services/voice/requirements.txt')], env=env, check=True)
    download = ('import os; from huggingface_hub import snapshot_download; '
                'snapshot_download(repo_id=os.environ["BIAS_VOICE_MODEL"], revision=os.environ["BIAS_VOICE_REVISION"], cache_dir=os.environ["BIAS_VOICE_CACHE_DIR"])')
    subprocess.run([python, '-c', download], env=env, check=True)
    os.chdir(ROOT)
    os.execve(python, [python, '-m', 'uvicorn', 'local_voice:app', '--host', '127.0.0.1', '--port', str(PORT)], env)


def bench(base_url: str) -> None:
    check()
    fixture = json.loads((ROOT / 'benchmark.json').read_text())
    headers = {'Content-Type': 'application/json'}
    if token := os.getenv('BIAS_VOICE_API_KEY'):
        headers['Authorization'] = f'Bearer {token}'
    report = {'mac_warm_synthesis_ms': fixture['mac_warm_synthesis_ms'], 'runs': []}
    for index in range(3):
        started = perf_counter()
        request = Request(base_url.rstrip('/') + '/v1/audio/speech',
                          data=json.dumps({'model': 'qwen3-tts', 'input': fixture['text'], 'language': fixture['language'],
                                           'voice': 'reference', 'response_format': 'wav'}).encode(), headers=headers)
        with urlopen(request, timeout=180) as response:
            audio = response.read()
        elapsed = round((perf_counter()-started)*1000, 1)
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or not wav.getnframes():
                raise ValueError('Unexpected server WAV format')
            seconds = wav.getnframes() / wav.getframerate()
        with urlopen(base_url.rstrip('/') + '/health', timeout=5) as response:
            status = json.load(response)
        if not status['runtime']['device'].startswith('cuda:'):
            raise ValueError('The voice server did not report running on CUDA')
        report['runs'].append({'request_ms': elapsed, 'audio_seconds': round(seconds, 2),
                               'runtime': status['runtime'], 'generation': status['last_generation']})
        (ROOT / f'gpu-voice-{index+1}.wav').write_bytes(audio)
        (ROOT / 'gpu-voice-result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    package = commands.add_parser('pack')
    package.add_argument('output', type=Path)
    commands.add_parser('check')
    commands.add_parser('serve')
    measure = commands.add_parser('bench')
    measure.add_argument('--base-url', default=f'http://127.0.0.1:{PORT}')
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.output)
    elif args.command == 'check':
        check()
        print('Package hashes verified.')
    elif args.command == 'serve':
        serve()
    else:
        bench(args.base_url)
