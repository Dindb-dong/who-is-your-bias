"""Concrete runtime configuration with declared corpus and modality identities."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus
from bias.motion import MotionLibrary
from bias.snapshots import PersonaSnapshot, corpus_digest


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def local(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Runtime paths must remain inside the selected repository/artifact root')
    return path


def prepare(args) -> tuple[Settings, Corpus, dict, dict]:
    if not args.allow_candidates:
        raise ValueError('Use --allow-candidates to select this optional research configuration')
    repo = args.repository.resolve(); artifacts = args.artifacts.resolve()
    config_path = args.config or repo/'reports/jensen-runtime-v1/configuration.json'
    config = json.loads(config_path.read_text())
    if config.get('schema_version') != 'jensen-runtime-configuration.v1' or config.get('persona_id') != 'jensen-huang':
        raise ValueError('Expected the versioned Jensen runtime configuration')
    if config['text']['mode'] != 'evidence-preview' or config['text']['language_model_configured']:
        raise ValueError('This launcher does not implicitly enable language-model generation')
    corpus_path = local(artifacts, config['corpus']['artifact_path']); corpus = Corpus.load(corpus_path)
    if corpus.persona.id != config['persona_id'] or corpus_digest(corpus) != config['corpus']['corpus_sha256']:
        raise ValueError('The corpus differs from frozen original-dialogue-v4')
    snap = config['snapshots']['voice' if args.with_voice else 'preview']; snapshot_path = local(repo, snap['path'])
    if digest(snapshot_path) != snap['sha256']: raise ValueError('Runtime snapshot bytes changed')
    snapshot = PersonaSnapshot.load(snapshot_path, corpus)
    catalog_path = local(repo, config['motion']['catalog_path'])
    if digest(catalog_path) != config['motion']['catalog_sha256']: raise ValueError('Motion catalog bytes changed')
    asset_root = local(repo, config['motion']['asset_root']); motion = MotionLibrary.load(catalog_path, asset_root)
    if motion.catalog.persona_id != corpus.persona.id: raise ValueError('Motion belongs to a different persona')
    avatar_path = local(repo, config['avatar']['repository_path'])
    if digest(avatar_path) != config['avatar']['sha256']: raise ValueError('Optional avatar bytes changed')
    web = args.web_root.resolve() if args.web_root else repo/'apps/web/dist'
    web_avatar = local(web, config['avatar']['public_path'].lstrip('/'))
    if not (web/'index.html').is_file() or not web_avatar.is_file() or digest(web_avatar) != config['avatar']['sha256']:
        raise ValueError('Build the current web app first; its served avatar must match the pinned GLB')
    voice = config['voice']; provider = ProviderSettings(); interpreter = None
    if args.with_voice:
        if args.inference_python is None: raise ValueError('--with-voice needs --inference-python')
        if not local(repo, voice['binding_path']).is_file(): raise ValueError('Pinned voice runtime binding is missing')
        # Resolving this symlink discards the virtualenv and selects its base Python.
        executable = args.inference_python.expanduser().absolute()
        probe = subprocess.run([str(executable), '-c',
            "import importlib.util,json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
            "'modules':{name:importlib.util.find_spec(name) is not None for name in ['torch','qwen_tts']}}))"],
            capture_output=True, text=True, timeout=15)
        if probe.returncode != 0: raise ValueError('The selected voice Python could not complete its dependency preflight')
        interpreter = json.loads(probe.stdout)
        if not all(interpreter['modules'].values()): raise ValueError('The selected voice Python lacks torch or qwen_tts')
        provider = ProviderSettings(base_url=f'http://127.0.0.1:{args.voice_port}/v1', model=voice['provider_model'],
                                    api_key=os.getenv('BIAS_VOICE_API_KEY'))
    # Explicit settings avoid silently inheriting a historical provider or snapshot.
    settings = Settings(corpus_dir=corpus_path, web_dir=web, persona_snapshot=snapshot_path,
        retrieval_backend=config['text']['retrieval_backend'], style_examples=False,
        llm=ProviderSettings(), asr=ProviderSettings(), tts=provider,
        tts_voice=voice['provider_voice'] if args.with_voice else 'default',
        tts_persona_id='jensen-huang' if args.with_voice else None, tts_language_hint=args.with_voice,
        motion_catalog=catalog_path, motion_asset_root=asset_root, request_timeout=180 if args.with_voice else 30)
    summary = motion.summary()
    public = {'configuration_sha256':digest(config_path),'persona_id':corpus.persona.id,
        'corpus_sha256':corpus_digest(corpus),'snapshot_sha256':digest(snapshot_path),'snapshot':snapshot.summary(),
        'text_mode':'evidence-preview','retrieval_backend':settings.retrieval_backend,
        'language_model_configured':False,'voice_enabled':args.with_voice,'avatar_sha256':digest(avatar_path),
        'voice_interpreter':interpreter,
        'motion_catalog_sha256':digest(catalog_path),'motion':{key:summary[key] for key in
        ['source_events','observed_frames','observed_seconds','preview_runtime_primitives','reviewed_runtime_primitives']}}
    return settings, corpus, config, public


@contextmanager
def managed_voice(args, config: dict):
    """Keep provider and API in one process tree/network namespace."""
    process = None; stream = None
    if args.with_voice:
        repo = args.repository.resolve(); voice = config['voice']
        command = [str(args.inference_python.expanduser().absolute()), '-m', voice['launcher_module'],
            '--binding', str(local(repo,voice['binding_path'])), '--artifacts', str(args.artifacts.resolve()),
            '--condition', voice['condition'], '--allow-candidates', '--port', str(args.voice_port)]
        env = dict(os.environ); env['PYTHONPATH'] = str(repo/'src')+os.pathsep+str(repo)
        if getattr(args,'output',None):
            stream = (args.output/'voice-service.log').open('w'); target = stream
        else: target = None
        process = subprocess.Popen(command,cwd=repo,env=env,stdout=target,stderr=subprocess.STDOUT)
    try:
        if process:
            deadline=time.monotonic()+90; health=None
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError('Voice launcher exited before HTTP readiness')
                try:
                    with urlopen(f'http://127.0.0.1:{args.voice_port}/health',timeout=2) as response: health=json.load(response)
                    break
                except (OSError,ValueError): time.sleep(.1)
            if health is None: raise RuntimeError('Voice HTTP readiness timed out')
            # The service is lazy: readiness is not a completed model load or waveform.
            if health.get('persona_id') != 'jensen-huang': raise RuntimeError('Voice HTTP identity differs')
        yield
    finally:
        if process:
            process.terminate()
            try: process.wait(timeout=8)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=5)
        if stream: stream.close()
