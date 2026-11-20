"""Build the local studio with whole-answer voice and local Jensen avatar options."""
from __future__ import annotations

import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.common import digest, read_json
from scripts.avatar_demo import GESTURE_STUDIO, PROJECT

VERSION = "jensen-photo-fit-v4"
ASSETS = PROJECT / "data/local/avatars" / VERSION
GAUSSIAN_BUNDLE = "jensen_a_c1c3_240k_sh"
GAUSSIAN_ASSETS = ASSETS.parent / "gaussian-head"
GESTURE_ASSETS = ASSETS.parent / "gesture"
GESTURE_MODELS = ("jensen-gesture-basis.json", "jensen-gesture-basis-expanded.json")
GAUSSIAN_SOURCE = Path(__file__).with_name("gaussian_head")
GAUSSIAN_SHA256 = {
    "json": "3891190a06c81e1968a625b868f6e41f5651cb86a41ef3749991a81155c1f753",
    "bin": "32d5bc15901a8673901c6234ae7af946d2bfcf2c0aa5e7ed9c816d1c3aff599c",
}


def gaussian_paths(root):
    return {ext: root / f"{GAUSSIAN_BUNDLE}.{ext}" for ext in GAUSSIAN_SHA256}


def verify_gaussian_assets(root):
    for ext, path in gaussian_paths(root).items():
        if not path.is_file() or digest(path) != GAUSSIAN_SHA256[ext]:
            raise ValueError(f"Invalid Gaussian head asset: {path.name}")


def gaussian_record():
    if not any(path.exists() for path in gaussian_paths(GAUSSIAN_ASSETS).values()):
        return None
    verify_gaussian_assets(GAUSSIAN_ASSETS)
    return {
        "name": GAUSSIAN_BUNDLE, "sha256": GAUSSIAN_SHA256,
        "renderer_pr": 21, "renderer_revision": "61385492e18c47a49104ba26cd9cbab46e6068f2",
        "lip_motion": "tracked_flame_mouth_with_speech_cues_and_energy_fallback", "usage": "internal_research_only",
    }


def replace_text(path, *edits):
    source = path.read_text()
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"Studio source moved: {path.name}: {old}")
        source = source.replace(old, new, 1)
    path.write_text(source)


def edit_voice(web, *, composite=False):
    app = web / "src/App.tsx"
    if composite:
        replace_text(app,
            ("    const useLiveVoice = Boolean(",
             "    const useLiveVoice = avatarQuery.get('voicePlayback') === 'segments' && Boolean("),
            ("    if (!question || busy || connection", "    if (!question || busy || currentRequest.current || connection"),
            ("disabled={busy} aria-label=\"Play response\"", "disabled={busy || streamingVoice || playback.status !== 'idle'} aria-label=\"Play response\""))
        cache_full_turn_voice(web)
        return
    source = app.read_text()
    # Drop the sentence PCM player from the local UI. The existing playback
    # controller owns the full-turn clip and shares it with manual Listen.
    for first, last, replacement in (
        ("  const [pcmState,", "  const measuredMotion", ""),
        ("    const useLiveVoice =", "      await streamChat(", "    try {\n"),
        ("        if (event.event === 'meta')", "        if (event.event === 'status')",
         "        if (event.event === 'meta') setSessionId(event.data.session_id);\n"),
    ):
        start = source.index(first)
        end = source.index(last, start)
        source = source[:start] + replacement + source[end:]
    app.write_text(source)
    replace_text(app,
        ("import { createPcmPlayback, initialPcmState } from './hooks/pcmPlayback';\n"
         "import type { PcmState } from './hooks/pcmPlayback';\n", ""),
        ("currentRequest.current?.abort(); pcmPlayer.current?.stop();", "currentRequest.current?.abort();"),
        ("    pcmPlayer.current?.stop();\n", ""),
        ("    if (!question || busy || connection", "    if (!question || busy || currentRequest.current || connection"),
        ("    pcmPlayer.current?.stop(); setPcmState(initialPcmState); setActiveTurn(undefined);",
         "    setActiveTurn(undefined);"),
        ("controller.signal, async (event) => {", "controller.signal, (event) => {"),
        ("      }, { live: useLiveVoice });", "      });"),
        ("        if (useLiveVoice) await pcmPlayer.current?.finish(audioFrames);\n"
         "        else if (current.autoPlay", "        if (current.autoPlay"),
        ("        pcmPlayer.current?.stop(error instanceof Error ? error.message : 'Streaming playback failed.');\n", ""),
        ("pcmPlayer.current?.stop(); setPcmState(initialPcmState); setActiveTurn(turn);", "setActiveTurn(turn);"),
        ("  const speaking = pcmState.status === 'playing' || playback.status === 'playing';\n"
         "  const visualPlayback = streamingVoice ? { ...playback, ...pcmState, speechTiming: undefined, audioSha256: undefined } : playback;\n",
         "  const speaking = playback.status === 'playing';\n"),
        ("(playback.status === 'loading' || pcmState.status === 'loading')", "playback.status === 'loading'"),
        ("timeline={streamingVoice ? undefined : activeTurn?.behavior}", "timeline={speaking ? activeTurn?.behavior : undefined}"),
        ("elapsedMs={visualPlayback.elapsedMs}", "elapsedMs={playback.elapsedMs}"),
        ("amplitude={speaking ? visualPlayback.amplitude : 0}", "amplitude={speaking ? playback.amplitude : 0}"),
        ("speechTiming={visualPlayback.speechTiming} audioSha256={visualPlayback.audioSha256} audioElapsedMs={visualPlayback.audioElapsedMs}",
         "speechTiming={playback.speechTiming} audioSha256={playback.audioSha256} audioElapsedMs={playback.audioElapsedMs}"),
        ("streamingVoice ? 'Streaming voice · PCM-clock mouth movement' : voice === 'provider'",
         "voice === 'provider'"),
        ("aria-label={streamingVoice || speaking || playback.status === 'loading' ?", "aria-label={playback.status !== 'idle' ?"),
        ("onClick={() => streamingVoice ? cancel() : speaking || playback.status === 'loading' ? playback.stop() : setSettings(true)}",
         "onClick={() => playback.status !== 'idle' ? playback.stop() : setSettings(true)}"),
        ("{streamingVoice || speaking || playback.status === 'loading' ? <Square", "{playback.status !== 'idle' ? <Square"),
        ("disabled={busy} aria-label=\"Play response\"", "disabled={busy || playback.status !== 'idle'} aria-label=\"Play response\""),
        ("(playback.error || pcmState.error || microphone.error)", "(playback.error || microphone.error)"),
        ("{playback.error || pcmState.error || microphone.error}", "{playback.error || microphone.error}"),
        ("<LiveTiming measurement={liveMeasurement} pcm={pcmState.status === 'idle' ? null : pcmState}/>",
         "<LiveTiming measurement={liveMeasurement}/>"),
        ("setVoice={(next) => { if (streamingVoice) cancel(); playback.stop(); setVoice(next); }}",
         "setVoice={(next) => { playback.stop(); setVoice(next); }}"))
    cache_full_turn_voice(web)


def cache_full_turn_voice(web):
    replace_text(web / "src/hooks/playback.ts",
        ("import type { Turn } from '../api';", "import type { PreparedSpeech, Turn } from '../api';"),
        ("  let generation = 0; let disposed", "  let preparedReply: { text: string; language: Turn['language']; clip: PreparedSpeech } | undefined;\n"
         "  let generation = 0; let disposed"),
        ("dispose() { disposed = true;", "dispose() { disposed = true; preparedReply = undefined;"),
        ("          const prepared = environment.prepareSpeech ? await environment.prepareSpeech(speechText, turn.language, abort.signal) : undefined;\n"
         "          if (!isCurrent()) return;",
         "          let prepared = preparedReply?.text === speechText && preparedReply.language === turn.language ? preparedReply.clip : undefined;\n"
         "          if (!prepared && environment.prepareSpeech) {\n"
         "            prepared = await environment.prepareSpeech(speechText, turn.language, abort.signal);\n"
         "            if (!isCurrent()) return;\n"
         "            preparedReply = { text: speechText, language: turn.language, clip: prepared };\n"
         "          }"))
    replace_text(web / "src/components/SettingsDialog.tsx",
        ("Use the configured speech provider.", "Read the complete reply with the configured speech provider."))


def edit_studio(web):
    src = web / "src"
    replace_text(src / "avatar/builtin.ts",
        ("export type BuiltinAvatar = 'human' | 'portrait' | 'jensen';",
         f"export const JENSEN_PHOTO_AVATAR = {{ id: '{VERSION}', modelUrl: '/models/{VERSION}.glb' }} as const;\n"
         "export type BuiltinAvatar = 'human' | 'portrait' | 'jensen' | 'jensen-photo';"))
    replace_text(src / "avatar/retarget.ts",
        ("createResearchHumanMapping, JENSEN_SOURCE_FITTED_AVATAR,",
         "createResearchHumanMapping, JENSEN_PHOTO_AVATAR, JENSEN_SOURCE_FITTED_AVATAR,"),
        ("[RESEARCH_HUMAN_AVATAR.modelUrl, JENSEN_SOURCE_FITTED_AVATAR.modelUrl]",
         "[RESEARCH_HUMAN_AVATAR.modelUrl, JENSEN_SOURCE_FITTED_AVATAR.modelUrl, JENSEN_PHOTO_AVATAR.modelUrl]"))
    replace_text(src / "App.tsx",
        ("import { JENSEN_SOURCE_FITTED_AVATAR,", "import { JENSEN_PHOTO_AVATAR, JENSEN_SOURCE_FITTED_AVATAR,"),
        ("if (next === 'jensen' && persona?.id", "if ((next === 'jensen' || next === 'jensen-photo') && persona?.id"),
        ("if (next === 'jensen') setFraming", "if (next === 'jensen' || next === 'jensen-photo') setFraming"),
        ("if (!avatarChoiceMade.current) setBuiltinAvatar(persona.id === 'jensen-huang' ? 'portrait' : 'human');",
         "if (!avatarChoiceMade.current) {\n"
         "      setBuiltinAvatar(persona.id === 'jensen-huang' ? 'jensen-photo' : 'human');\n"
         "      if (persona.id === 'jensen-huang') setFraming('upper_body');\n"
         "    }"),
        ("builtinAvatar === 'jensen' && isJensen ? JENSEN_SOURCE_FITTED_AVATAR.modelUrl : undefined",
         "builtinAvatar === 'jensen' && isJensen ? JENSEN_SOURCE_FITTED_AVATAR.modelUrl : "
         "builtinAvatar === 'jensen-photo' && isJensen ? JENSEN_PHOTO_AVATAR.modelUrl : undefined"),
        ("builtinAvatar === 'jensen' && isJensen ? 'JENSEN · SOURCE-FITTED CANDIDATE'",
         "builtinAvatar === 'jensen-photo' && isJensen ? 'JENSEN · PHOTO-FIT V4' : "
         "builtinAvatar === 'jensen' && isJensen ? 'JENSEN · SOURCE-FITTED V3'"),
        ("selectBuiltinAvatar(isJensen ? 'portrait' : 'human')", "selectBuiltinAvatar(isJensen ? 'jensen-photo' : 'human')"))
    replace_text(src / "components/SettingsDialog.tsx",
        ("[{ id: 'jensen' as const, title: 'Jensen source-fitted candidate',",
         "[{ id: 'jensen-photo' as const, title: 'Jensen photo-fit v4', note: 'Photo-derived face texture with an articulated rig.' }, "
         "{ id: 'jensen' as const, title: 'Jensen source-fitted v3',"),
        ("        <label className=\"switch-row\"><span>Camera framing",
         f"        {{props.persona?.id === 'jensen-huang' && props.builtinAvatar === 'jensen-photo' && <p>"
         f"Face texture adapted from Stanford photographs by Anders Eidesvik, CC BY-SA 4.0. "
         f"<a href=\"/models/{VERSION}.LICENSE.md\" target=\"_blank\" rel=\"noreferrer\">Sources, license and changes</a>.</p>}}\n"
         "        <label className=\"switch-row\"><span>Camera framing"),
        ("Restore default portrait", "Restore default avatar"))


def appearance_record():
    manifest = ASSETS / "appearance.json"
    if not manifest.is_file():
        return None
    record = read_json(manifest)
    if (record["version"] != VERSION
            or digest(ASSETS / f"{VERSION}.glb") != record["sha256"]
            or digest(ASSETS / f"{VERSION}.LICENSE.md") != record["license_sha256"]):
        raise ValueError("Photo-fit GLB or license differs from its appearance receipt")
    return record


def edit_gaussian(web):
    src = web / "src"
    shutil.copytree(GAUSSIAN_SOURCE, src, dirs_exist_ok=True)
    replace_text(src / "avatar/builtin.ts",
        ("'jensen' | 'jensen-photo';", "'jensen' | 'jensen-photo' | 'jensen-gaussian';"))
    replace_text(src / "App.tsx",
        ("import { AvatarStage } from './components/AvatarStage';",
         "import { AvatarStage } from './components/AvatarStage';\n"
         "import { GaussianHeadStage } from './components/GaussianHeadStage';"),
        ("(next === 'jensen' || next === 'jensen-photo') && persona?.id",
         "(next === 'jensen' || next === 'jensen-photo' || next === 'jensen-gaussian') && persona?.id"),
        ("  const isJensen = persona?.id === 'jensen-huang';",
         "  const isJensen = persona?.id === 'jensen-huang';\n"
         "  const showGaussian = isJensen && !model && builtinAvatar === 'jensen-gaussian';"),
        ("      setBuiltinAvatar(persona.id === 'jensen-huang' ? 'jensen-photo' : 'human');",
         "      const photoPreferred = new URLSearchParams(window.location.search).get('head') === 'photo';\n"
         "      setBuiltinAvatar(persona.id === 'jensen-huang' ? (photoPreferred ? 'jensen-photo' : 'jensen-gaussian') : 'human');"),
        ("builtinAvatar === 'jensen-photo' && isJensen ? JENSEN_PHOTO_AVATAR.modelUrl",
         "(builtinAvatar === 'jensen-photo' || builtinAvatar === 'jensen-gaussian') && isJensen ? JENSEN_PHOTO_AVATAR.modelUrl"),
        ('<div className="avatar-viewport"><AvatarStage',
         f'<div className="avatar-viewport">{{showGaussian ? <GaussianHeadStage bundle="{GAUSSIAN_BUNDLE}" '
         'speaking={speaking} amplitude={playback.amplitude} audioElapsedMs={playback.audioElapsedMs} '
         'speechTiming={playback.speechTiming} audioSha256={playback.audioSha256} '
         'reducedMotion={reducedMotion} onReady={onAvatarReady} onError={onAvatarError}/> : <AvatarStage'),
        ('onReady={onAvatarReady} onError={onAvatarError}/></div>',
         'onReady={onAvatarReady} onError={onAvatarError}/>}</div>'),
        ("model ? 'YOUR CUSTOM RIG' : builtinAvatar === 'jensen-photo'",
         "model ? 'YOUR CUSTOM RIG' : showGaussian ? 'JENSEN · GAUSSIAN HEAD · RESEARCH PREVIEW' : builtinAvatar === 'jensen-photo'"),
        ("title={playback.timingMessage}>{voice === 'provider' ?",
         "title={showGaussian ? undefined : playback.timingMessage}>{showGaussian && voice === 'provider' ? (new URLSearchParams(window.location.search).get('mouth') !== 'energy' && playback.speechTiming ? 'Provider voice · timed mouth shapes' : 'Provider voice · audio-driven jaw') : voice === 'provider' ?"),
        ("<button onClick={() => selectBuiltinAvatar('portrait')}>Use stylized portrait</button>",
         "<button onClick={() => selectBuiltinAvatar(showGaussian ? 'jensen-photo' : 'portrait')}>"
         "{showGaussian ? 'Use photo-fit v4' : 'Use stylized portrait'}</button>"),
        ("</section><AttentionPanel enabled={page === 'studio'}",
         "</section>{!showGaussian && <><AttentionPanel enabled={page === 'studio'}"),
        ("!activeTurn.behavior.motion)}/></div>", "!activeTurn.behavior.motion)}/></>}</div>"))
    replace_text(src / "components/SettingsDialog.tsx",
        ("[{ id: 'jensen-photo' as const, title: 'Jensen photo-fit v4',",
         "[{ id: 'jensen-gaussian' as const, title: 'Jensen Gaussian head', "
         "note: 'Research preview · speech-driven mouth and tracked head motion.' }, "
         "{ id: 'jensen-photo' as const, title: 'Jensen photo-fit v4',"),
        ('        <label className="switch-row"><span>Camera framing',
         "        {props.builtinAvatar === 'jensen-gaussian' && !props.modelName && <p>The Gaussian preview uses its recorded camera.</p>}\n"
         "        {(props.builtinAvatar !== 'jensen-gaussian' || props.modelName) && "
         '<label className="switch-row"><span>Camera framing'),
        ('</select></label><label className="upload-model">', '</select></label>}<label className="upload-model">'))


def build_studio(studio, record, gaussian=None):
    tag = f'{record["sha256"]}-{record["license_sha256"]}' if record else "base"
    composite = studio == GESTURE_STUDIO
    upstream_web = studio / "apps/web"
    gesture_files = [GESTURE_ASSETS / name for name in GESTURE_MODELS if (GESTURE_ASSETS / name).is_file()] if composite else []
    parts = [digest(Path(__file__))]
    if gaussian:
        parts.extend(GAUSSIAN_SHA256.values())
        source = upstream_web / "src" if composite else GAUSSIAN_SOURCE
        parts += [digest(path) for path in sorted(source.rglob("*")) if path.suffix in {".ts", ".tsx", ".css"}]
    parts += [f"{path.name}:{digest(path)}" for path in gesture_files]
    key = sha256("\n".join(parts).encode()).hexdigest()
    output = PROJECT / "data/local/studio-builds" / studio.name / f"{tag}-{key}" / "dist"
    if (output / "index.html").is_file():
        return output
    node = PROJECT / ".tools/node-runtime/bin/node"
    if not node.is_file():
        node_path = shutil.which("node")
        if not node_path:
            raise ValueError("Install Node.js before building the YAICON studio")
        node = Path(node_path)
    deps = studio / "node_modules"
    output.parent.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent.parent) as stage:
        web = Path(stage) / "web"
        web.mkdir()
        for name in ("index.html", "package.json", "vite.config.ts", "tsconfig.json"):
            shutil.copyfile(upstream_web / name, web / name)
        for name in ("src", "public"):
            shutil.copytree(upstream_web / name, web / name)
        (web / "node_modules").symlink_to(deps, target_is_directory=True)
        if record:
            for ext in ("glb", "LICENSE.md"):
                shutil.copyfile(ASSETS / f"{VERSION}.{ext}", web / "public/models" / f"{VERSION}.{ext}")
            edit_studio(web)
        edit_voice(web, composite=composite)
        if gaussian:
            models = web / "public/models/gaussian-head"
            models.mkdir(parents=True, exist_ok=True)
            for asset in gaussian_paths(GAUSSIAN_ASSETS).values():
                shutil.copyfile(asset, models / asset.name)
            verify_gaussian_assets(models)
            if composite:
                replace_text(web / "src/App.tsx",
                    ("const gaussianRequested = avatarQuery.get('head') === 'gaussian';",
                     "const gaussianRequested = (avatarQuery.get('head') ?? 'gaussian') === 'gaussian';"))
            else:
                edit_gaussian(web)
        if gesture_files:
            gestures = web / "public/models/gesture"
            gestures.mkdir(parents=True, exist_ok=True)
            for asset in gesture_files:
                shutil.copyfile(asset, gestures / asset.name)
        env = dict(os.environ, PATH=str(node.parent) + os.pathsep + os.environ.get("PATH", ""))
        subprocess.run([node, deps / "typescript/bin/tsc", "--noEmit"], cwd=web, env=env, check=True)
        subprocess.run([node, deps / "vite/bin/vite.js", "build"], cwd=web, env=env, check=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            (web / "dist").rename(output)
        except OSError:
            if not (output / "index.html").is_file():
                raise
    return output
