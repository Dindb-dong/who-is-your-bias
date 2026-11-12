import { useEffect, useRef } from 'react';
import { Activity, AudioLines, Check, Upload, X } from 'lucide-react';
import type { MotionLibraryStatus, Persona } from '../api';
import type { MotionMode } from '../avatar/types';
import type { AvatarFraming } from '../avatar/builtin';
import type { VoiceMode } from '../hooks/usePlayback';

interface Props {
  open: boolean; close: () => void; persona: Persona | null; voice: VoiceMode; setVoice: (voice: VoiceMode) => void;
  autoPlay: boolean; setAutoPlay: (value: boolean) => void; reducedMotion: boolean; setReducedMotion: (value: boolean) => void;
  modelName?: string; loadModel: (file: File) => void; resetModel: () => void;
  builtinAvatar: 'human' | 'portrait'; setBuiltinAvatar: (avatar: 'human' | 'portrait') => void;
  framing: AvatarFraming; setFraming: (framing: AvatarFraming) => void;
  motionMode: MotionMode; setMotionMode: (mode: MotionMode) => void;
  motionLibrary: MotionLibraryStatus | null; motionLibraryError?: string;
}

export function SettingsDialog(props: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { if (props.open && !dialog.current?.open) dialog.current?.showModal(); else if (!props.open && dialog.current?.open) dialog.current?.close(); }, [props.open]);
  return <dialog ref={dialog} className="settings-dialog" aria-labelledby="settings-title" onCancel={props.close} onClick={(event) => { if (event.target === dialog.current) props.close(); }}>
    <div className="dialog-body"><div className="dialog-heading"><div><span className="eyebrow">MAKE IT YOUR SESSION</span><h2 id="settings-title">Studio settings</h2></div><button onClick={props.close} className="icon-button" aria-label="Close settings"><X size={20}/></button></div>
      <section><h3>Voice playback</h3><p>Choose how responses are spoken. Built-in voices are generic and do not reproduce the subject's voice.</p><div className="voice-options">{([{ id: 'off', title: 'Text only', note: 'Keep the conversation quiet.' }, { id: 'browser', title: 'Built-in voice', note: 'Generic browser voice · approximate mouth motion.' }, { id: 'provider', title: 'Connected voice', note: props.persona?.capabilities.speech ? 'Use the configured speech provider.' : 'Connect a speech server to enable.' }] as const).map((option) => <button key={option.id} aria-pressed={props.voice === option.id} className={`voice-option ${props.voice === option.id ? 'selected' : ''}`} disabled={option.id === 'provider' && !props.persona?.capabilities.speech} onClick={() => props.setVoice(option.id)}><AudioLines size={18}/><span><strong>{option.title}</strong><small>{option.note}</small></span>{props.voice === option.id && <Check size={17}/>}</button>)}</div><label className="switch-row"><span>Read new responses aloud</span><input type="checkbox" checked={props.autoPlay} onChange={(event) => props.setAutoPlay(event.target.checked)} disabled={props.voice === 'off'}/></label></section>
      <section><h3>Avatar asset</h3><p>Choose a research avatar or load your own rig. The included models are artistic prototypes.</p>
        <div className="voice-options">{([
          { id: 'human', title: 'Research human', note: 'Human mesh with facial expressions and articulated limbs.' },
          { id: 'portrait', title: 'Stylized portrait', note: props.persona?.id === 'jensen-huang' ? 'Jensen-inspired preview with glasses, gray hair and a dark jacket.' : 'Lightweight portrait with designed expressions.' },
        ] as const).map((option) => <button key={option.id} type="button" aria-pressed={!props.modelName && props.builtinAvatar === option.id} className={`voice-option ${!props.modelName && props.builtinAvatar === option.id ? 'selected' : ''}`} onClick={() => props.setBuiltinAvatar(option.id)}><span><strong>{option.title}</strong><small>{option.note}</small></span>{!props.modelName && props.builtinAvatar === option.id && <Check size={17}/>}</button>)}</div>
        <label className="switch-row"><span>Camera framing</span><select aria-label="Avatar camera framing" value={props.framing} onChange={(event) => props.setFraming(event.target.value as AvatarFraming)}><option value="portrait">Portrait</option><option value="full_body">Full body</option></select></label><label className="upload-model"><Upload size={19}/><span>{props.modelName || 'Choose a local GLB model'}<small>Stays in your browser · up to 50 MB</small></span><input type="file" accept=".glb" onChange={(event) => { const file = event.target.files?.[0]; if (file) props.loadModel(file); event.target.value = ''; }}/></label>{props.modelName && <button className="text-button" onClick={props.resetModel}>Restore default portrait</button>}<label className="switch-row"><span>Reduce automatic motion</span><input type="checkbox" checked={props.reducedMotion} onChange={(event) => props.setReducedMotion(event.target.checked)}/></label></section>
      <section><h3>Conversation movement</h3><p>Designed motion is the default. Source clips replay a recorded movement once at its original speed, independently of voice. Context matching does not predict a person's emotion or gesture choice.</p>
        <div className="voice-options">{([
          { id: 'disabled', title: 'Designed motion', note: 'Use the existing authored expression and gesture cues.', enabled: true },
          { id: 'reviewed', title: 'Reviewed source motion', note: props.motionLibrary?.reviewed_runtime_primitives ? `${props.motionLibrary.reviewed_runtime_primitives} reviewed source windows available.` : 'No speaker-reviewed source windows are available.', enabled: Boolean(props.motionLibrary?.reviewed_runtime_primitives) },
          { id: 'geometry_preview', title: 'Source geometry preview', note: props.motionLibrary?.preview_runtime_primitives ? `${props.motionLibrary.preview_runtime_primitives} measured windows. Speaker association remains unreviewed.` : 'Load a measured-motion library on the server to enable.', enabled: Boolean(props.motionLibrary?.available && props.motionLibrary.preview_runtime_primitives) },
        ] as const).map((option) => <button key={option.id} type="button" aria-pressed={props.motionMode === option.id} className={`voice-option ${props.motionMode === option.id ? 'selected' : ''}`} disabled={!option.enabled} onClick={() => props.setMotionMode(option.id)}><Activity size={18}/><span><strong>{option.title}</strong><small>{option.note}</small></span>{props.motionMode === option.id && <Check size={17}/>}</button>)}</div>
        {props.motionMode === 'geometry_preview' && <p style={{ marginTop: 10 }}>Source-geometry preview is enabled for evidence-supported answers in the full conversation mode. It does not establish that the tracked motion belongs to the named speaker.</p>}
        {props.reducedMotion && <p style={{ marginTop: 10 }}>Measured replays are paused while reduced motion is enabled.</p>}
        {props.motionLibraryError && <p role="status" style={{ marginTop: 10 }}>{props.motionLibraryError}</p>}
      </section>
      <div className="settings-note">{props.persona?.capabilities.llm ? 'A language model is connected.' : 'Source preview is active. Configure a language model on the server for generated dialogue.'}</div>
    </div>
  </dialog>;
}


