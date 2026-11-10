import { useEffect, useRef } from 'react';
import { AudioLines, Check, Upload, X } from 'lucide-react';
import type { Persona } from '../api';
import type { VoiceMode } from '../hooks/usePlayback';

interface Props {
  open: boolean; close: () => void; persona: Persona | null; voice: VoiceMode; setVoice: (voice: VoiceMode) => void;
  autoPlay: boolean; setAutoPlay: (value: boolean) => void; reducedMotion: boolean; setReducedMotion: (value: boolean) => void;
  modelName?: string; loadModel: (file: File) => void; resetModel: () => void;
}

export function SettingsDialog(props: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { if (props.open && !dialog.current?.open) dialog.current?.showModal(); else if (!props.open && dialog.current?.open) dialog.current?.close(); }, [props.open]);
  return <dialog ref={dialog} className="settings-dialog" aria-labelledby="settings-title" onCancel={props.close} onClick={(event) => { if (event.target === dialog.current) props.close(); }}>
    <div className="dialog-body"><div className="dialog-heading"><div><span className="eyebrow">MAKE IT YOUR SESSION</span><h2 id="settings-title">Studio settings</h2></div><button onClick={props.close} className="icon-button" aria-label="Close settings"><X size={20}/></button></div>
      <section><h3>Voice playback</h3><p>Choose how responses are spoken. Built-in voices do not reproduce Elon Musk's voice.</p><div className="voice-options">{([{ id: 'off', title: 'Text only', note: 'Keep the conversation quiet.' }, { id: 'browser', title: 'Built-in voice', note: 'Generic browser voice · approximate mouth motion.' }, { id: 'provider', title: 'Connected voice', note: props.persona?.capabilities.speech ? 'Use the configured speech provider.' : 'Connect a speech server to enable.' }] as const).map((option) => <button key={option.id} aria-pressed={props.voice === option.id} className={`voice-option ${props.voice === option.id ? 'selected' : ''}`} disabled={option.id === 'provider' && !props.persona?.capabilities.speech} onClick={() => props.setVoice(option.id)}><AudioLines size={18}/><span><strong>{option.title}</strong><small>{option.note}</small></span>{props.voice === option.id && <Check size={17}/>}</button>)}</div><label className="switch-row"><span>Read new responses aloud</span><input type="checkbox" checked={props.autoPlay} onChange={(event) => props.setAutoPlay(event.target.checked)} disabled={props.voice === 'off'}/></label></section>
      <section><h3>Avatar asset</h3><p>The reference rig is a stylized prototype. Bring a rigged GLB to test a custom likeness and morph targets.</p><label className="upload-model"><Upload size={19}/><span>{props.modelName || 'Choose a local GLB model'}<small>Stays in your browser · up to 50 MB</small></span><input type="file" accept=".glb" onChange={(event) => { const file = event.target.files?.[0]; if (file) props.loadModel(file); event.target.value = ''; }}/></label>{props.modelName && <button className="text-button" onClick={props.resetModel}>Restore reference rig</button>}<label className="switch-row"><span>Reduce automatic motion</span><input type="checkbox" checked={props.reducedMotion} onChange={(event) => props.setReducedMotion(event.target.checked)}/></label></section>
      <div className="settings-note">{props.persona?.capabilities.llm ? 'A language model is connected.' : 'Source preview is active. Configure a language model on the server for generated dialogue.'}</div>
    </div>
  </dialog>;
}

