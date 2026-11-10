import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowRight, ArrowUp, ArrowUpRight, BookOpen, ChevronDown, ChevronRight, CircleHelp, FlaskConical, Globe2, Layers3, LoaderCircle, MessageCircle, Mic, Plus, Radio, RotateCcw, Settings2, Sparkles, Square, Volume2, VolumeX, X } from 'lucide-react';
import { getJson, streamChat } from './api';
import type { Citation, ExperimentMode, Language, Persona, Source, Turn } from './api';
import { AvatarStage } from './components/AvatarStage';
import { SourcesView } from './components/SourcesView';
import { experiments, ResearchLab } from './components/ResearchLab';
import { SettingsDialog } from './components/SettingsDialog';
import { MotionLab } from './components/MotionLab';
import { usePlayback } from './hooks/usePlayback';
import type { VoiceMode } from './hooks/usePlayback';
import { useMicrophone } from './hooks/useMicrophone';

interface Message { id: string; role: 'user' | 'assistant'; text: string; turn?: Turn; pending?: boolean; error?: string }
type Page = 'studio' | 'sources' | 'lab';
const defaultQuestions = ['Why did Tesla start with a sports car?', 'How do you solve difficult problems?', 'What makes an idea worth pursuing?'];
const stages: Record<string, string> = { retrieving: 'Finding relevant sources', generating: 'Composing a response', planning: 'Planning expression', complete: 'Response ready' };

function Mark({ small = false }: { small?: boolean }) {
  return <span className={`brand-mark ${small ? 'small' : ''}`} aria-hidden="true"><i/><i/><i/></span>;
}

function EvidencePanel({ citations, onClose }: { citations: Citation[]; onClose: () => void }) {
  return <aside className="evidence-drawer" aria-label="Response evidence"><div className="drawer-header"><div><span className="eyebrow">FOLLOW THE THOUGHT</span><h2>Response evidence</h2></div><button className="icon-button" onClick={onClose} aria-label="Close evidence"><X size={18}/></button></div><p className="drawer-intro">Source notes used for this response. These are paraphrases, not direct quotations.</p>{citations.length ? citations.map((citation, index) => <article className="evidence-card" key={citation.evidence_id}><div className="evidence-index">{String(index + 1).padStart(2, '0')}<span>{citation.locator.label}</span></div><h3>{citation.title}</h3><p>{citation.text}</p><a href={citation.url} target="_blank" rel="noreferrer">Read original source <ArrowUpRight size={14}/></a></article>) : <div className="empty-state"><BookOpen size={24}/><p>This response does not have supporting source records.</p></div>}</aside>;
}

export default function App() {
  const [page, setPage] = useState<Page>('studio');
  const [persona, setPersona] = useState<Persona | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [connection, setConnection] = useState<'connecting' | 'ready' | 'error'>('connecting');
  const [connectionError, setConnectionError] = useState('');
  const [reload, setReload] = useState(0);
  const [language, setLanguage] = useState<Language>('en');
  const [mode, setMode] = useState<ExperimentMode>('behavior');
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [activeTurn, setActiveTurn] = useState<Turn | undefined>();
  const [evidence, setEvidence] = useState<Citation[] | null>(null);
  const [settings, setSettings] = useState(false);
  const [voice, setVoice] = useState<VoiceMode>('off');
  const [autoPlay, setAutoPlay] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [model, setModel] = useState<{ name: string; url: string } | undefined>();
  const [avatarError, setAvatarError] = useState<string | null>(null);
  const [avatarReady, setAvatarReady] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  const currentRequest = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const chatBottom = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const playback = usePlayback();
  const playbackPreferences = useRef({ autoPlay, voice });
  playbackPreferences.current = { autoPlay, voice };
  const microphone = useMicrophone(language, setDraft);
  const onAvatarReady = useCallback(() => { setAvatarReady(true); }, []);
  const onAvatarError = useCallback((error: Error) => { setAvatarError(error.message); }, []);

  useEffect(() => {
    const controller = new AbortController();
    setConnection('connecting');
    Promise.all([
      getJson<Persona>('/api/persona', controller.signal),
      getJson<{ sources: Source[] }>('/api/sources', controller.signal),
    ]).then(([profile, corpus]) => { setPersona(profile); setSources(corpus.sources); setConnection('ready'); setConnectionError(''); })
      .catch((error: unknown) => { if (!controller.signal.aborted) { setConnection('error'); setConnectionError(error instanceof Error ? error.message : 'The conversation service is unavailable.'); } });
    return () => controller.abort();
  }, [reload]);
  useEffect(() => { chatBottom.current?.scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth', block: 'nearest' }); }, [messages.length, busy, reducedMotion]);
  useEffect(() => () => { generation.current += 1; currentRequest.current?.abort(); }, []);
  useEffect(() => () => { if (model) URL.revokeObjectURL(model.url); }, [model]);

  function cancel() {
    generation.current += 1;
    currentRequest.current?.abort(); currentRequest.current = null;
    setMessages((previous) => previous.map((message) => message.pending ? { ...message, pending: false, text: message.text || 'Response stopped.' } : message));
    setBusy(false); setStatus(''); playback.stop(); microphone.stop();
  }
  function reset() {
    cancel(); setMessages([]); setSessionId(undefined); setActiveTurn(undefined); setEvidence(null); setDraft('');
    composer.current?.focus();
  }
  function navigate(next: Page) { cancel(); setPage(next); setEvidence(null); }
  async function send(text = draft) {
    const question = text.trim();
    if (!question || busy || connection !== 'ready') return;
    playback.stop(); microphone.stop(); setDraft(''); setEvidence(null);
    const ticket = ++generation.current;
    const controller = new AbortController(); currentRequest.current = controller;
    const answerId = crypto.randomUUID();
    setMessages((previous) => [...previous, { id: crypto.randomUUID(), role: 'user', text: question }, { id: answerId, role: 'assistant', text: '', pending: true }]);
    setBusy(true); setStatus('Finding relevant sources');
    let received: Turn | undefined;
    try {
      await streamChat({ message: question, session_id: sessionId, mode, language }, controller.signal, (event) => {
        if (ticket !== generation.current) return;
        if (event.event === 'meta') setSessionId(event.data.session_id);
        if (event.event === 'status') setStatus(stages[event.data.stage] ?? event.data.stage.replaceAll('_', ' '));
        if (event.event === 'token') setMessages((previous) => previous.map((message) => message.id === answerId ? { ...message, text: message.text + event.data.text } : message));
        if (event.event === 'result') {
          received = event.data.turn; setActiveTurn(received);
          setMessages((previous) => previous.map((message) => message.id === answerId ? { ...message, text: event.data.turn.text, turn: event.data.turn, pending: false } : message));
        }
      });
      const current = playbackPreferences.current;
      if (ticket === generation.current && received && current.autoPlay && current.voice !== 'off') void playback.play(received, current.voice);
    } catch (error) {
      if (ticket === generation.current && !controller.signal.aborted) {
        const message = error instanceof Error ? error.message : 'The response could not be completed.';
        setMessages((previous) => previous.map((item) => item.id === answerId ? { ...item, pending: false, error: message } : item));
      }
    } finally {
      if (ticket === generation.current) { setBusy(false); setStatus(''); currentRequest.current = null; setMessages((previous) => previous.map((item) => item.id === answerId ? { ...item, pending: false } : item)); }
    }
  }
  function loadModel(file: File) {
    if (!file.name.toLowerCase().endsWith('.glb') || file.size > 50 * 1024 * 1024) { setAvatarError('Choose a GLB file smaller than 50 MB.'); return; }
    setAvatarError(null); setAvatarReady(false); setModel({ name: file.name, url: URL.createObjectURL(file) });
  }
  function listen(turn: Turn) { if (busy) return; setActiveTurn(turn); void playback.play(turn, voice === 'off' ? 'browser' : voice); }
  const questions = persona?.example_questions[language] ?? defaultQuestions;
  const speaking = playback.status === 'playing';
  const stageState = speaking ? 'Speaking' : playback.status === 'loading' ? 'Preparing voice' : microphone.listening ? 'Listening' : busy ? 'Thinking' : 'Ready when you are';

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={(event) => { event.preventDefault(); navigate('studio'); }} aria-label="Who is your bias home"><Mark/><span>who is<br/>your <b>bias?</b></span></a>
      <div className="workspace-label">PERSONA WORKSPACE<span>01</span></div>
      <nav aria-label="Main navigation"><button className={page === 'studio' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('studio')}><MessageCircle size={19}/><span>Conversation</span>{page === 'studio' && <i/>}</button><button className={page === 'sources' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('sources')}><BookOpen size={19}/><span>Source library</span><span className="nav-count">{sources.length || '—'}</span></button><button className={page === 'lab' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('lab')}><FlaskConical size={19}/><span>Research lab</span></button></nav>
      <div className="sidebar-subject"><span className="eyebrow">IN THE STUDIO</span><div className="subject-mini"><div className="monogram">EM</div><div><strong>Elon Musk</strong><span>Public-source persona</span></div></div><div className="subject-lines"><span>Engineering</span><span>Space</span><span>Ideas</span></div></div>
      <div className="sidebar-bottom"><button className="nav-item" onClick={() => setSettings(true)}><Settings2 size={18}/><span>Studio settings</span></button><button className="nav-item" onClick={() => setInfoOpen(!infoOpen)}><CircleHelp size={18}/><span>About this project</span></button><div className="simulation-note"><span className="tiny-dot"/> An AI simulation, openly labeled.</div></div>
    </aside>
    <div className="workspace">
      <header className="topbar"><div className="breadcrumb"><span>Workspace</span><ChevronRight size={14}/><strong>{page === 'studio' ? 'Conversation' : page === 'sources' ? 'Source library' : 'Research lab'}</strong></div><div className="topbar-actions"><span className={`connection ${connection}`}><i/>{connection === 'ready' ? 'Studio connected' : connection === 'connecting' ? 'Connecting' : 'Service offline'}</span><label className="language-select"><Globe2 size={15}/><select aria-label="Response language" value={language} onChange={(event) => setLanguage(event.target.value as Language)}><option value="en">English</option><option value="ko">한국어</option></select><ChevronDown size={12}/></label><button className="icon-button mobile-settings" onClick={() => setSettings(true)} aria-label="Studio settings"><Settings2 size={18}/></button><div className="user-avatar">Y</div></div></header>
      <main>
        {infoOpen && <div className="about-banner"><Sparkles size={19}/><p><strong>Who Is Your Bias?</strong> A research workspace for source-grounded dialogue, voice, and embodied behavior. This is an AI simulation, not Elon Musk or an official representative. The reference avatar and its movement rules are designed prototypes.</p><button className="icon-button" onClick={() => setInfoOpen(false)} aria-label="Close project information"><X size={17}/></button></div>}
        {connection === 'error' && <div className="connection-error" role="alert"><span><strong>The conversation service is offline.</strong> {connectionError}</span><button onClick={() => setReload((value) => value + 1)}><RotateCcw size={14}/> Reconnect</button></div>}
        {page === 'sources' ? <SourcesView sources={sources}/> : page === 'lab' ? <><ResearchLab language={language}/><MotionLab modelUrl={model?.url} reducedMotion={reducedMotion}/></> : <>
          <div className="studio-heading"><div><div className="eyebrow">CURIOSITY, IN CONVERSATION</div><h1>Meet the mind. <span>Explore the ideas.</span></h1><p>A conversation with an evidence-grounded Elon Musk persona.</p></div><button className="secondary-button" onClick={reset}><Plus size={16}/> New conversation</button></div>
          <div className="studio-grid">
            <section className="stage-card" aria-label="Avatar studio">
              <div className="stage-top"><div><span className="tag dark"><span className="tiny-dot"/> AI SIMULATION</span><div className="stage-title">Elon Musk<span>01 / RESEARCH PERSONA</span></div></div><button className="stage-options" onClick={() => setSettings(true)} aria-label="Avatar settings"><Settings2 size={18}/></button></div>
              <div className="avatar-viewport"><AvatarStage speaking={speaking} timeline={activeTurn?.behavior} elapsedMs={playback.elapsedMs} amplitude={speaking ? playback.amplitude : 0} modelUrl={model?.url} reducedMotion={reducedMotion} speechTiming={playback.speechTiming} audioSha256={playback.audioSha256} audioElapsedMs={playback.audioElapsedMs} onReady={onAvatarReady} onError={onAvatarError}/></div>
              <div className="stage-caption"><span className="caption-rule"/><span>{model ? 'YOUR CUSTOM RIG' : 'STYLIZED REFERENCE RIG'}</span><span className="caption-rule"/></div>
              <div className="stage-bottom"><div className="live-status"><span className={`status-orb ${speaking || busy ? 'animated' : ''}`}/><div><strong>{stageState}</strong><span title={playback.timingMessage}>{voice === 'provider' ? playback.timingStatus === 'available' ? 'Provider voice · audio-timed mouth cues' : 'Provider voice · approximate mouth movement' : voice === 'browser' || speaking ? 'Generic browser voice · approximate lip sync' : 'Text mode · voice available'}</span></div></div><button className="round-control" aria-label={speaking || playback.status === 'loading' ? 'Stop audio' : 'Enable voice settings'} onClick={() => speaking || playback.status === 'loading' ? playback.stop() : setSettings(true)}>{speaking || playback.status === 'loading' ? <Square size={16}/> : voice === 'off' ? <VolumeX size={18}/> : <Volume2 size={18}/>}</button></div>
              {!avatarReady && !avatarError && <div className="avatar-loading"><LoaderCircle size={18} className="spin"/> Preparing the rig</div>}
              {avatarError && <div className="avatar-error" role="alert">{avatarError}<button onClick={() => { setModel(undefined); setAvatarError(null); }}>Use reference rig</button></div>}
            </section>
            <section className="conversation-card" aria-label="Conversation">
              <div className="conversation-header"><div><MessageCircle size={18}/><h2>Let's talk</h2></div><span className="preview-label">{persona?.capabilities.llm ? 'LLM connected' : 'Source preview'}</span></div>
              <div className="message-list" role="log" aria-label="Conversation messages" aria-live="polite" aria-relevant="additions text">
                {!messages.length && <div className="welcome-message"><div className="assistant-avatar"><Mark small/></div><span className="eyebrow">A GOOD QUESTION IS A GOOD START</span><h2>{language === 'ko' ? '어떤 이야기가 궁금해?' : 'Where should we begin?'}</h2><p>{language === 'ko' ? '우주, 공학, 창업에 관한 공개 인터뷰와 글을 바탕으로 질문해 봐.' : 'Explore space, engineering, and the thinking behind big ideas — with the original sources close by.'}</p><div className="starter-questions">{questions.slice(0, 3).map((question) => <button key={question} disabled={connection !== 'ready'} onClick={() => void send(question)}><span>{question}</span><ArrowUpRight size={16}/></button>)}</div><div className="preview-explainer"><BookOpen size={14}/><span>{persona?.capabilities.llm ? 'Responses use a connected model and selected evidence.' : 'Source preview uses published evidence. Connect a model for generated dialogue.'}</span></div></div>}
                {messages.map((message) => <article key={message.id} className={`message ${message.role}`}><div className="message-avatar">{message.role === 'assistant' ? <Mark small/> : 'Y'}</div><div className="message-content"><div className="message-author">{message.role === 'assistant' ? 'Research persona' : 'You'}{message.turn && <span>{message.turn.provider === 'evidence' ? 'Source notes' : 'Generated'}</span>}</div>{message.text && <p className="message-text">{message.text}</p>}{message.pending && <div className="thinking"><LoaderCircle size={13} className="spin"/>{status || 'Working on your question'}</div>}{message.error && <div className="message-error" role="alert">{message.error}</div>}{message.turn && <div className="message-tools"><button onClick={() => setEvidence(message.turn!.citations)}><BookOpen size={13}/>{message.turn.citations.length ? `${message.turn.citations.length} evidence records` : 'No supporting evidence'}</button><button onClick={() => listen(message.turn!)} disabled={busy} aria-label="Play response"><Volume2 size={14}/><span>Listen</span></button>{message.turn.grounding.status === 'insufficient' && <span className="insufficient-label">Limited evidence</span>}{message.turn.grounding.status === 'unverified' && <span className="insufficient-label">Unverified answer</span>}</div>}</div></article>)}
                <div ref={chatBottom}/>
              </div>
              {(playback.error || microphone.error) && <div className="inline-error" role="alert">{playback.error || microphone.error}</div>}
              <form className={`composer ${microphone.listening ? 'listening' : ''}`} onSubmit={(event) => { event.preventDefault(); void send(); }}><textarea ref={composer} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={microphone.listening ? 'Listening… you can edit before sending' : language === 'ko' ? '궁금한 것을 물어봐…' : 'Ask something you’re curious about…'} aria-label="Your message" rows={2} maxLength={2000} disabled={busy} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); } }}/><div className="composer-bottom"><button className={`microphone-button ${microphone.listening ? 'active' : ''}`} type="button" onClick={() => { playback.stop(); microphone.toggle(); }} disabled={busy || !microphone.supported} aria-label={microphone.listening ? 'Stop microphone' : 'Use microphone'} title={microphone.supported ? 'Dictate a question' : 'Microphone transcription is unavailable in this browser'}><Mic size={16}/><span>{microphone.listening ? 'Listening' : 'Use your voice'}</span></button><span className="keyboard-hint">↵ to send</span>{busy ? <button type="button" className="send-button" onClick={cancel} aria-label="Stop response"><Square size={15}/></button> : <button type="submit" className="send-button" disabled={!draft.trim() || connection !== 'ready'} aria-label="Send message"><ArrowUp size={18}/></button>}</div></form>
              <div className="conversation-footer"><span className="tiny-dot"/> AI simulation · Historical sources · Verify important claims</div>
            </section>
          </div>
          <section className="session-strip"><div className="session-evidence"><div className="strip-icon"><Layers3 size={20}/></div><div><span className="eyebrow">GROUNDED IN PUBLIC MATERIAL</span><strong>{persona?.stats.sources ?? '—'} sources <i/> {persona?.stats.evidence ?? '—'} evidence records</strong></div><button onClick={() => navigate('sources')} aria-label="Explore source library"><ArrowRight size={18}/></button></div><div className="session-model"><div><span className="eyebrow">CONVERSATION APPROACH</span><label><select value={mode} onChange={(event) => setMode(event.target.value as ExperimentMode)} disabled={busy} aria-label="Conversation approach">{experiments.map((experiment) => <option key={experiment.id} value={experiment.id}>{experiment.letter} · {experiment.name}</option>)}</select><ChevronDown size={14}/></label></div><button className="text-button" onClick={() => navigate('lab')}>Compare <ArrowUpRight size={14}/></button></div></section>
          <div className="studio-footnote"><span><Radio size={13}/> {activeTurn ? `${Math.round(activeTurn.metrics.total_ms)} ms server response · designed behavior cues` : 'Designed motion · reference avatar · source-grounded conversation'}</span><button onClick={() => setInfoOpen(!infoOpen)}>How this works <ArrowDown size={12}/></button></div>
        </>}
      </main>
      <nav className="mobile-nav" aria-label="Mobile navigation"><button className={page === 'studio' ? 'active' : ''} onClick={() => navigate('studio')}><MessageCircle size={19}/>Studio</button><button className={page === 'sources' ? 'active' : ''} onClick={() => navigate('sources')}><BookOpen size={19}/>Sources</button><button className={page === 'lab' ? 'active' : ''} onClick={() => navigate('lab')}><FlaskConical size={19}/>Lab</button></nav>
    </div>
    {evidence && <><button className="drawer-scrim" onClick={() => setEvidence(null)} aria-label="Close response evidence"/><EvidencePanel citations={evidence} onClose={() => setEvidence(null)}/></>}
    <SettingsDialog open={settings} close={() => setSettings(false)} persona={persona} voice={voice} setVoice={(next) => { playback.stop(); setVoice(next); }} autoPlay={autoPlay} setAutoPlay={setAutoPlay} reducedMotion={reducedMotion} setReducedMotion={setReducedMotion} modelName={model?.name} loadModel={loadModel} resetModel={() => { setModel(undefined); setAvatarError(null); }}/>
  </div>;
}

