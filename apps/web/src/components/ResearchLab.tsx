import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, FlaskConical, Play, Square } from 'lucide-react';
import { streamChat } from '../api';
import type { ExperimentMode, Language, Turn } from '../api';
import { GroundingBadge } from './GroundingBadge';

export const experiments: { id: ExperimentMode; letter: string; name: string; description: string }[] = [
  { id: 'profile', letter: 'A', name: 'Profile only', description: 'A persona card with no retrieved evidence.' },
  { id: 'rag', letter: 'B', name: 'Evidence grounded', description: 'The profile plus relevant source records.' },
  { id: 'behavior', letter: 'C', name: 'Context & behavior', description: 'Evidence plus situation-aware presentation.' },
];
type Comparison = { text: string; turn?: Turn; error?: string };

export function ResearchLab({ language }: { language: Language }) {
  const [prompt, setPrompt] = useState('Why did Tesla start with an expensive sports car?');
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<Partial<Record<ExperimentMode, Comparison>>>({});
  const pending = useRef<AbortController | null>(null);
  const ticket = useRef(0);
  useEffect(() => () => { ticket.current += 1; pending.current?.abort(); }, []);
  async function compare() {
    if (!prompt.trim() || running) return;
    const requestTicket = ++ticket.current;
    const controller = new AbortController(); pending.current = controller;
    setResults({}); setRunning(true);
    for (const { id } of experiments) {
      if (controller.signal.aborted || requestTicket !== ticket.current) break;
      try {
        await streamChat({ message: prompt.trim(), mode: id, language }, controller.signal, (event) => {
          if (requestTicket !== ticket.current) return;
          if (event.event === 'token') setResults((previous) => ({ ...previous, [id]: { text: (previous[id]?.text ?? '') + event.data.text } }));
          if (event.event === 'result') setResults((previous) => ({ ...previous, [id]: { text: event.data.turn.text, turn: event.data.turn } }));
        });
      } catch (error) {
        if (!controller.signal.aborted && requestTicket === ticket.current) setResults((previous) => ({ ...previous, [id]: { text: previous[id]?.text ?? '', error: error instanceof Error ? error.message : 'Comparison failed.' } }));
      }
    }
    if (requestTicket === ticket.current) { setRunning(false); pending.current = null; }
  }
  function cancel() { ticket.current += 1; pending.current?.abort(); pending.current = null; setRunning(false); }
  return <section className="lab-page">
    <div className="section-heading"><div><span className="eyebrow">A SMALL EXPERIMENT, A CLEARER ANSWER</span><h1>What makes a persona?</h1><p>Ask the same question across three configurations. Compare the actual responses.</p></div><FlaskConical size={42} strokeWidth={1.1}/></div>
    <div className="experiment-input"><label htmlFor="comparison-question">Your comparison question</label><textarea id="comparison-question" value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={2000} rows={2}/><button className="primary-button" onClick={() => running ? cancel() : void compare()} disabled={!prompt.trim()}>{running ? <Square size={15}/> : <Play size={15}/>} {running ? 'Stop comparison' : 'Run comparison'}</button></div>
    <div className="comparison-grid">{experiments.map((experiment) => { const result = results[experiment.id]; return <article key={experiment.id} className="comparison-card"><div className="experiment-letter">{experiment.letter}</div><h2>{experiment.name}</h2><p className="muted">{experiment.description}</p><div className="comparison-response" aria-live="polite">{result?.text || <span className="placeholder">{running ? 'Waiting for the response…' : 'Run the experiment to see a response.'}</span>}{result?.error && <p className="error-text" role="alert">{result.error}</p>}</div>{result?.turn && <><div className="comparison-metrics"><span>{result.turn.provider === 'evidence' ? 'Source preview' : 'LLM generation'}</span><span>{result.turn.citations.length} sources cited</span><span>{Math.round(result.turn.metrics.total_ms)} ms server</span></div><GroundingBadge grounding={result.turn.grounding} language={language}/><div className="comparison-citations">{result.turn.citations.map((citation) => <a key={citation.evidence_id} href={citation.url} target="_blank" rel="noreferrer">{citation.title}<ArrowUpRight size={13}/></a>)}</div></>}</article>; })}</div>
    <div className="research-note"><FlaskConical size={19}/><p><strong>Compare evidence, not an invented score.</strong> No-key previews show how each configuration handles source evidence. Persona fidelity needs a configured model and a separate human evaluation. Server time excludes speech generation and playback.</p></div>
  </section>;
}

