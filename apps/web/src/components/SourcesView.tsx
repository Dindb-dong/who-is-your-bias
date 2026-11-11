import { useMemo, useState } from 'react';
import { ArrowUpRight, Search, BookOpen, ShieldCheck } from 'lucide-react';
import type { Persona, Source } from '../api';

export function SourcesView({ sources, corpusInfo }: { sources: Source[]; corpusInfo?: Persona['qa_corpus'] }) {
  const [query, setQuery] = useState('');
  const [split, setSplit] = useState<'all' | 'train' | 'test'>('all');
  const filtered = useMemo(() => sources.filter((source) => (split === 'all' || (split === 'test' ? source.split !== 'train' : source.split === split)) && `${source.title} ${source.publisher}`.toLowerCase().includes(query.toLowerCase())), [sources, query, split]);
  return <section className="sources-page">
    <div className="section-heading"><div><span className="eyebrow">THE RESEARCH BEHIND THE PERSONA</span><h1>Every idea has a source.</h1><p>First-person interviews and authored articles. Dates and context stay attached.</p></div><div className="large-stat">{sources.length}<span>primary sources</span></div></div>
    {corpusInfo && corpusInfo.question_answer_records > 0 && <div className="research-note"><BookOpen size={19}/><p><strong>{corpusInfo.materialization.includes('bounded') ? 'Bounded excerpt pilot' : 'Original interview Q&A'}</strong> · {corpusInfo.training_question_answers} exchanges available to retrieval. The selected corpus contains {corpusInfo.complete_exchanges} complete exchanges and {corpusInfo.excerpts} excerpts, including withheld evaluation material.</p></div>}
    <div className="source-toolbar"><label className="search-field"><Search size={17}/><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a source" aria-label="Find a source"/></label><div className="segmented">{(['all', 'train', 'test'] as const).map((value) => <button key={value} className={split === value ? 'active' : ''} onClick={() => setSplit(value)}>{value === 'all' ? 'All sources' : value === 'train' ? 'In conversation' : 'Held out'}</button>)}</div></div>
    <div className="source-list">{filtered.map((source, index) => <article key={source.id} className="source-card"><div className="source-number">{String(index + 1).padStart(2, '0')}</div><div className="source-main"><div className="source-meta"><span>{source.publisher}</span><span>{new Date(`${source.published_at}T00:00:00Z`).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' })}</span></div><h2>{source.title}</h2><p>{source.attribution}</p><span className={`tag ${source.split !== 'train' ? 'amber' : ''}`}>{source.split !== 'train' ? <ShieldCheck size={12}/> : <BookOpen size={12}/>} {source.split !== 'train' ? (source.split === 'validation' ? 'Development holdout' : 'Test holdout') : 'Available to retrieval'}</span></div><a className="icon-button source-link" href={source.url} target="_blank" rel="noreferrer" aria-label={`Open ${source.title}`}><ArrowUpRight size={20}/></a></article>)}</div>
    {!filtered.length && <div className="empty-state"><Search size={24}/><h3>No matching sources</h3><p>Try a different title or publisher.</p></div>}
    <div className="research-note"><ShieldCheck size={19}/><p><strong>Source-level separation.</strong> Held-out interviews are excluded from the conversation corpus. Original interview excerpts retain their question and answer relationship. Development and test events are excluded from retrieval. Source coverage and identity likeness are separate questions.</p></div>
  </section>;
}

