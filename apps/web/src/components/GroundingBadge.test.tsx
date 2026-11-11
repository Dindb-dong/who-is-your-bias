import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { createEventParser } from '../api';
import type { Grounding, StreamEvent, Turn } from '../api';
import { GroundingBadge } from './GroundingBadge';

const grounding: Grounding = { status: 'unverified', retrieved_ids: ['source-a'], invalid_citation_ids: [], citation_status: 'complete', citation_unit_count: 2, cited_unit_count: 2, malformed_citations: [], semantic_status: 'unreviewed' };
const turn: Turn = {
  id: 'turn', session_id: 'session', text: 'Generated reply.', language: 'en', mode: 'rag', provider: 'openai', citations: [],
  behavior: { duration_ms: 1000, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 200 } }, grounding,
  metrics: { retrieval_ms: 1, generation_ms: 2, total_ms: 3 },
};
function parseGrounding(value: unknown, provider: Turn['provider'] = 'openai') {
  const events: StreamEvent[] = [];
  const parser = createEventParser((event) => events.push(event));
  parser.push(`event: result\ndata: ${JSON.stringify({ turn: { ...turn, provider, grounding: value } })}\n\n`);
  return events;
}

describe('citation coverage and source review labels', () => {
  it('never presents complete citation markers as verified facts', () => {
    const html = renderToStaticMarkup(<GroundingBadge grounding={grounding}/>);
    expect(html).toContain('Citation coverage 2/2');
    expect(html).toContain('Facts not reviewed');
    expect(html).not.toContain('Verified answer');
    expect(html).toContain('Whether those sources support');
  });
  it('shows Korean missing markers separately from semantic review', () => {
    const html = renderToStaticMarkup(<GroundingBadge grounding={{ ...grounding, citation_status: 'missing', cited_unit_count: 0 }} language="ko"/>);
    expect(html).toContain('내용 검토 전');
    expect(html).toContain('출처 표기 누락');
  });
  it('distinguishes curated previews, limited evidence, and legacy responses', () => {
    expect(renderToStaticMarkup(<GroundingBadge grounding={{ ...grounding, status: 'grounded', citation_status: 'not_required', semantic_status: 'curated_source' }}/>)).toContain('Reviewed source preview');
    expect(renderToStaticMarkup(<GroundingBadge grounding={{ status: 'insufficient', retrieved_ids: [], invalid_citation_ids: [] }}/>)).toContain('Limited evidence');
    expect(renderToStaticMarkup(<GroundingBadge grounding={{ status: 'grounded', retrieved_ids: ['source-a'], invalid_citation_ids: [] }}/>)).toContain('Facts not reviewed');
  });
  it('accepts additive diagnostics and legacy server payloads', () => {
    expect(parseGrounding(grounding)).toHaveLength(1);
    expect(parseGrounding({ status: 'unverified', retrieved_ids: [], invalid_citation_ids: [] })).toHaveLength(1);
  });
  it.each([
    { citation_status: 'entailed' }, { semantic_status: 'verified' }, { citation_unit_count: -1 },
    { citation_unit_count: 1.5 }, { cited_unit_count: 3 }, { malformed_citations: [42] },
    { reason: 17 }, { semantic_status: 'curated_source' },
  ])('rejects invalid diagnostic payloads: %j', (changes) => {
    expect(() => parseGrounding({ ...grounding, ...changes })).toThrow('Invalid result');
  });
});
