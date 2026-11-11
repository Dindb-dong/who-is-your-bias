import type { Grounding, Language } from '../api';

/** Marker coverage remains separate from whether the source supports a claim. */
export function GroundingBadge({ grounding, language = 'en' }: { grounding: Grounding; language?: Language }) {
  const korean = language === 'ko';
  const curated = grounding.semantic_status === 'curated_source';
  const semanticLabel = grounding.status === 'insufficient'
    ? (korean ? '근거 부족' : 'Limited evidence')
    : curated ? (korean ? '검토된 자료 미리보기' : 'Reviewed source preview')
      : (korean ? '내용 검토 전' : 'Facts not reviewed');
  const count = grounding.cited_unit_count !== undefined && grounding.citation_unit_count !== undefined
    ? `${grounding.cited_unit_count}/${grounding.citation_unit_count}` : undefined;
  const status = grounding.citation_status;
  let coverageLabel: string | undefined;
  if (status === 'complete' || status === 'partial') coverageLabel = korean ? `출처 표기${count ? ` ${count}` : ''}` : `Citation coverage${count ? ` ${count}` : ''}`;
  else if (status === 'missing') coverageLabel = korean ? '출처 표기 누락' : 'Missing citation markers';
  else if (status === 'invalid') coverageLabel = korean ? '잘못된 출처 표기' : 'Invalid citation markers';
  const explanation = curated
    ? (korean ? '검토된 공개 자료의 요약을 표시합니다.' : 'This preview displays reviewed summaries of public sources.')
    : (korean ? '출처 표기는 문장 단위로 확인합니다. 출처가 답변 내용을 뒷받침하는지는 별도 검토가 필요합니다.' : 'Coverage counts sentence-like units with source markers. Whether those sources support the answer still needs review.');
  return <span className="grounding-badge" data-semantic-status={grounding.semantic_status ?? 'unreviewed'} title={explanation}>
    <span className={`grounding-label ${curated ? 'curated' : 'unreviewed'}`}>{semanticLabel}</span>
    {coverageLabel && <span className={`citation-coverage ${status}`} data-citation-status={status}>{coverageLabel}</span>}
  </span>;
}
