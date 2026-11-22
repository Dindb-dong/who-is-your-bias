"""Reviewed translations of exact, fixed QA limitations.

These are authored UI/guard sentences, not quotations or persona memories.
Exact matching avoids a second model changing 'unverified' into 'false'.
Unknown generated sentences still use the configured translation service.
"""

REVIEWED_KOREAN = {
    "I can't make a personal buy-or-sell recommendation or promise a stock's future return.":
        "개인에게 주식을 사거나 팔라고 권하거나 미래 수익을 보장할 수는 없습니다.",
    "A sensible discussion would compare current filings, valuation and concentration risk with your time horizon; this demo does not have verified live market data.":
        "최신 공시, 기업 가치와 집중 투자 위험을 투자 기간과 함께 검토해야 하며, 이 시연에는 검증된 실시간 시장 데이터가 없습니다.",
    "I can't verify that allegation from the sources available here, and I shouldn't repeat a personal rumor as fact.":
        "여기서 이용할 수 있는 자료로는 그 의혹을 확인할 수 없으며, 개인에 관한 소문을 사실처럼 반복하지 않겠습니다.",
    "I can discuss a specific, reliable public statement if one is provided and reviewed.":
        "구체적이고 신뢰할 만한 공개 입장이 제공되고 검토되면 그 내용을 설명할 수 있습니다.",
    "I can't verify that current claim from the sources available to this demo.":
        "이 시연에서 이용할 수 있는 자료만으로는 해당 최신 정보를 확인할 수 없습니다.",
    "I can discuss dated public statements, but I shouldn't present them as today's news or a confirmed forecast.":
        "과거의 공개 발언을 설명할 수는 있지만, 오늘의 뉴스나 확정된 전망으로 제시하지는 않겠습니다.",
    "I don't have reviewed project documentation for that detail.":
        "그 세부 사항을 확인할 검토된 프로젝트 자료가 없습니다.",
    "The team should confirm the implementation or measurement before I describe it as a result.":
        "구현 내용이나 측정 결과는 팀이 확인한 뒤 설명하겠습니다.",
    "The reviewed code documents model configuration, but it does not establish the team's model-selection rationale or superiority in a controlled comparison.":
        "검토한 코드에는 모델 설정 방법이 있지만, 팀의 모델 선정 이유나 비교 실험에서의 우위는 확인되지 않습니다.",
    "The reviewed code has separate preparation paths for text, voice references and visual assets.":
        "검토한 코드에는 텍스트, 참조 음성, 시각 자료를 각각 준비하는 경로가 있습니다.",
    "These records do not verify a jointly reviewed, utterance-aligned text, audio and face dataset, so I cannot claim a full MCCCD reproduction.":
        "발화별 텍스트·음성·얼굴을 함께 검증한 데이터셋은 이 자료에서 확인되지 않으므로, MCCCD 전체 파이프라인을 재현했다고 주장할 수 없습니다.",
}
