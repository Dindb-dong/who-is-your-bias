# 데이터 형식과 검수 기준

## 데이터가 만들어지는 곳

```text
data/local/
  sources/<영상 제목> [<source_id>]/
    source.json                 출처·촬영분·용도·원본 해시
    media.mp4 또는 media.wav     입력 파일 복사본
    subtitle.srt/vtt/json        제공한 자막 원본
    transcript.raw.json          자막 해석 또는 ASR 원본
    review.csv                  사람이 고치는 검수표
    review_history/             저장 전 검수본
    audio.wav                   필요할 때 만든 24 kHz mono 음성
  batches/                      일괄 수집 진행 기록
  datasets/dataset-<실행ID>/
    train.jsonl                 검색용 검수 문답
    eval.jsonl                  평가용 검수 문답
    style.jsonl                 검색용 독백·강연 (문답과 별도)
    style.eval.jsonl            평가용 독백·강연
    sources.json                출처 및 검수 파일 해시 스냅샷
    build_report.json           문답 수·누락·중복·경고
  experiments/                  실험별 프롬프트·답변·평가표
  references/                   참조 음성 및 정확한 전사
  handoff/                      영상 담당자에게 보낼 턴 폴더
```

`source_id`는 한 파일/영상의 ID, `recording_id`는 실제 촬영분의 ID다. 재업로드·하이라이트도 같은 방송이면 recording_id를 통일한다. 자동으로 원 방송을 추론하지 않는다.

## 검수표에서 고칠 것

| 필드 | 입력 방법 |
| --- | --- |
| segment_id | 구간마다 고유한 ID. 새 행은 seg-00003처럼 중복 없이 지정 |
| start, end | 원본 시작을 0으로 둔 초 단위 시각. 시작 < 종료 |
| text | 실제 들리는 원문. 말끝·반복·추임새를 의미 없이 삭제하지 않음 |
| speaker | target=대상 인물, interviewer=질문자, other=그 외, unknown=미확인 |
| turn_id | 같은 질문과 그 답변에 같은 값. 예: q001 |
| kind | qa=문답, style=독백/강연, exclude=제외 |
| topic | 질문 주제. 예: AI, leadership, failure |
| situation | 질문과 앞선 맥락에서 알 수 있는 상황. 예: advice, criticism, reflection |
| context | 질문 이해에 필요한 앞선 상황 |
| translation_ko | 선택 번역. 원문 text를 덮어쓰지 않음 |
| reviewed | 원문·화자·시간·문답 연결을 확인했으면 true |
| voice_clean | 실제 들어보고 대상 단독 발화, 낮은 배경음, 전사 일치까지 확인했으면 true |
| notes | 불명확한 부분, 오류, 작업 메모 |

질문과 답변이 여러 자막 행으로 나뉘면 같은 turn_id를 붙인다. 질문자가 다시 질문하기 시작하면 새 turn_id를 쓴다. 한 행에 두 화자가 섞이면 행과 시간 구간을 나눈다. 겹쳐 말하는 구간은 참조 음성으로 지정하지 않는다.

유튜브 자동 자막은 앞부분을 반복하는 롤링 자막이 있을 수 있다. 원본은 그대로 보존한다. 검수표에서 내용을 한 번만 남기고, 중복 행은 exclude로 표시한다. 실제 말의 반복을 자동 삭제하지 않기 위해 초벌 단계에서 무조건 중복 제거하지 않는다.

## 내보내기 규칙

- 같은 turn_id에서 제외하지 않은 모든 구간이 검수되어야 한다. 미검수 그룹은 건너뛰고 보고서에 남긴다.
- qa는 질문자와 대상 인물의 답변을 모두 포함해야 한다. 질문이 없는 강연에는 질문을 만들어 붙이지 않고 style로 둔다.
- 질문과 답변이 겹치거나 순서가 잘못되면 수정할 때까지 내보내지 않는다. 대상 발화끼리 겹친 롤링 자막도 검수 대상이다.
- 시간은 원음 길이 안에 있어야 한다. 자막만 등록했다면 원음과 대조할 수 없으므로 시간 형식만 검사한다.
- 같은 recording_id는 train/eval 중 한쪽에만 존재한다. 원본 파일 해시와 동일 답변을 추가로 검사한다.
- 원문이 완전히 같은 답변은 같은 split에서는 하나만 남기고, 다른 split에 걸치면 중단한다. 짧고 흔한 답변까지 중복으로 판정할 수 있으므로 파일럿에서 중복 규칙을 조정할 수 있다. 의미상 중복·유사 편집본은 사람이 확인한다.
- style.eval은 보존하지만 첫 A/B/C는 문답 eval만 사용한다.
- 카드의 말투 근거 evidence_ids는 train 문답 또는 train 독백 ID만 허용한다. 사실은 text와 train source_id를 기록한다. 카드의 문장이 실제 근거와 일치하는지는 사람이 검수한다.
- 검수표 변경 후 **데이터셋 만들기**를 다시 누르면 새 버전이 생긴다. 과거 실험은 과거 데이터셋을 유지한다.

## 문답 JSONL 예시

아래는 형식 설명용 가상 데이터다.

```json
{
  "schema_version": "1.0",
  "dialogue_id": "source-001__q001",
  "source_id": "source-001",
  "recording_id": "interview-001",
  "target_id": "jensen-huang",
  "split": "train",
  "kind": "qa",
  "language": "en",
  "synthetic": true,
  "question": "How should I start a project?",
  "answer_verbatim": "Start with one small test.",
  "translation_ko": "작은 실험 하나로 시작하세요.",
  "topic": "project",
  "situation": "advice",
  "context": "A student asks for practical advice.",
  "search_text": "How should I start a project? project",
  "start": 10.0,
  "end": 18.0,
  "answer_start": 14.0,
  "answer_end": 18.0,
  "segment_ids": ["seg-00001", "seg-00002"],
  "review_status": "approved"
}
```

검색용 답변은 train, 실제 평가 정답은 eval로 분리한다. 원문과 번역도 별도로 유지한다.

## A/B/C에서 통제하는 것과 한계

동일 인물 카드·사실·질문·출력 언어를 사용한다. B와 C는 같은 예시 수, 같은 질문/답변 글자 상한을 사용하고 실제 글자 수를 기록한다. C는 `0.7 × 주제 점수 + 0.3 × 상황 점수`로 정렬한다. 가중치는 연구로 확정된 값이 아닌 첫 구현값이다.

검색은 TF-IDF 가중치를 적용한 단어 빈도 코사인 유사도다. 한국어 형태소 분석이나 다국어 임베딩은 없고, 영어 자료에 한국어 질문을 바로 검색하면 잘 찾지 못할 수 있다. 원문 영어 질문으로 검색하고 `output_language=ko`로 생성하는 실험은 가능하다. 실제 답변 언어를 바꾸는 것은 화법 재현을 추가로 검증해야 하는 별도 조건이다.

예시 글자 상한을 맞춰도 실제 토큰 수는 다르다. 정식 발표 실험에서는 선택한 모델의 토큰 수를 점검하고, 예시 수·길이 차이를 보고한다. 상황 태그는 평가 정답의 반응을 보고 붙이지 않는다. 모든 검색 점수가 0인 질문은 검색 실패로 분류해 분석한다.
