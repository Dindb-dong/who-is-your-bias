from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import batch, dataset, demo, experiments, handoff
from pipeline.common import PipelineError, boolean, digest, load_env, read_csv, read_json, read_jsonl, write_json

PROJECT = Path(__file__).resolve().parent
load_env(PROJECT / ".env")
st.set_page_config(page_title="Who's your bias? · 데이터 작업실", page_icon="🎙", layout="wide")
st.title("Who's your bias?")
st.caption("26/2학기 YAICON · 영상에서 검수 데이터와 아바타 입력까지")
mode = st.sidebar.radio("작업 자료", ["실제 자료", "가상 예제"])
root = Path(os.environ.get("YAICON_DATA_DIR", str(PROJECT / "data"))) / ("local" if mode == "실제 자료" else "demo")
root.mkdir(parents=True, exist_ok=True)
step = st.sidebar.radio("진행 단계", ["시작", "1. 자료 입력", "2. 전사·검수", "3. 데이터셋", "4. 답변 실험", "5. 음성·인계", "Aside 일괄 수집"])
st.sidebar.caption("자료는 이 컴퓨터에 저장됩니다. API 실행 버튼을 누를 때 해당 입력이 지정한 서비스로 전달됩니다.")
registered = dataset.sources(root)


def success(path):
    st.success("저장했습니다.")
    st.code(str(path), language=None)


def save_upload(upload, folder, prefix):
    if not upload:
        return None
    path = Path(folder) / (prefix + Path(upload.name).suffix.lower())
    path.write_bytes(upload.getbuffer())
    return path


def choose_source():
    if not registered:
        st.info("먼저 자료를 등록하세요.")
        return None, None
    by_id = {source["source_id"]: source for source in registered}
    sid = st.selectbox("출처", list(by_id), format_func=lambda key: (
        f"{by_id[key]['title']} [{key}]" if by_id[key].get("title") else key
    ))
    path = dataset.source_dir(root, sid)
    st.caption(f"저장 폴더 · {path.relative_to(PROJECT) if path.is_relative_to(PROJECT) else path}")
    return path, by_id[sid]


try:
    if step == "시작":
        st.markdown("**자료 입력 → 전사·검수 → 데이터셋 → 답변 실험 → 음성·인계**")
        st.write("준비한 영상과 자막을 등록하고, 누가 어떤 질문에 답했는지 확인한 뒤 실험에 사용합니다.")
        st.write("처음에는 왼쪽에서 ‘가상 예제’를 선택해 저장된 결과를 살펴보세요. 예제 음성은 말소리가 아닌 테스트 신호입니다.")
        a, b, c = st.columns(3)
        a.metric("등록 출처", len(registered))
        b.metric("검색용 촬영분", len({s["recording_id"] for s in registered if s["split"] == "train"}))
        c.metric("평가용 촬영분", len({s["recording_id"] for s in registered if s["split"] == "eval"}))
        if mode == "가상 예제" and not registered:
            if st.button("가상 예제로 전체 흐름 만들기", type="primary"):
                with st.spinner("가상 자료로 실행 중"):
                    st.json(demo.create(root))
        elif (root / "demo_result.json").exists():
            st.info("이 예제의 문답은 가상이며 젠슨 황의 실제 발언이 아닙니다.")
            st.json(read_json(root / "demo_result.json"))
        st.markdown("사용 안내는 프로젝트 폴더의 **사용가이드.md**, 계획은 **Plan.md**에 있습니다.")

    elif step == "1. 자료 입력":
        if mode == "가상 예제":
            st.info("실제 파일은 왼쪽 ‘실제 자료’에서 등록하세요. 예제와 섞지 않습니다.")
        else:
            st.write("영상과 자막을 함께 넣으면 바로 검수할 수 있습니다. 영상만 넣으면 다음 단계에서 자동 전사를 실행합니다.")
            with st.form("ingest"):
                left, right = st.columns(2)
                sid = left.text_input("출처 ID", value="source-001", help="파일마다 다른 영문·숫자 ID")
                rid = right.text_input("원본 촬영분 ID", value="recording-001", help="같은 방송의 편집본은 같은 ID")
                split = left.selectbox("용도", ["train", "eval"], format_func=lambda x: "검색·말투 자료" if x == "train" else "평가 전용")
                language = right.selectbox("원문 언어", ["en", "ko", "zh", "ja"])
                uploaded = st.file_uploader("영상 또는 음성", type=sorted(s.lstrip(".") for s in dataset.media.MEDIA_SUFFIXES))
                local = st.text_input("또는 이 컴퓨터의 영상 파일 경로", help="큰 영상은 업로드 대신 절대 경로 입력")
                subtitles = st.file_uploader("자막 (선택)", type=["srt", "vtt", "json"])
                url = st.text_input("유튜브 영상 URL 또는 출처 URL (선택)", help="로컬 파일이 없을 때만 다운로드. 로그인 제한 영상은 로컬 파일 사용")
                publisher = st.text_input("원본 게시 채널·기관 (선택)", help="파일로 등록한 자료를 시연의 출처로 표시할 때 필요합니다.")
                published = st.date_input("원본 게시일 (선택)", value=None, format="YYYY-MM-DD")
                notes = st.text_area("출처·이용 범위·작업 메모")
                submitted = st.form_submit_button("자료 등록", type="primary")
            if submitted:
                with tempfile.TemporaryDirectory(dir=root) as temp, st.spinner("자료를 등록하는 중"):
                    media_path = save_upload(uploaded, temp, "media") or (local.strip() or None)
                    subtitle_path = save_upload(subtitles, temp, "subtitle")
                    display_upload = uploaded or (subtitles if not local.strip() else None)
                    title = Path(display_upload.name).stem if display_upload else None
                    result = dataset.register(root, sid, rid, split, "jensen-huang", language,
                                              media_path, subtitle_path, url.strip(), notes, title=title,
                                              publisher=publisher.strip(), published_at=published)
                success(result)

    elif step == "Aside 일괄 수집":
        if mode == "가상 예제":
            st.info("왼쪽에서 ‘실제 자료’를 선택하면 수집할 수 있습니다.")
        else:
            st.caption(f"저장 위치 · {root / 'sources'}")
            with st.form("aside_search"):
                query = st.text_input("유튜브 검색어", value="Jensen Huang NVIDIA interview Q&A")
                channel = st.text_input("채널 이름 (선택)", help="검색 결과에서 입력한 채널 이름과 일치하는 항목만 표시")
                limit = st.number_input("후보 수", min_value=1, max_value=20, value=5)
                search_clicked = st.form_submit_button("Aside로 검색")
            if search_clicked:
                with st.spinner("Aside에서 후보를 찾는 중"):
                    st.session_state["aside_hits"] = batch.search(query, limit, channel)
                st.session_state.pop("aside_selected", None)
                if not st.session_state["aside_hits"]:
                    st.info("검색 결과가 없습니다. 검색어나 채널 이름을 조정하세요.")
            hits = {hit["videoId"]: hit for hit in st.session_state.get("aside_hits", [])}
            selected = st.multiselect("처리할 영상 후보", list(hits), key="aside_selected",
                                     format_func=lambda vid: hits[vid]["title"] + " · " + hits[vid].get("channelName", ""))
            if hits:
                st.dataframe([{"영상": hit["title"], "채널": hit.get("channelName"),
                               "URL": batch.URL + vid, "길이(초)": hit.get("durationSeconds"),
                               "게시일": hit.get("publishDate")} for vid, hit in hits.items()], hide_index=True)
            links = st.text_area("직접 넣을 유튜브 URL (한 줄에 하나)")
            language = st.selectbox("자막·전사 언어", ["en", "ko", "zh", "ja"])
            keep = st.checkbox("원본 영상 또는 음성도 확보", help="참조 음성을 만들 때 선택하세요. 미선택 시 자막을 우선합니다.")
            use_llm = st.checkbox("LLM으로 화자·문답·태그 초벌 제안", disabled=not batch.configured_llm(),
                                  help="설정한 모델로 전사를 보내며 호출 비용이 발생할 수 있습니다.")
            if not batch.configured_llm():
                st.caption("LLM 초벌 제안은 .env의 LLM_BASE_URL과 LLM_MODEL을 설정한 뒤 사용할 수 있습니다.")
            st.caption("수집 결과는 검색용(train)으로 등록합니다. 검수 화면에서 화자·시간·문답을 확인하세요.")
            active = None
            if st.button("선택한 영상 일괄 처리", type="primary"):
                entries = [(batch.URL + vid, hits[vid]["title"]) for vid in selected]
                entries += [(line.strip(), "") for line in links.splitlines() if line.strip()]
                active = batch.new_job(root, entries, language, keep, use_llm)
            jobs = sorted((root / "batches").glob("batch-*.json"), reverse=True)
            if jobs:
                if active:
                    st.session_state["batch_history"] = active
                job_path = st.selectbox("일괄 작업 기록", jobs, format_func=lambda p: p.stem, key="batch_history")
                if st.button("미완료 항목 이어서 처리"):
                    active = job_path
                if active:
                    status = st.empty()
                    def show_progress(i, total, item):
                        status.info(f"{i}/{total} 처리 중 · {item['title'] or item['video_id']}")
                    batch.run_job(root, active, show_progress)
                    status.success("실행을 마쳤습니다. 아래에서 영상별 결과를 확인하세요.")
                report = read_json(active or job_path)
                st.dataframe(report["items"], hide_index=True)
                st.caption("done 항목은 ‘2. 전사·검수’에서 열 수 있습니다. failed 항목은 오류를 확인한 뒤 이어서 처리하세요.")

    elif step == "2. 전사·검수":
        path, source = choose_source()
        if path:
            with st.expander("시연 출처 정보", expanded=not all(source.get(k) for k in ("url", "channel", "upload_date"))):
                source_url = st.text_input("원본 URL", source.get("url") or "", key=f"source-url-{source['source_id']}")
                source_publisher = st.text_input("게시 채널·기관", source.get("channel") or "", key=f"source-publisher-{source['source_id']}")
                source_date = st.date_input("게시일", date.fromisoformat(source["upload_date"]) if source.get("upload_date") else None,
                                            format="YYYY-MM-DD", key=f"source-date-{source['source_id']}")
                if st.button("시연 출처 정보 저장"):
                    success(dataset.save_source_attribution(root, source["source_id"], source_url.strip(), source_publisher.strip(), source_date))
            if source["media_file"]:
                with st.expander("원본 재생", expanded=False):
                    media_path = path / source["media_file"]
                    if media_path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
                        st.video(str(media_path))
                    else:
                        st.audio(str(media_path))
            if not (path / "review.csv").exists():
                st.info("자막이 없습니다. 첫 전사 시 선택한 모델 파일을 내려받습니다. CPU로도 실행할 수 있습니다.")
                model = st.selectbox("전사 모델", ["small", "base", "tiny", "medium", "large-v3"])
                device = st.selectbox("실행 장치", ["cpu", "cuda"], help="학교 NVIDIA GPU가 연결된 환경에서만 cuda")
                if st.button("자동 전사 시작"):
                    with st.spinner("자동 전사 중 · 영상 길이와 장치에 따라 시간이 걸립니다"):
                        dataset.transcribe_source(root, source["source_id"], model, device, source["language"])
                    st.rerun()
            else:
                st.markdown("같은 문답의 질문과 답변에 **동일한 turn_id**를 입력하세요. 질문 화자는 `interviewer`, 젠슨 황은 `target`입니다. 강연·독백은 `kind=style`로 둡니다.")
                st.caption("원음을 들으며 텍스트와 시간을 고친 뒤 reviewed를 체크합니다. 단독 화자·잡음·전사 일치까지 확인한 음성에만 voice_clean을 체크하세요. situation은 질문 이전 맥락에서 정합니다.")
                rows = read_csv(path / "review.csv")
                for row in rows:
                    row["start"], row["end"] = float(row["start"]), float(row["end"])
                    row["reviewed"], row["voice_clean"] = boolean(row["reviewed"]), boolean(row["voice_clean"])
                edited = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", hide_index=True,
                    key="review-" + source["source_id"] + digest(path / "review.csv")[:10],
                    column_config={"speaker": st.column_config.SelectboxColumn(options=["unknown", "target", "interviewer", "other"]),
                                   "kind": st.column_config.SelectboxColumn(options=["qa", "style", "exclude"]),
                                   "reviewed": st.column_config.CheckboxColumn(), "voice_clean": st.column_config.CheckboxColumn()})
                if st.button("검수표 저장", type="primary"):
                    success(dataset.save_review(root, source["source_id"], edited.fillna("").to_dict("records"),
                                                reviewer="Workbench reviewer"))
                st.download_button("검수표 CSV 내려받기", (path / "review.csv").read_bytes(), source["source_id"] + "-review.csv")
                with st.expander("엑셀 등에서 편집한 CSV 가져오기"):
                    upload = st.file_uploader("검수 CSV", type=["csv"])
                    if st.button("CSV 적용") and upload:
                        with tempfile.TemporaryDirectory(dir=root) as temp:
                            file = save_upload(upload, temp, "review")
                            success(dataset.save_review(root, source["source_id"], read_csv(file),
                                                        reviewer="Workbench CSV reviewer"))

    elif step == "3. 데이터셋":
        st.write("검수 완료한 문답을 모읍니다. 같은 촬영분의 검색/평가 혼입과 동일 답변 중복을 검사합니다.")
        if registered:
            st.dataframe(pd.DataFrame(registered)[["source_id", "recording_id", "split", "language"]], hide_index=True)
        if st.button("데이터셋 만들기", type="primary"):
            with st.spinner("원음 길이와 검수 자료를 확인하는 중"):
                success(dataset.build(root))
        if (root / "latest_dataset.json").exists():
            data = Path(read_json(root / "latest_dataset.json")["path"])
            st.json(read_json(data / "build_report.json"))
            st.caption("검수표를 수정한 뒤 새 데이터셋을 만드세요. 로컬페르소나시연.command를 다시 실행하면 최신 데이터셋이 시연에 반영됩니다.")
            for name in ["train.jsonl", "eval.jsonl", "style.jsonl", "reviews.json"]:
                if (data / name).exists():
                    st.download_button(name + " 내려받기", (data / name).read_bytes(), name)
            runtime_file = PROJECT / ".tools/local-runtime.json"
            if mode == "실제 자료" and runtime_file.exists():
                latest_run = Path(read_json(runtime_file)["output"])
                if (latest_run / "integration.json").exists():
                    integration = read_json(latest_run / "integration.json")
                    if "imports" in integration:
                        with st.expander("최근 로컬 시연에 반영된 자료"):
                            if integration["workbench_dataset"] != data.name:
                                st.info("최신 데이터셋을 시연에 반영하려면 시연을 다시 실행하세요.")
                            st.json({k: integration[k] for k in ("imports", "skipped", "pr19_restore")})

    elif step == "4. 답변 실험":
        datasets = sorted((root / "datasets").glob("dataset-*"), reverse=True)
        if not datasets:
            st.info("먼저 데이터셋을 만드세요.")
        else:
            data = st.selectbox("사용할 데이터셋", datasets, format_func=lambda p: p.name)
            target = "synthetic-persona" if mode == "가상 예제" else "jensen-huang"
            default_card = experiments.persona_card(target, "Synthetic Example" if mode == "가상 예제" else "Jensen Huang")
            card = root / "persona.json"
            st.markdown("**A** 인물 카드 · **B** 카드+주제 검색 · **C** 카드+주제·상황 검색")
            st.caption("현재 검색은 단어 겹침에 기반한 비교 기준선입니다. 모델 연결 전에도 검색 예시와 프롬프트를 확인할 수 있습니다.")
            with st.expander("공통 인물 카드 편집", expanded=False):
                card_text = st.text_area("검수 근거가 있는 말투·사실만 추가", value=json.dumps(read_json(card) if card.exists() else default_card, ensure_ascii=False, indent=2), height=250)
            question = st.text_area("실험 질문", help="비워두면 평가용 인터뷰의 질문 전체를 사용합니다.")
            situation = st.text_input("질문의 상황", help="예: advice, criticism, reflection. 답변의 반응을 미리 적지 마세요.")
            language = st.selectbox("답변 언어", ["en", "ko"])
            k = st.slider("B/C가 참고할 문답 수", 1, 5, 3)
            if st.button("A/B/C 실험 입력 만들기", type="primary"):
                parsed = json.loads(card_text)
                write_json(card, parsed)
                success(experiments.prepare(data, card, question, situation, language, k))
            runs = sorted((root / "experiments").glob("abc-*"), reverse=True)
            if runs:
                run = st.selectbox("실험 기록", runs, format_func=lambda p: p.name)
                requests = read_jsonl(run / "requests.jsonl")
                st.dataframe(pd.DataFrame([{k: r[k] for k in ["request_id", "condition", "example_count", "example_chars"]} for r in requests]), hide_index=True)
                with st.expander("실제 LLM 입력과 검색 점수 보기"):
                    st.json(requests)
                st.download_button("프롬프트 묶음 내려받기", (run / "requests.jsonl").read_bytes(), "requests.jsonl")
                st.caption(".env에 모델 주소·모델명·키를 설정한 경우에만 아래 버튼으로 호출합니다. 조건별로 API 사용료가 발생할 수 있습니다.")
                if st.button("설정한 LLM으로 답변 생성"):
                    with st.spinner("A/B/C 답변 생성 중"):
                        result = experiments.run_llm(run)
                    success(result)
                    st.json(read_json(result / "run.json"))
                response_files = sorted(run.glob("responses-*/responses.jsonl"), reverse=True)
                if response_files:
                    st.dataframe(pd.DataFrame(read_jsonl(response_files[0])), hide_index=True)
                    ratings = response_files[0].parent / "blind_ratings.csv"
                    st.download_button("조건을 가린 평가표", ratings.read_bytes(), "blind_ratings.csv")

    elif step == "5. 음성·인계":
        st.write("참조 음성을 먼저 준비한 뒤 답변 텍스트와 묶습니다. ElevenLabs는 기존 voice_id를 사용하며, 이 화면에서 새 목소리를 등록하지 않습니다.")
        with st.expander("검수한 참조 음성 추출"):
            path, source = choose_source()
            if path and (path / "review.csv").exists():
                options = [r["segment_id"] for r in read_csv(path / "review.csv")
                           if boolean(r["reviewed"]) and boolean(r["voice_clean"]) and r["speaker"] == "target" and r["kind"] != "exclude"]
                if options:
                    segment = st.selectbox("단독 음성으로 확인한 구간", options)
                    if st.button("참조 WAV와 전사 추출"):
                        success(dataset.reference(root, source["source_id"], segment))
                else:
                    st.info("검수 화면에서 대상 발화의 reviewed와 voice_clean을 확인하세요.")
        references = sorted((root / "references").glob("*/reference.json"), reverse=True)
        ref = st.selectbox("사용할 음성 참조", [None, *references], format_func=lambda p: "없음 (ElevenLabs voice_id만 사용 가능)" if p is None else p.parent.name)
        if ref:
            st.audio(str(ref.parent / "reference.wav"))
            st.write(read_json(ref)["reference_text"])
        responses = []
        for file in sorted((root / "experiments").glob("*/responses-*/responses.jsonl"), reverse=True):
            responses.extend(r for r in read_jsonl(file) if r.get("status") == "ok")
        selected = st.selectbox("생성된 답변 불러오기", [None, *range(len(responses))], format_func=lambda i: "직접 입력" if i is None else responses[i]["request_id"] + " · " + responses[i]["answer"][:40])
        text = st.text_area("말할 답변 (읽고 수정한 뒤 인계)", value="" if selected is None else responses[selected]["answer"], key=f"speech-{selected}", height=150)
        language = st.selectbox("음성 언어", ["en", "ko"])
        if st.button("동욱에게 넘길 묶음 만들기", type="primary"):
            success(handoff.package(root, text, language, "synthetic-persona" if mode == "가상 예제" else "jensen-huang", ref))
        packages = sorted((root / "handoff").glob("turn-*"), reverse=True)
        if packages:
            pack = st.selectbox("인계 묶음", packages, format_func=lambda p: p.name)
            turn = read_json(pack / "avatar_turn.json")
            st.json(turn)
            st.caption("묶음 폴더 전체를 공유하세요. audio_file이 비어 있으면 아직 영상 재생용 음성이 없는 상태입니다.")
            if turn.get("audio_file"):
                st.audio(str(pack / turn["audio_file"]))
            else:
                if st.button("ElevenLabs로 음성 생성 (API 호출)"):
                    with st.spinner("음성을 생성하는 중"):
                        success(handoff.elevenlabs(pack))
                audio = st.file_uploader("또는 Qwen/CosyVoice 등에서 생성한 음성 연결", type=["wav", "mp3", "flac", "m4a"])
                if st.button("생성 음성 연결") and audio:
                    with tempfile.TemporaryDirectory(dir=root) as temp:
                        success(handoff.attach_audio(pack, save_upload(audio, temp, "speech")))
except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
    st.error(str(exc))
