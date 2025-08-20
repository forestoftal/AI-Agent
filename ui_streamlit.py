# ui_streamlit.py
import os, io, uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app import (
    # 과제 파이프라인
    step_show_preview, step_row_count, step_active_inactive_count,
    step_pure_cid_lists, step_manual_drop_cids,
    # RAG & 업로드 색인 & 서술형
    ensure_lectures_indexed, answer_with_rag, generate_essay_questions,
    index_uploaded_pdf,
    # 일반 Chat
    chat_general
)

load_dotenv()

# -------------------- 페이지 & 스타일 --------------------
st.set_page_config(page_title="화합물빅데이터 | AI 조교", layout="wide")


st.title("🧪 화합물빅데이터 · AI 조교 (Streamlit)")

# -------------------- 세션 태그 (업로드 PDF 전용) --------------------
if "session_tag" not in st.session_state:
    st.session_state.session_tag = uuid.uuid4().hex[:8]

# -------------------- 사이드바: 데이터 선택 --------------------
st.sidebar.header("📁 데이터 선택")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

uploaded = st.sidebar.file_uploader("CSV 업로드 (또는 아래에서 data/ 파일 선택)", type=["csv"])
local_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
chosen = st.sidebar.selectbox("data/ 내 파일 선택", ["(선택 안함)"] + local_files)

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.path = ""

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded)
    st.session_state.path = uploaded.name
elif chosen != "(선택 안함)":
    st.session_state.df = pd.read_csv(os.path.join(DATA_DIR, chosen))
    st.session_state.path = os.path.join(DATA_DIR, chosen)

# -------------------- 탭 구성 --------------------
tab1, tab2 = st.tabs(["🧭 과제 & 실습 도우미", "📚 강의자료 RAG & 서술형 연습"])

# ==================== 탭 1: 과제 & 실습 도우미 ====================
with tab1:
    st.subheader("과제 워크플로우")
    st.caption("버튼을 누를 때마다 해당 단계의 **코드와 출력**이 함께 표시됩니다. 다음 단계 권장도 함께 표시됩니다.")

    if st.session_state.df is None:
        st.info("좌측에서 CSV를 업로드하거나 data/ 폴더의 파일을 선택하세요.")
    else:
        df = st.session_state.df

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("① 데이터 미리보기 (head)"):
                head, code = step_show_preview(df, n=5)
                st.markdown("#### 코드")
                st.code(code, language="python")
                st.markdown("#### 출력 (상위 5행)")
                st.dataframe(head, use_container_width=True)

        with c2:
            if st.button("② 전체 Row 개수 세기"):
                out, code = step_row_count(df)
                st.markdown("#### 코드")
                st.code(code, language="python")
                st.markdown("#### 출력")
                st.write(out)
                st.success("다음: ③ Active/Inactive 개수 세기를 권장합니다.")

        with c3:
            if st.button("③ Active/Inactive 개수 세기"):
                try:
                    out, code = step_active_inactive_count(df)
                    st.markdown("#### 코드")
                    st.code(code, language="python")
                    st.markdown("#### 출력")
                    st.write(out)
                    st.success("다음: ④ 순수 Active/Inactive CID 목록 추출을 권장합니다.")
                except Exception as e:
                    st.error(str(e))

        st.markdown("---")

        if st.button("④ 순수 'Active' / 'Inactive' CID 목록 추출"):
            try:
                out, code = step_pure_cid_lists(df)
                st.markdown("#### 코드")
                st.code(code, language="python")
                st.markdown("#### 출력 (앞 20개 미리보기)")
                st.write({
                    "active_cids": out["active_cids"][:20],
                    "inactive_cids": out["inactive_cids"][:20],
                    "cid_column": out["cid_column"]
                })
                # 다운로드
                a_buf = io.StringIO("\n".join(map(str, out["active_cids"])))
                i_buf = io.StringIO("\n".join(map(str, out["inactive_cids"])))
                st.download_button("Active CID 리스트 다운로드", a_buf.getvalue(), file_name="active_cids.txt")
                st.download_button("Inactive CID 리스트 다운로드", i_buf.getvalue(), file_name="inactive_cids.txt")
                st.success("다음: ⑤ 전처리(특정 CID 제거)를 권장합니다.")
            except Exception as e:
                st.error(str(e))

        st.markdown("**⑤ 전처리: 특정 CID 제거**")
        default_drop = "28127,28145"
        drop_text = st.text_input("제거할 CID(콤마 구분)", value=default_drop)
        if st.button("제거 실행"):
            try:
                drop_list = [int(x.strip()) for x in drop_text.split(",") if x.strip()]
                df2, code, removed = step_manual_drop_cids(df, drop_list)
                st.markdown("#### 코드")
                st.code(code, language="python")
                st.markdown(f"#### 제거된 행 수: **{removed}**")
                st.dataframe(df2.head(5), use_container_width=True)
                if st.checkbox("이 전처리 결과를 현재 데이터로 확정"):
                    st.session_state.df = df2
                    st.success("현재 세션의 데이터가 전처리된 테이블로 업데이트되었습니다.")
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.caption("💡 팁: 위 단계별 코드는 보고서에 그대로 인용 가능. 필요하면 PCA/스케일링/시각화/학습·평가 단계도 버튼으로 확장해 줄 수 있어요.")

# ==================== 탭 2: 강의자료 RAG & 서술형 연습 ====================
with tab2:
    st.subheader("강의자료 기반 질의응답 (RAG)")
    st.caption("처음 한 번은 강의자료 색인을 수행합니다. 강의 PDF는 `data/lectures/` 폴더에 넣어두거나, 아래 업로드 기능을 이용하세요.")

    # 1) 기본 강의자료 색인/조회
    try:
        all_topics = ensure_lectures_indexed()  # 신규만 색인 + 전체 토픽 조회
    except Exception as e:
        all_topics = []
        st.warning(f"강의자료 색인 중 문제가 발생했습니다: {e}")

    # 2) 세션 전용 업로드 → 즉시 색인
    st.markdown("### 📤 강의자료 업로드(세션 전용)")
    st.caption(f"이 브라우저 세션에만 적용되는 임시 색인입니다. 세션 태그: `{st.session_state.session_tag}`")
    up_pdf = st.file_uploader("새로운 PDF 업로드(여러 개 가능)", type=["pdf"], accept_multiple_files=True)
    if up_pdf:
        for f in up_pdf:
            try:
                topic_added = index_uploaded_pdf(f.getvalue(), f.name, st.session_state.session_tag)
                st.success(f"업로드/색인 완료: {f.name} → topic: {topic_added} (sess:{st.session_state.session_tag})")
                if topic_added not in all_topics:
                    all_topics.append(topic_added)
            except Exception as e:
                st.error(f"{f.name} 색인 중 오류: {e}")

    # 3) 토픽 선택 + 질문
    if not all_topics:
        st.info("강의자료가 없거나 색인이 안 되어 있습니다. `data/lectures/`에 PDF를 넣거나 위에서 업로드하세요.")
    else:
        all_topics = sorted(all_topics)
        chosen_topics = st.multiselect("질의에 사용할 강의 토픽 선택", all_topics, default=all_topics)

        q = st.text_input("질문을 입력하세요", placeholder="예: 'SMILES 표기법의 장단점을 설명하라' 또는 'ML과 DL의 차이'")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("RAG로 답변하기"):
                if not q.strip():
                    st.warning("질문을 입력하세요.")
                else:
                    try:
                        ans, ctx = answer_with_rag(
                            q,
                            topics=chosen_topics,
                            session_tag=st.session_state.session_tag  # 업로드 문서 포함
                        )
                        st.markdown("#### 답변")
                        st.write(ans)
                        with st.expander("🔎 참고 컨텍스트"):
                            for i,c in enumerate(ctx,1):
                                st.markdown(f"**[{i}] {c['where']}**")
                                st.write(c["text"])
                    except Exception as e:
                        st.error(f"RAG 질의 중 오류: {e}")

        with col2:
            if st.button("일반 Chat (컨텍스트 없이)"):
                if not q.strip():
                    st.warning("질문을 입력하세요.")
                else:
                    try:
                        st.markdown("#### 답변")
                        st.write(chat_general(q))
                    except Exception as e:
                        st.error(f"Chat 중 오류: {e}")

        st.markdown("---")
        st.subheader("📝 서술형 연습문제 생성")
        st.caption("선택한 토픽의 컨텍스트를 바탕으로 서술형 문제를 자동 생성합니다.")
        n = st.number_input("토픽당 문제 수", 1, 10, 5)
        difficulty = st.selectbox("난이도", ["쉬움","중간","어려움"], index=1)
        if st.button("문제 생성"):
            try:
                items, md_text, csv_bytes = generate_essay_questions(
                    chosen_topics, n_per_topic=int(n), difficulty=difficulty
                )
                st.markdown("#### 미리보기 (Markdown)")
                st.markdown(md_text)
                st.download_button("CSV로 다운로드", data=csv_bytes, file_name="essay_questions.csv", mime="text/csv")
                st.download_button("Markdown로 다운로드", data=md_text, file_name="essay_questions.md")
            except Exception as e:
                st.error(f"문제 생성 중 오류: {e}")

    st.divider()
    st.caption("📂 강의자료 기본 위치: `data/lectures/`  |  예: chemical_structure.pdf, nomenclature.pdf, smiles.pdf, ml.pdf")
