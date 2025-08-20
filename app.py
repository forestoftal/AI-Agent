# app.py
# ---------------------------------------------
# "화합물빅데이터" 과제 도우미 + RAG + 서술형 문제 생성 백엔드
# 업로드 PDF 즉시 색인(세션 전용) 지원
# ---------------------------------------------
import os, re, io, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv

# -------- OpenAI ----------
load_dotenv()
from openai import OpenAI
oai = OpenAI()   # OPENAI_API_KEY는 .env에

# ================== 과제 파이프라인 (순수 pandas) ==================
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols: return cols[name.lower()]
    canon = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in df.columns}
    for name in candidates:
        key = re.sub(r'[^a-z0-9]', '', name.lower())
        for k,v in canon.items():
            if key in k: return v
    return None

def _detect_activity_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, [
        "Activity","activity","Outcome","outcome","Bioactivity","bioactivity",
        "AID result","assay_outcome","ActiveInactive","activity_flag","class"
    ])

def _detect_cid_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["CID","cid","Molecule CID","pubchem_cid","PUBCHEM_CID","compound_cid"])

def step_show_preview(df: pd.DataFrame, n=5) -> Tuple[pd.DataFrame, str]:
    code = f"""import pandas as pd
df = pd.read_csv("PATH_TO_YOUR_FILE.csv")
df.head({n})"""
    return df.head(n), code

def step_row_count(df: pd.DataFrame) -> Tuple[Dict[str,int], str]:
    code = "row_count = len(df)\nrow_count"
    return {"rows": len(df)}, code

def step_active_inactive_count(df: pd.DataFrame) -> Tuple[Dict[str,int], str]:
    col = _detect_activity_col(df)
    if not col:
        raise ValueError("활성/비활성(Activity) 컬럼을 찾지 못했습니다.")
    s = df[col].astype(str).str.strip().str.lower()
    code = f'''col = "{col}"
s = df[col].astype(str).str.strip().str.lower()
active_cnt = (s=="active").sum()
inactive_cnt = (s=="inactive").sum()
active_cnt, inactive_cnt'''
    return {"active": int((s=="active").sum()),
            "inactive": int((s=="inactive").sum()),
            "column": col}, code

def step_pure_cid_lists(df: pd.DataFrame) -> Tuple[Dict[str,List[int]], str]:
    cid = _detect_cid_col(df); col = _detect_activity_col(df)
    if not cid or not col: raise ValueError("CID 또는 Activity 컬럼을 찾지 못했습니다.")
    s = df[col].astype(str).str.strip().str.lower()
    act = df.loc[s=="active", cid].dropna().astype(int).unique().tolist()
    ina = df.loc[s=="inactive", cid].dropna().astype(int).unique().tolist()
    code = f'''cid_col = "{cid}"; act_col = "{col}"
s = df[act_col].astype(str).str.strip().str.lower()
pure_active_cids = (df.loc[s=="active", cid_col].dropna().astype(int).unique().tolist())
pure_inactive_cids = (df.loc[s=="inactive", cid_col].dropna().astype(int).unique().tolist())
pure_active_cids[:10], pure_inactive_cids[:10]'''
    return {"active_cids": act, "inactive_cids": ina, "cid_column": cid}, code

def step_manual_drop_cids(df: pd.DataFrame, drop_list: List[int]) -> Tuple[pd.DataFrame, str, int]:
    cid = _detect_cid_col(df)
    if not cid: raise ValueError("CID 컬럼을 찾지 못했습니다.")
    before = len(df)
    df2 = df[~df[cid].astype(str).isin([str(x) for x in drop_list])].copy()
    removed = before - len(df2)
    code = f'''drop_list = {drop_list}
cid_col = "{cid}"
df = df[~df[cid_col].astype(str).isin([str(x) for x in drop_list])].copy()
len(df)'''
    return df2, code, int(removed)

# ================== RAG (강의자료 PDF 색인/검색) ==================
# 최소 의존성: LangChain 로더만 사용, 벡터DB는 Chroma 직접 사용
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb

LECTURE_DIR = os.path.join("data", "lectures")
CHROMA_DIR  = "chroma_db"
COLLECTION  = "course_docs"

def _topic_from_path(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def index_lectures(lecture_dir: str = LECTURE_DIR) -> List[str]:
    """data/lectures/의 PDF/TXT를 분할→임베딩→Chroma에 저장.
       메타: topic, source, page, file_hash 저장."""
    os.makedirs(lecture_dir, exist_ok=True)
    files = [os.path.join(lecture_dir, f) for f in os.listdir(lecture_dir)
             if f.lower().endswith((".pdf",".txt",".md"))]
    if not files:
        return []

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION not in [c.name for c in client.list_collections()]:
        coll = client.create_collection(COLLECTION)
    else:
        coll = client.get_collection(COLLECTION)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    added_topics = set()
    for path in files:
        topic = _topic_from_path(path)
        added_topics.add(topic)
        fhash = _file_hash(path)
        existing = coll.get(where={"file_hash": fhash})
        if existing and existing.get("ids"):
            continue  # 이미 색인됨

        # 로드 → 청크
        if path.lower().endswith(".pdf"):
            docs = PyPDFLoader(path).load()
        else:
            docs = TextLoader(path, encoding="utf-8").load()
        chunks = splitter.split_documents(docs)

        texts = [c.page_content for c in chunks]
        metas = []
        for c in chunks:
            metas.append({
                "source": c.metadata.get("source", path),
                "page": c.metadata.get("page", None),
                "topic": topic,
                "file_hash": fhash
            })
        vecs = embedder.embed_documents(texts)
        ids = [f"{topic}-{fhash}-{i}" for i in range(len(texts))]
        coll.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)

    return sorted(added_topics)

def ensure_lectures_indexed() -> List[str]:
    """강의자료가 색인되어 있지 않으면 색인한다. 반환: 전체 토픽 리스트"""
    os.makedirs(LECTURE_DIR, exist_ok=True)
    _ = index_lectures(LECTURE_DIR)  # 신규만 추가
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_collection(COLLECTION)
    all_docs = coll.get(limit=1_000_000, include=["metadatas"])
    all_topics = sorted({m.get("topic","") for m in all_docs.get("metadatas",[]) if m.get("topic")})
    return all_topics

def rag_search(query: str, top_k: int = 4,
               topics: Optional[List[str]] = None,
               session_tag: Optional[str] = None) -> List[Dict[str,Any]]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_collection(COLLECTION)
    qvec = OpenAIEmbeddings(model="text-embedding-3-small").embed_query(query)

    # where 필터 구성: 선택 토픽 / 세션 업로드
    ors = []
    if topics:
        ors.extend([{"topic": t} for t in topics])
    if session_tag:
        ors.append({"session": session_tag})

    if not ors:
        where = None
    elif len(ors) == 1:
        where = ors[0]   # 그냥 dict 하나만 주기
    else:
        where = {"$or": ors}


    res = coll.query(query_embeddings=[qvec], n_results=top_k, where=where)
    hits = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        hits.append({"text": doc,
                     "where": f"{meta.get('topic','')} | {meta.get('source','')} p.{meta.get('page')} (sess:{meta.get('session','-')})"})
    return hits

def answer_with_rag(question: str,
                    topics: Optional[List[str]] = None,
                    session_tag: Optional[str] = None,
                    temperature: float = 0.2) -> Tuple[str, List[Dict[str,Any]]]:
    hits = rag_search(question, top_k=4, topics=topics, session_tag=session_tag)
    ctx = "\n---\n".join([f"[{h['where']}]\n{h['text']}" for h in hits]) if hits else "(컨텍스트 없음)"
    system = "너는 '화합물빅데이터' 과목 조교봇이다. 반드시 주어진 컨텍스트 안에서, 출처를 근거로 설명하라."
    user = f"컨텍스트:\n{ctx}\n\n질문: {question}\n규칙: 자료에 없는 내용은 '자료에 명시 없음'이라고 답해라."
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=temperature
    )
    return resp.choices[0].message.content, hits

# ================== 업로드 PDF 즉시 색인(세션 전용) ==================
def index_uploaded_pdf(file_bytes: bytes, filename: str, session_tag: str) -> str:
    """
    업로드(PDF)를 임시 저장 후 해당 파일만 색인.
    메타데이터에 session_tag를 넣어 현재 세션에서만 필터링되도록 함.
    """
    os.makedirs("tmp_uploads", exist_ok=True)
    tmp_dir = Path("tmp_uploads") / session_tag
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fpath = tmp_dir / filename
    with open(fpath, "wb") as f:
        f.write(file_bytes)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION not in [c.name for c in client.list_collections()]:
        coll = client.create_collection(COLLECTION)
    else:
        coll = client.get_collection(COLLECTION)

    # 로드 & 청크
    docs = PyPDFLoader(str(fpath)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c.page_content for c in chunks]
    topic = _topic_from_path(filename)
    fhash = _file_hash(str(fpath))
    metas = []
    for c in chunks:
        metas.append({
            "source": c.metadata.get("source", str(fpath)),
            "page": c.metadata.get("page", None),
            "topic": topic,
            "file_hash": fhash,
            "session": session_tag,   # ★ 세션 태그
        })
    vecs = embedder.embed_documents(texts)
    ids = [f"{topic}-{session_tag}-{i}" for i in range(len(texts))]
    coll.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)
    return topic

# ================== 서술형 문제 자동 생성 ==================
def generate_essay_questions(topics: List[str], n_per_topic: int = 5, difficulty: str = "중간") -> Tuple[List[Dict[str,Any]], str, bytes]:
    """선택 토픽 컨텍스트로 서술형 문제 생성.
       반환: 문제 리스트, Markdown 문자열, CSV bytes"""
    # 각 토픽에서 대표 청크를 몇 개씩 모음
    seed_context = []
    for t in topics:
        seeds = rag_search(f"{t} 핵심 개념 요약", top_k=3, topics=[t])
        for s in seeds:
            seed_context.append(f"[{s['where']}]\n{s['text']}")
    if not seed_context:
        raise RuntimeError("선택한 토픽에서 컨텍스트를 찾지 못했습니다. 강의자료가 색인되었는지 확인하세요.")

    prompt = f"""너는 대학 '화합물빅데이터' 과목 조교다.
아래 컨텍스트를 참고하여 각 토픽별로 서술형 문제를 만들어라.
- 난이도: {difficulty}
- 각 토픽당 {n_per_topic}문제
- 문제는 포괄적 개념 정리, 비교 설명, 절차/근거 서술형으로 구성
- 해답 포인트(채점 기준 핵심 키워드 3~6개)를 함께 제시

컨텍스트:
{'\n---\n'.join(seed_context)}

출력 형식(JSON lines):
{{"topic": "<topic>", "question": "<문제>", "answer_points": ["키워드1","키워드2",...]}}
"""
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"정확하고 간결한 한국어 서술형 문제 메이커"},
                  {"role":"user","content":prompt}],
        temperature=0.6
    ).choices[0].message.content

    # 라인 단위 JSON 파싱
    items: List[Dict[str,Any]] = []
    for line in resp.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("{") and line.endswith("}"):
            try:
                import json
                obj = json.loads(line)
                items.append({
                    "topic": obj.get("topic",""),
                    "question": obj.get("question",""),
                    "answer_points": obj.get("answer_points",[])
                })
            except Exception:
                continue

    # Markdown & CSV 생성
    md_buf = io.StringIO()
    md_buf.write(f"# 서술형 연습문제 ({difficulty})\n\n")
    by_topic: Dict[str,List[Dict[str,Any]]] = {}
    for it in items:
        by_topic.setdefault(it["topic"], []).append(it)

    for t, qs in by_topic.items():
        md_buf.write(f"## {t}\n")
        for i,q in enumerate(qs,1):
            md_buf.write(f"{i}. {q['question']}\n")
            if q.get("answer_points"):
                md_buf.write(f"   - 해답 포인트: {', '.join(q['answer_points'])}\n")
        md_buf.write("\n")

    csv_buf = io.StringIO()
    csv_buf.write("topic,question,answer_points\n")
    for it in items:
        ap = "; ".join(it.get("answer_points", []))
        def esc(s: str) -> str:
            s = s.replace('"','""')
            return f'"{s}"'
        csv_buf.write(f"{esc(it['topic'])},{esc(it['question'])},{esc(ap)}\n")

    return items, md_buf.getvalue(), csv_buf.getvalue().encode("utf-8")

# ================== 일반 Chat (RAG 없이) ==================
def chat_general(question: str) -> str:
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"너는 '화합물빅데이터' 과목 조교다. 간결하고 정확하게 답하라."},
            {"role":"user","content":question}
        ],
        temperature=0.4
    )
    return resp.choices[0].message.content
