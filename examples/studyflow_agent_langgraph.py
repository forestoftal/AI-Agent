

from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ics import Calendar, Event
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os, re, csv


load_dotenv()

# Step Functions

def load_and_split_pdf(state):
    pdf_path = state["pdf_path"]
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return {**state, "chunks": splitter.split_documents(docs)}

def summarize_chunks(state):
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    prompt = PromptTemplate.from_template("Extract key points and definitions from this text:\n\n{input}")
    chain = prompt | llm

    summaries = []
    for chunk in state["chunks"]:
        output = chain.invoke({"input": chunk.page_content})
        summaries.append(output.content if hasattr(output, "content") else str(output))

    return {**state, "summaries": summaries}

def check_summary(state):
    if not state.get("summaries"):
        return {**state, "__condition": "retry"}
    return {**state, "__condition": "ok"}

def generate_flashcards(state):
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    prompt = PromptTemplate.from_template(
        "Create 3–5 flashcards (Q&A) based on this summary:\n\n{input}\n\nFormat:\nQ: ...\nA: ..."
    )
    chain = prompt | llm

    summaries = state["summaries"]
    if isinstance(summaries, str):
        summaries = [summaries]

    flashcards = [chain.invoke({"input": s}) for s in summaries]
    return {**state, "raw_flashcards": flashcards}

def parse_flashcards(state):
    raw_list = state["raw_flashcards"]
    all_cards = []
    for raw in raw_list:
        text = raw.content if hasattr(raw, "content") else str(raw)
        qa_pairs = re.findall(r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)', text, re.DOTALL)
        for q, a in qa_pairs:
            all_cards.append({"question": q.strip(), "answer": a.strip()})
    return {**state, "flashcards": all_cards}

def check_flashcard_count(state):
    if len(state.get("flashcards", [])) < 3:
        return {**state, "__condition": "regenerate"}
    return {**state, "__condition": "next"}

def create_schedule(state):
    flashcards = state.get("flashcards", [])
    start_date = state.get("start_date")
    if not start_date:
        raise KeyError("Missing 'start_date' in state")

    cal = Calendar()
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    per_day = 10
    for i in range(0, len(flashcards), per_day):
        day = start_date + timedelta(days=i // per_day)
        event = Event()
        event.name = f"Review Flashcards {i+1}-{min(i+per_day, len(flashcards))}"
        event.begin = day.strftime("%Y-%m-%d")
        cal.events.add(event)
    os.makedirs("output_langgraph", exist_ok=True)
    with open("output_langgraph/review_schedule_langgraph.ics", 'w') as f:
        f.writelines(cal.serialize_iter())
    return {**state}

def save_outputs(state):
    os.makedirs("output_langgraph", exist_ok=True)
    with open("output_langgraph/summaries_langgraph.txt", 'w', encoding='utf-8') as f:
        f.write("\n---\n".join(state["summaries"]))
    with open("output_langgraph/flashcards_langgraph.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answer'])
        writer.writeheader()
        for card in state["flashcards"]:
            writer.writerow(card)
    return {}

# Graph Construction
graph = StateGraph(state_schema=dict)

graph.add_node("load", load_and_split_pdf)
graph.add_node("summarize", summarize_chunks)
graph.add_node("check_summary", check_summary)
graph.add_node("generate_flashcard", generate_flashcards)
graph.add_node("parse_flashcard", parse_flashcards)
graph.add_node("check_count", check_flashcard_count)
graph.add_node("schedule", create_schedule)
graph.add_node("save", save_outputs)

#  Edges (flow control)
graph.set_entry_point("load")
graph.add_edge("load", "summarize")
graph.add_edge("summarize", "check_summary")
graph.add_conditional_edges(
    "check_summary",
    lambda s: s.get("__condition", "ok"),
    {
        "retry": "summarize",
        "ok": "generate_flashcard"
    }
)
graph.add_edge("generate_flashcard", "parse_flashcard")
graph.add_edge("parse_flashcard", "check_count")
graph.add_conditional_edges(
    "check_count",
    lambda s: s["__condition"],
    {
        "regenerate": "generate_flashcard",
        "next": "schedule"
    }
)
graph.add_edge("schedule", "save")

# Compile and Run
app = graph.compile()

app.invoke({
    "pdf_path": "molecularbio.pdf",
    "start_date": "2025-07-16"
})
