# Directory: studyflow_agent

# === main.py ===
import os
from dotenv import load_dotenv
from studyflow_agent.extract import load_and_split, summarize_chunks
from studyflow_agent.flashcard_gen import generate_flashcards
from studyflow_agent.formatter import parse_to_qa, save_as_csv
from studyflow_agent.planner import create_review_schedule

# === Load API Key from .env ===
load_dotenv("studyflow_agent/.env")

# === File Paths ===
pdf_path = "studyflow_agent/molecularbio.pdf"
summary_output = "studyflow_agent/output/summaries.txt"
flashcard_output = "studyflow_agent/output/flashcards.csv"
schedule_output = "studyflow_agent/output/review_schedule.ics"

# Create output folder if not exists
os.makedirs("studyflow_agent/output", exist_ok=True)

print("[1] Loading and splitting PDF...")
chunks = load_and_split(pdf_path)

print("[2] Summarizing content...")
summaries = summarize_chunks(chunks)
with open(summary_output, 'w', encoding='utf-8') as f:
    f.write("\n---\n".join(summaries))

print("[3] Generating flashcards...")
raw_flashcards = generate_flashcards(summaries)

print("[4] Parsing flashcards...")
all_flashcards = []
for raw in raw_flashcards:
    all_flashcards.extend(parse_to_qa(raw))
save_as_csv(all_flashcards, flashcard_output)

print("[5] Creating review schedule...")
create_review_schedule(all_flashcards, start_date_str="2025-07-16", output_file=schedule_output)

print("\u2705 Done. Flashcards and schedule saved in output/.")


# === extract.py ===
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def summarize_chunks(chunks):
    import os
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    prompt = PromptTemplate.from_template(
        "Extract key points and definitions from this text:\n\n{input}"
    )
    chain = prompt | llm
    return [chain.invoke({"input": chunk.page_content}) for chunk in chunks]


# === flashcard_gen.py ===
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os

def generate_flashcards(summary_list):
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    prompt = PromptTemplate.from_template(
        "Create 3–5 flashcards (Q&A) based on this summary:\n\n{input}\n\nFormat:\nQ: ...\nA: ..."
    )
    chain = prompt | llm
    return [chain.invoke({"input": s}) for s in summary_list]


# === formatter.py ===
import csv
import re

def parse_to_qa(raw_text):
    flashcards = []
    qa_pairs = re.findall(r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)', raw_text, re.DOTALL)
    for q, a in qa_pairs:
        flashcards.append({"question": q.strip(), "answer": a.strip()})
    return flashcards

def save_as_csv(flashcards, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answer'])
        writer.writeheader()
        for card in flashcards:
            writer.writerow(card)


# === planner.py ===
from ics import Calendar, Event
from datetime import datetime, timedelta

def create_review_schedule(flashcards, start_date_str, output_file):
    cal = Calendar()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    per_day = 10
    for i in range(0, len(flashcards), per_day):
        day = start_date + timedelta(days=i // per_day)
        event = Event()
        event.name = f"Review Flashcards {i+1}-{min(i+per_day, len(flashcards))}"
        event.begin = day.strftime("%Y-%m-%d")
        cal.events.add(event)
    with open(output_file, 'w') as f:
        f.writelines(cal.serialize_iter())


# === memory_store.py ===
import json
import os

def save_memory(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_memory(file):
    if not os.path.exists(file):
        return {}
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


# === utils.py ===
def log_step(message):
    print(f"[LOG] {message}")
