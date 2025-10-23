import os
import json
import re
import asyncio
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Agentic RAG Demo", layout="wide")
st.title("📊 Financial Agentic RAG Demo")

# =========================
# Sidebar
# =========================
st.sidebar.header("Tools")
use_calculator = st.sidebar.checkbox("Use Finance Calculator", value=True)

query = st.text_area("Enter your query:", "")

# =========================
# Simple finance calculator
# =========================
def finance_calculator(query: str):
    try:
        expr = re.findall(r"[\d\.\+\-\*\/\(\) ]+", query)
        if expr:
            expr = "".join(expr)
            result = eval(expr)
            return f"Result: {result}"
    except Exception:
        return "Could not compute arithmetic expression."
    return None

# =========================
# Chroma DB setup
# =========================
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "/tmp/chroma_db")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

async def prepare_chroma_db_async():
    if not os.listdir(CHROMA_DB_DIR):  # only if empty
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        docs = [Document(page_content=t) for t in chunks]
        embeddings = get_embeddings()
        db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR)
        db.persist()
        return db
    else:
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=get_embeddings())

# =========================
# Cached embeddings and QA chain
# =========================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_qa_chain():
    embeddings = get_embeddings()
    # Initialize Chroma DB
    try:
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    except ValueError:
        db = Chroma(embedding_function=embeddings, persist_directory=None)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Load LLM
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        device=-1  # CPU-friendly
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    return qa_chain

# =========================
# Async query execution
# =========================
async def run_query_async(qa_chain, query_text):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, qa_chain.run, query_text)
    return result

# =========================
# Main UI
# =========================
if st.button("Run Agent") and query:
    # 1️⃣ Finance calculator
    if use_calculator:
        calc_result = finance_calculator(query)
        if calc_result:
            st.markdown(f"**Finance Calculator:** {calc_result}")

    # 2️⃣ Prepare Chroma DB (async)
    with st.spinner("📦 Preparing document database..."):
        asyncio.run(prepare_chroma_db_async())

    # 3️⃣ Load QA chain (cached)
    qa_chain = load_qa_chain()

    # 4️⃣ Run query async
    with st.spinner("🤖 Thinking... retrieving from documents..."):
        doc_result = asyncio.run(run_query_async(qa_chain, query))
        st.markdown(f"**Document Retrieval:** {doc_result}")






