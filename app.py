import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import re

st.set_page_config(page_title="Agentic RAG Demo", layout="wide")
st.title("📊 Financial Agentic RAG Demo")

# =========================
# Sidebar and query input first
# =========================
st.sidebar.header("Tools")
use_calculator = st.sidebar.checkbox("Use Finance Calculator", value=True)

query = st.text_area("Enter your query:", "")

# =========================
# Simple arithmetic calculator
# =========================
def finance_calculator(query: str):
    try:
        # Extract basic arithmetic expressions
        expr = re.findall(r"[\d\.\+\-\*\/\(\) ]+", query)
        if expr:
            expr = "".join(expr)
            result = eval(expr)
            return f"Result: {result}"
    except Exception:
        return "Could not compute arithmetic expression."
    return None

# =========================
# Persistent Chroma DB (small demo)
# =========================
CHROMA_DB_DIR = "chroma_db"

def prepare_chroma_db():
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)
        # Load chunks.json
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        docs = [Document(page_content=t) for t in chunks]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR)
        db.persist()

# =========================
# Load pipeline and retriever (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_pipeline_and_retriever():
    with st.spinner("Loading embeddings and LLM... this may take ~30s"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        retriever = db.as_retriever()

        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )
    return qa_chain

# =========================
# Handle query when user clicks
# =========================
if st.button("Run Agent") and query:
    # Step 1: simple calculator (non-blocking)
    calc_result = finance_calculator(query) if use_calculator else None
    if calc_result:
        st.markdown(f"**Finance Calculator:** {calc_result}")

    # Step 2: load DB & QA chain only if needed
    prepare_chroma_db()  # build DB if missing
    qa_chain = load_pipeline_and_retriever()

    with st.spinner("🤖 Thinking... retrieving from documents..."):
        doc_result = qa_chain.run(query)
        st.markdown(f"**Document Retrieval:** {doc_result}")





