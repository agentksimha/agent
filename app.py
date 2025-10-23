import json
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="Agentic RAG Demo", layout="wide")
st.title("📊 Financial Agentic RAG Demo")

# =========================
# Load pipeline and retriever
# =========================
@st.cache_resource(show_spinner=True)
def load_pipeline_and_retriever():
    with st.spinner("Loading documents, embeddings, and model... this may take 1–2 minutes"):
        # Load split texts from chunks.json
        with open("chunks.json", "r", encoding="utf-8") as f:
            split_texts = json.load(f)

        # Convert to Document objects
        docs = [Document(page_content=t) for t in split_texts]

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(docs, embeddings)
        retriever = db.as_retriever()

        # LLM pipeline
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

        # QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

    return qa_chain

# Load QA chain
qa_chain = load_pipeline_and_retriever()

# =========================
# Simple financial calculator tool
# =========================
def finance_calculator(query: str):
    if "growth" in query.lower():
        return "Company growth rate estimated at 7.5% YoY."
    elif "revenue" in query.lower():
        return "Revenue in Q2 increased by 12%."
    return "No relevant financial data found."

# =========================
# Streamlit sidebar and query input
# =========================
st.sidebar.header("Tools")
use_calculator = st.sidebar.checkbox("Use Finance Calculator", value=True)

query = st.text_area("Enter your query:", "")

if st.button("Run Agent") and query:
    with st.spinner("Thinking..."):
        # Step 1: calculator tool
        calc_result = finance_calculator(query) if use_calculator else ""

        # Step 2: retrieve from documents
        doc_result = qa_chain.run(query)

        # Display results
        st.subheader("💡 Results")
        if calc_result:
            st.markdown(f"**Finance Calculator:** {calc_result}")
        st.markdown(f"**Document Retrieval:** {doc_result}")



