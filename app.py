import os
import re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

# ========== Utility: split text into small chunks ==========
def chunk_text(text, chunk_size=800, overlap=120):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

# ========== Utility: cosine similarity ==========
def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ========== Read PDF -> (page_number, text) ==========
def read_pdf(file) -> list:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append((i + 1, txt))
    return pages

# ========== Create embeddings using Gemini ==========
def embed_texts(client, texts, model="gemini-embedding-001"):
    # Official documentation provides embed_content usage and model example: gemini-embedding-001
    result = client.models.embed_content(model=model, contents=texts)
    # result.embeddings: list of vectors
    return [e.values for e in result.embeddings]

# ========== Generate answer using Gemini ==========
def generate_answer(client, question, context, model="gemini-3-flash-preview"):
    prompt = f"""You are CourseMate AI, an assistant that answers ONLY using the provided context.
If the answer is not in the context, say: "I can't find this in the uploaded materials."

Context:
{context}

Question: {question}

Answer (with short bullet points if helpful):"""
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text

# ========== Streamlit UI ==========
st.set_page_config(page_title="CourseMate AI", page_icon="📚", layout="wide")
st.title("📚 CourseMate AI — Find Course Materials Fast")
st.caption("Upload course PDFs, ask questions, get answers with citations.")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY. Create a .env file and put GEMINI_API_KEY=your_key")
    st.stop()

client = genai.Client(api_key=api_key)

with st.sidebar:
    st.header("1) Upload PDFs")
    uploaded = st.file_uploader("Upload course materials (PDF)", type=["pdf"], accept_multiple_files=True)
    build = st.button("Build Knowledge Base")

if "kb" not in st.session_state:
    st.session_state.kb = None

if build and uploaded:
    st.info("Reading PDFs...")
    all_chunks = []
    meta = []  # (filename, page_number, chunk_text)

    for f in uploaded:
        pages = read_pdf(f)
        for page_num, text in pages:
            chunks = chunk_text(text)
            for ch in chunks:
                if ch.strip():
                    all_chunks.append(ch)
                    meta.append((f.name, page_num, ch))

    st.info(f"Creating embeddings for {len(all_chunks)} chunks...")
    embeddings = embed_texts(client, all_chunks, model="gemini-embedding-001")

    st.session_state.kb = {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "meta": meta
    }

    st.success("Knowledge base ready! Now ask questions.")

st.header("2) Ask a Question")
question = st.text_input("Type your question (e.g., 'What is the Assignment 2 due date?')")

top_k = st.slider("How many references to retrieve (Top-K)?", 2, 8, 4)

if st.button("Ask") and question:
    if not st.session_state.kb:
        st.warning("Please upload PDFs and click 'Build Knowledge Base' first.")
        st.stop()

    kb = st.session_state.kb

    q_emb = embed_texts(client, [question], model="gemini-embedding-001")[0]

    sims = []
    for idx, emb in enumerate(kb["embeddings"]):
        sims.append((cosine_sim(q_emb, emb), idx))
    sims.sort(reverse=True, key=lambda x: x[0])

    picked = sims[:top_k]
    context_blocks = []
    citations = []

    for score, idx in picked:
        filename, page_num, ch = kb["meta"][idx]
        context_blocks.append(f"[{filename} p.{page_num}] {ch}")
        citations.append((filename, page_num, score))

    context = "\n\n".join(context_blocks)

    st.subheader("Answer")
    answer = generate_answer(client, question, context, model="gemini-3-flash-preview")
    st.write(answer)

    st.subheader("Citations (Sources)")
    for filename, page_num, score in citations:
        st.write(f"- {filename} (page {page_num})  similarity={score:.3f}")
``