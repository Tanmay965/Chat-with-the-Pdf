import os
import re
import time

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")

## Settings (keep these the same as in the notebook)
GEN_MODEL = "gemini-3.5-flash"
EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NOT_FOUND = "I couldn't find that in the document."

SYSTEM_INSTRUCTIONS = (
    "You are a careful assistant that answers questions about one PDF document. "
    "Use ONLY the context excerpts provided. Do not use outside knowledge. "
    "Cite the page for each fact like (p. 3). "
    "If the context does not contain the answer, reply exactly: "
    "\"I couldn't find that in the document.\" "
    "Keep answers clear and concise."
)


## Pipeline functions (same logic as the notebook)
def read_pdf(file):
    pages = []
    for i, page in enumerate(PdfReader(file).pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for p in pages:
        text = p["text"]
        if len(text) < 20:
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if cut > start + size // 2:
                    end = cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append({"id": len(chunks), "page": p["page"], "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def embed_texts(client, texts, task_type, batch_size=50, pause=1.0):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors.extend([e.values for e in resp.embeddings])
                break
            except Exception:
                time.sleep(2 ** attempt * 2)
        else:
            raise RuntimeError("Embedding failed after several retries. Check your API key and quota.")
        time.sleep(pause)
    arr = np.array(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def retrieve(client, question, chunks, vectors, top_k):
    q_vec = embed_texts(client, [question], "RETRIEVAL_QUERY", pause=0)[0]
    scores = vectors @ q_vec
    best = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]


def generate_answer(client, question, retrieved):
    context = "\n\n".join(f"[Excerpt {i + 1} | page {r['page']}]\n{r['text']}" for i, r in enumerate(retrieved))
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTIONS, temperature=0.2),
            )
            return (resp.text or "").strip()
        except Exception:
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("Generation failed after several retries. Check your API key and quota.")


## Sidebar: API key and settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
top_k = st.sidebar.slider("Excerpts sent to the model", 1, 8, 4)
min_similarity = st.sidebar.slider("Refuse below this similarity", 0.0, 1.0, 0.30, 0.01,
                                   help="If even the best excerpt scores below this, the app answers 'not found' without calling the LLM.")
st.sidebar.caption("Use the number you found in Step 12 of the notebook.")

## Main page
st.title("📄 Chat with your PDF")
st.write("Upload a PDF, ask a question, and get an answer with page citations. If the document does not contain the answer, the bot says so.")

if not api_key:
    st.info("Paste your Gemini API key in the sidebar to start.")
    st.stop()

client = genai.Client(api_key=api_key)
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is None:
    st.stop()

## Build the index once per uploaded file
file_key = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    with st.spinner("Reading the PDF and building the search index..."):
        pages = read_pdf(uploaded)
        chunks = chunk_pages(pages)
        if not chunks:
            st.error("No readable text found. This PDF may be scanned images, which this app cannot read.")
            st.stop()
        try:
            vectors = embed_texts(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
        except Exception as e:
            st.error(f"Could not build the index: {e}")
            st.stop()
    st.session_state.update(file_key=file_key, chunks=chunks, vectors=vectors, n_pages=len(pages), history=[])

chunks, vectors = st.session_state["chunks"], st.session_state["vectors"]
st.caption(f"{uploaded.name}: {st.session_state['n_pages']} pages, {len(chunks)} chunks indexed")

## Chat
for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Sources"):
            for s in turn["sources"]:
                st.markdown(f"**Page {s['page']}** (similarity {s['score']:.2f})")
                st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

question = st.chat_input("Ask a question about the PDF")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                retrieved = retrieve(client, question, chunks, vectors, top_k)
                if retrieved[0]["score"] < min_similarity:
                    answer = NOT_FOUND
                else:
                    answer = generate_answer(client, question, retrieved)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.write(answer)
        with st.expander("Sources"):
            for s in retrieved:
                st.markdown(f"**Page {s['page']}** (similarity {s['score']:.2f})")
                st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
    st.session_state["history"].append({"question": question, "answer": answer, "sources": retrieved})
