import os
import uuid
from typing import List
import google.generativeai as genai

# ---------------------------
# Use the correct embedding model
# ---------------------------
EMBED_MODEL = "models/gemini-embedding-001"

# Simple in-memory vector store
VECTOR_STORE = []

# ---------------------------
# Helper: load and chunk text
# ---------------------------
def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, chunk_size=1000, overlap=200) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = i + chunk_size
        chunks.append(text[i:end].strip())
        i = end - overlap
    return [c for c in chunks if len(c) > 20]

# ---------------------------
# Create vector store
# ---------------------------
def create_vector_store(folder_path: str):
    VECTOR_STORE.clear()
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".txt"):
            continue
        full_path = os.path.join(folder_path, fname)
        text = load_txt(full_path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            try:
                emb_resp = genai.embed_content(model=EMBED_MODEL, content=chunk)
                VECTOR_STORE.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "embedding": emb_resp["embedding"],
                    "source": fname
                })
            except Exception as e:
                print(f" Error embedding chunk {i} of {fname}: {e}")
    return VECTOR_STORE

# ---------------------------
# Simple cosine similarity
# ---------------------------
def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-10)

# ---------------------------
# Retrieve top-k relevant docs
# ---------------------------
def retrieve_docs(vector_store: list, query: str, top_k: int = 3) -> str:
    try:
        q_emb_resp = genai.embed_content(model=EMBED_MODEL, content=query)
        q_emb = q_emb_resp["embedding"]
    except Exception as e:
        print(" Error embedding query:", e)
        return ""

    scored = []
    for doc in vector_store:
        score = cosine_similarity(q_emb, doc["embedding"])
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_docs = [d["text"] for _, d in scored[:top_k]]
    return "\n---\n".join(top_docs)
