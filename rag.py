import re
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from llm import llm, BIG

TOP_K = 8
MIN_SIM = 0.25

@lru_cache(1)
def _model():
    m = SentenceTransformer("all-MiniLM-L6-v2")
    m.max_seq_length = 128
    return m

@lru_cache(1)
def _col():
    return chromadb.PersistentClient(path="chroma_db").get_collection("reviews")

def retrieve(question, k=TOP_K):
    emb = _model().encode([question], normalize_embeddings=True).tolist()
    r = _col().query(query_embeddings=emb, n_results=k)
    return [{"id": f"S{i+1}", "text": d, "meta": m, "sim": 1 - dist}
            for i, (d, m, dist) in enumerate(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))]

SYSTEM = """You answer questions about Airbnb stays in Edinburgh using ONLY the guest review excerpts provided.
Rules:
- Use only the sources. Do not use outside knowledge.
- Cite sources inline like [S1], [S3]. Always use square brackets.
- You only see a small sample of reviews. Never say something is the "most common", "most often" or "main" issue. Say what these reviews mention, for example "Several of the retrieved reviews mention...".
- If the sources don't clearly answer the question, say "There is not enough evidence in the reviews to answer this."
- Be concise: 3-5 sentences."""

def clean(text):
    text = re.sub(r"[【\[]\s*(S\d+)\s*[】\]]", r"[\1]", text)
    return re.sub(r"(?<![\[\w])S(\d+)(?![\]\w])", r"[S\1]", text)

def answer_rag(question):
    chunks = retrieve(question)
    if chunks[0]["sim"] < MIN_SIM:
        return {"content": "I couldn't find reviews relevant enough to answer that.", "chunks": chunks}
    ctx = "\n\n".join(
        f"[{c['id']}] (neighbourhood: {c['meta']['neighbourhood']}, room: {c['meta']['room_type']}, "
        f"date: {c['meta']['date']}) {c['text'][:600]}" for c in chunks)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<sources>\n{ctx}\n</sources>\n\nQuestion: {question}"}]
    return {"content": clean(llm(msgs, BIG, temperature=0.2)), "chunks": chunks}