import re
from functools import lru_cache
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rag import _model, _col, clean, SYSTEM
from llm import llm, BIG

TOP_K = 8
MIN_SIM = 0.46
POOL = 30
RRF_K = 60

def tok(s):
    return re.findall(r"[a-z0-9']+", s.lower())

@lru_cache(1)
def _bm25():
    R = pd.read_parquet("data/reviews.parquet")
    return BM25Okapi([tok(t) for t in R["comments"]])

def retrieve(question, k=TOP_K):
    q = _model().encode([question], normalize_embeddings=True)
    r = _col().query(query_embeddings=q.tolist(), n_results=POOL)
    vector_ids = r["ids"][0]
    scores = _bm25().get_scores(tok(question))
    keyword_ids = [str(i) for i in np.argsort(scores)[::-1][:POOL] if scores[i] > 0]
    fused = {}
    for ranked in (vector_ids, keyword_ids):
        for rank, i in enumerate(ranked):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank + 1)
    ids = sorted(fused, key=fused.get, reverse=True)[:k]
    g = _col().get(ids=ids, include=["documents", "metadatas", "embeddings"])
    pos = {i: j for j, i in enumerate(g["ids"])}
    out = []
    for n, i in enumerate(ids):
        j = pos[i]
        sim = float(np.dot(q[0], np.array(g["embeddings"][j])))
        out.append({"id": f"S{n+1}", "text": g["documents"][j], "meta": g["metadatas"][j], "sim": sim})
    return out

def answer_rag(question):
    chunks = retrieve(question)
    if max(c["sim"] for c in chunks) < MIN_SIM:
        return {"content": "I couldn't find reviews relevant enough to answer that.", "chunks": chunks}
    ctx = "\n\n".join(
        f"[{c['id']}] (neighbourhood: {c['meta']['neighbourhood']}, room: {c['meta']['room_type']}, "
        f"date: {c['meta']['date']}) {c['text'][:600]}" for c in chunks)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<sources>\n{ctx}\n</sources>\n\nQuestion: {question}"}]
    return {"content": clean(llm(msgs, BIG, temperature=0.2)), "chunks": chunks}