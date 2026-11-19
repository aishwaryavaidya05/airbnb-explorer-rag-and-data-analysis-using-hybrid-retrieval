import rag
import hybrid

RETRIEVAL_SET = [
    ("Is it hard to find a place to leave my car?", ["parking", "car park", "parked"]),
    ("Were there problems with loud neighbours or street sounds?", ["noise", "noisy", "loud", "thin walls"]),
    ("Was the flat freezing in winter?", ["cold", "heating", "freezing", "radiator", "heater"]),
    ("Did the host welcome us in person and show us around?", ["met us", "welcomed", "greeted", "showed us", "show us"]),
    ("How easy was it to get the keys and get in?", ["key", "lockbox", "check-in", "check in", "self check"]),
    ("Was the mattress and bedding comfortable?", ["bed", "mattress", "pillow"]),
    ("Can I walk to the castle from the apartment?", ["castle"]),
    ("Is the place close to the main train station?", ["waverley", "train station", "station"]),
    ("Were there lots of stairs to climb with luggage?", ["stairs", "steps", "flights", "climb"]),
    ("Is the kitchen well equipped for cooking?", ["kitchen", "cook", "equipped"]),
    ("Did anything break or not work properly?", ["broken", "broke", "not working", "didn't work", "faulty", "leak"]),
    ("Was the flat spotless when we arrived?", ["clean", "spotless", "immaculate"]),
    ("Was the wifi reliable enough to work remotely?", ["wifi", "wi-fi", "internet"]),
    ("How was the hot water and shower?", ["shower", "hot water", "water pressure"]),
    ("Are there good restaurants and pubs nearby?", ["restaurant", "pub", "cafe", "bars"]),
    ("Is the neighbourhood safe to walk at night?", ["safe", "felt safe"]),
    ("Did the host reply quickly to messages?", ["responsive", "quick to respond", "replied", "responded", "prompt"]),
    ("Was the place smaller than the photos suggested?", ["small", "tiny", "cramped", "compact", "smaller"]),
    ("Is the area good for families with children?", ["kids", "children", "child", "family", "baby", "toddler"]),
    ("Were there problems with smells, damp or pests?", ["smell", "smelly", "mouse", "mice", "pest", "damp", "mould"]),
    ("Is it a good base for the Royal Mile and Old Town?", ["royal mile", "old town"]),
    ("Did they provide breakfast items, coffee or tea?", ["breakfast", "coffee", "tea"]),
    ("Were the views from the flat good?", ["view", "views"]),
    ("Was it easy to get around by bus or on foot?", ["bus", "walk", "walking", "tram"]),
]

def score(fn):
    ranks, prec = [], []
    for q, kws in RETRIEVAL_SET:
        ch = fn(q, 8)
        flags = [any(w in c["text"].lower() for w in kws) for c in ch]
        ranks.append(next((i + 1 for i, f in enumerate(flags) if f), None))
        prec.append(sum(flags[:5]) / 5)
    n = len(RETRIEVAL_SET)
    out = {f"hit@{k}": f"{sum(1 for r in ranks if r and r <= k)}/{n}" for k in (1, 3, 5, 8)}
    out["MRR"] = round(sum(1 / r for r in ranks if r) / n, 3)
    out["precision@5"] = round(sum(prec) / n, 3)
    return out

v = score(rag.retrieve)
h = score(hybrid.retrieve)
print("\n| Metric | Vector only | Hybrid (BM25 + vector, RRF) |\n|---|---|---|")
for k in v:
    print(f"| {k} | {v[k]} | {h[k]} |")