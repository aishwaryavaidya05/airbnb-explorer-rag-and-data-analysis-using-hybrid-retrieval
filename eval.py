import json, time
import numpy as np
import pandas as pd
from router import route
from rag import retrieve
from analysis import answer_analysis, load

PAUSE = 2

ROUTER_CLEAR = [
    ("Do guests complain about noise?", "rag"),
    ("Are guests satisfied with check-in?", "rag"),
    ("What do visitors like about the location?", "rag"),
    ("Is parking mentioned as a problem?", "rag"),
    ("How do guests describe communication with hosts?", "rag"),
    ("Is the place clean according to guests?", "rag"),
    ("What do guests say about the beds?", "rag"),
    ("Do reviewers mention problems with heating?", "rag"),
    ("How do guests feel about the kitchen?", "rag"),
    ("What do people say about the stairs?", "rag"),
    ("Are guests happy with the wifi?", "rag"),
    ("What do guests say about arriving late?", "rag"),
    ("What share of listings are Entire home/apt?", "analysis"),
    ("Which 5 neighbourhoods are most expensive on average?", "analysis"),
    ("How many superhosts are there?", "analysis"),
    ("What is the correlation between price and rating?", "analysis"),
    ("Show the 10 cheapest listings with at least 20 reviews", "analysis"),
    ("How has review volume changed by year?", "analysis"),
    ("What is the median price per room type?", "analysis"),
    ("How many listings accommodate more than 4 guests?", "analysis"),
    ("Which neighbourhood has the highest average rating?", "analysis"),
    ("Plot the number of reviews per month", "analysis"),
    ("What is the average availability over the year?", "analysis"),
    ("How many listings have no reviews?", "analysis"),
]

ROUTER_AMBIG = [
    ("Is the Old Town expensive?", "analysis"),
    ("Are superhosts really better?", "analysis"),
    ("Which area is the quietest?", "rag"),
    ("Is the location rated highly?", "analysis"),
    ("What do guests think about value for money?", "rag"),
    ("Do cheaper listings get worse reviews?", "analysis"),
    ("Is cleanliness a problem?", "rag"),
    ("Where do guests say is the best area to stay?", "rag"),
]

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

L, R = load()
yr = R["date"].dt.year

ANALYSIS_SET = [
    ("scalar", "How many listings are there?", len(L)),
    ("scalar", "What is the median nightly price across all listings?", L.price.median()),
    ("scalar", "How many listings have more than 100 reviews?", int((L.number_of_reviews > 100).sum())),
    ("scalar", "How many listings cost more than 500 per night?", int((L.price > 500).sum())),
    ("scalar", "What is the average number of guests listings accommodate?", L.accommodates.mean()),
    ("scalar", "How many listings are superhosts?", int(L.host_is_superhost.eq(True).sum())),
    ("scalar", "How many reviews in the reviews table are from 2024?", int((yr == 2024).sum())),
    ("scalar", "What is the highest nightly price?", L.price.max()),
    ("scalar", "What is the average price of listings with at least 50 reviews?", L[L.number_of_reviews >= 50].price.mean()),
    ("group", "What is the average price by room type?", L.groupby("room_type").price.mean().to_dict()),
    ("group", "How many listings are there for each room type?", L.room_type.value_counts().to_dict()),
    ("group", "What is the median price by room type?", L.groupby("room_type").price.median().to_dict()),
    ("group", "What is the average review_scores_rating by room type?", L.groupby("room_type").review_scores_rating.mean().to_dict()),
    ("group", "How many reviews are there per year?", {str(k): v for k, v in R.groupby(yr).size().to_dict().items()}),
    ("top", "Which 5 neighbourhoods have the most listings?", list(L.neighbourhood.value_counts().head(5).index)),
    ("top", "Which 5 neighbourhoods have the highest average price?", list(L.groupby("neighbourhood").price.mean().nlargest(5).index)),
    ("top", "Which 3 room types have the most listings?", list(L.room_type.value_counts().head(3).index)),
]

def norm(x):
    s = str(x).strip().lower()
    return s[:-2] if s.endswith(".0") else s

def is_num(v):
    return pd.api.types.is_number(v) and not isinstance(v, (bool, np.bool_)) and pd.notna(v)

def close(a, b, tol):
    return abs(a - b) <= max(tol * abs(b), 0.01)

def as_frame(out):
    if out["table"] is not None:
        return out["table"]
    return pd.DataFrame({"value": [out["raw"]]})

def check(kind, truth, df):
    if kind == "scalar":
        nums = [float(v) for v in df.values.flatten() if is_num(v)]
        return any(close(n, float(truth), 0.005) for n in nums)
    if kind == "group":
        hit = 0
        for key, val in truth.items():
            for _, row in df.iterrows():
                cells = list(row.values)
                if any(norm(c) == norm(key) for c in cells):
                    rest = [float(c) for c in cells if is_num(c) and norm(c) != norm(key)]
                    if any(close(n, float(val), 0.01) for n in rest):
                        hit += 1
                        break
        return hit == len(truth)
    cells = {norm(c) for c in df.values.flatten()}
    return len(df) == len(truth) and all(norm(t) in cells for t in truth)

def run_router(rows, label):
    ok = 0
    for q, y in rows:
        got = route(q)["path"]
        ok += got == y
        if got != y:
            print(f"  ROUTER MISS [{label}] {q} | expected {y}, got {got}")
        time.sleep(PAUSE / 2)
    return ok

res = {}

res["router_clear"] = (run_router(ROUTER_CLEAR, "clear"), len(ROUTER_CLEAR))
res["router_ambiguous"] = (run_router(ROUTER_AMBIG, "ambiguous"), len(ROUTER_AMBIG))
print("Router clear:", res["router_clear"], "| ambiguous:", res["router_ambiguous"])

ranks, prec = [], []
for q, kws in RETRIEVAL_SET:
    ch = retrieve(q, 8)
    flags = [any(w in c["text"].lower() for w in kws) for c in ch]
    rank = next((i + 1 for i, f in enumerate(flags) if f), None)
    ranks.append(rank)
    prec.append(sum(flags[:5]) / 5)
    if rank is None or rank > 3:
        print(f"  RETRIEVAL MISS (first hit rank {rank}): {q}")
n = len(RETRIEVAL_SET)
for k in (1, 3, 5, 8):
    res[f"hit@{k}"] = (sum(1 for r in ranks if r and r <= k), n)
res["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
res["precision@5"] = round(sum(prec) / n, 3)
print({k: v for k, v in res.items() if k.startswith(("hit", "mrr", "prec"))})

by_kind = {"scalar": [0, 0], "group": [0, 0], "top": [0, 0]}
for kind, q, truth in ANALYSIS_SET:
    out = answer_analysis(q)
    right = check(kind, truth, as_frame(out))
    by_kind[kind][0] += right
    by_kind[kind][1] += 1
    print(("PASS" if right else "FAIL"), f"[{kind}]", q)
    time.sleep(PAUSE)
res["analysis"] = {k: tuple(v) for k, v in by_kind.items()}
res["analysis_total"] = (sum(v[0] for v in by_kind.values()), len(ANALYSIS_SET))

with open("eval_results.json", "w") as f:
    json.dump(res, f, indent=2)

print("\n| Component | Metric | Result |\n|---|---|---|")
print(f"| Router | Accuracy, clear questions | {res['router_clear'][0]}/{res['router_clear'][1]} |")
print(f"| Router | Accuracy, ambiguous questions | {res['router_ambiguous'][0]}/{res['router_ambiguous'][1]} |")
for k in (1, 3, 5, 8):
    print(f"| Retrieval | hit@{k} | {res[f'hit@{k}'][0]}/{n} |")
print(f"| Retrieval | MRR | {res['mrr']} |")
print(f"| Retrieval | precision@5 (keyword proxy) | {res['precision@5']} |")
for k, v in res["analysis"].items():
    print(f"| Analysis | {k} questions | {v[0]}/{v[1]} |")
print(f"| Analysis | Overall | {res['analysis_total'][0]}/{res['analysis_total'][1]} |")