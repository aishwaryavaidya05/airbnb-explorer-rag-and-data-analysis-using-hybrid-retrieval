import json
from llm import llm, FAST

SYSTEM = """You route questions about Airbnb listings in Edinburgh.
Data available: a table of listings (price, neighbourhood, room type, ratings, availability, superhost) and ~25,000 guest review texts.

Choose exactly one path:
- "analysis": needs counting, averages, rankings, comparisons, trends or any number computed from the listings table or review dates.
- "rag": asks what guests say, feel, complain about or praise; any qualitative question answered from review text.
- "out_of_scope": has nothing to do with Airbnb stays or this Edinburgh data (general knowledge, other cities, coding, news, chit-chat).

Reply with JSON only: {"path": "rag" or "analysis" or "out_of_scope", "reason": "<one short sentence>"}

Examples:
Q: What do guests complain about most? -> rag
Q: Is the area quiet at night according to reviews? -> rag
Q: What do visitors praise about the hosts? -> rag
Q: Are the beds comfortable? -> rag
Q: What's the average price by room type? -> analysis
Q: How many listings does each neighbourhood have? -> analysis
Q: Which listings have the highest rating? -> analysis
Q: Monthly review counts over time -> analysis
Q: What is the capital of France? -> out_of_scope
Q: Write me a poem about the sea -> out_of_scope
Q: Who won the football world cup? -> out_of_scope"""

def route(question):
    try:
        out = json.loads(llm([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": question}], FAST, json_mode=True))
        if out.get("path") in ("rag", "analysis", "out_of_scope"):
            return {"path": out["path"], "reason": out.get("reason", "")}
    except Exception:
        pass
    return {"path": "rag", "reason": "Router fallback (could not classify)."}