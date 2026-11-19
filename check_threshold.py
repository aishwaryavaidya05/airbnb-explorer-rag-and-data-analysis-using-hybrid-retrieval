from rag import retrieve, MIN_SIM

IN_DOMAIN = [
    "Is it hard to find a place to leave my car?",
    "Were there problems with loud neighbours?",
    "Was the flat freezing in winter?",
    "How easy was check-in?",
    "Was the bed comfortable?",
    "Is the place close to the castle?",
    "Did the host reply quickly?",
    "Was the kitchen well equipped?",
]

OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the football world cup?",
    "Explain quantum computing",
    "What is the best laptop to buy?",
    "Write me a poem about the sea",
    "How do I fix a flat bicycle tyre?",
    "What is the stock price of Tesla?",
]

def best(q):
    return retrieve(q, 1)[0]["sim"]

a = [(q, best(q)) for q in IN_DOMAIN]
b = [(q, best(q)) for q in OFF_TOPIC]

print("IN-DOMAIN")
for q, s in a:
    print(f"  {s:.3f}  {q}")
print("OFF-TOPIC")
for q, s in b:
    print(f"  {s:.3f}  {q}")

lo = min(s for _, s in a)
hi = max(s for _, s in b)
print(f"\nlowest in-domain: {lo:.3f} | highest off-topic: {hi:.3f} | current MIN_SIM: {MIN_SIM}")
print("\nthreshold | in-domain accepted | off-topic rejected")
for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    acc = sum(s >= t for _, s in a)
    rej = sum(s < t for _, s in b)
    print(f"  {t:.2f}    | {acc}/{len(a)}               | {rej}/{len(b)}")