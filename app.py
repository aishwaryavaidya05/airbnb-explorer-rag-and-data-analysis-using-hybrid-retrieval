import streamlit as st
from router import route
from hybrid import answer_rag
from analysis import answer_analysis

st.set_page_config(page_title="Edinburgh Airbnb Explorer", page_icon="🏠", layout="wide")
MAX_Q = 15

EXAMPLES = [
    "What do guests complain about most?",
    "Do guests say the area is noisy at night?",
    "How do guests describe the hosts?",
    "What's the average price by room type?",
    "Which 10 neighbourhoods have the most listings?",
    "How has the number of reviews changed by year?",
]

st.title("🏠 Edinburgh Airbnb Explorer")
st.caption("Ask about what guests say (RAG over reviews) or about the numbers (pandas). A router picks the path.")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("count", 0)

def show_table(df):
    plottable = df.shape[1] >= 2 and 1 < len(df) <= 60 and df.iloc[:, 1].dtype.kind in "if"
    if not plottable:
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(320, 38 + 35 * len(df)))
        return
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
    with right:
        chart = df.set_index(df.columns[0])[df.columns[1]]
        chart.index = chart.index.astype(str)
        first = str(df.columns[0]).lower()
        if any(k in first for k in ("date", "month", "year")):
            st.line_chart(chart, height=300)
        else:
            st.bar_chart(chart, height=300)

def render(m):
    if m["role"] == "user":
        st.markdown(m["content"])
        return
    if m.get("path") != "error":
        st.caption(f"Path: **{m['path']}** · {m['reason']}")
    st.markdown(m["content"])
    if m.get("path") == "rag" and m.get("chunks"):
        with st.expander(f"Sources ({len(m['chunks'])})"):
            for c in m["chunks"]:
                st.markdown(f"**[{c['id']}]** similarity {c['sim']:.2f} · {c['meta']['neighbourhood']} · "
                            f"{c['meta']['room_type']} · {c['meta']['date']}")
                st.write(c["text"])
    if m.get("path") == "analysis":
        if m.get("table") is not None:
            show_table(m["table"])
        with st.expander("Code that ran"):
            st.code(m.get("code", ""), language="python")

with st.sidebar:
    st.header("Try these")
    picked = None
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            picked = q
    st.caption(f"Questions used: {st.session_state.count}/{MAX_Q}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render(m)

question = st.chat_input("Ask a question about Edinburgh Airbnbs...") or picked
if question:
    if st.session_state.count >= MAX_Q:
        st.warning("Question limit reached for this session. Refresh to start again.")
        st.stop()
    st.session_state.count += 1
    um = {"role": "user", "content": question}
    st.session_state.messages.append(um)
    with st.chat_message("user"):
        render(um)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = route(question)
                if r["path"] == "out_of_scope":
                    out = {"content": "That looks outside what I can answer. I only cover Edinburgh Airbnb listings and guest reviews. Try a question about prices, neighbourhoods, or what guests say."}
                elif r["path"] == "rag":
                    out = answer_rag(question)
                else:
                    out = answer_analysis(question)
                msg = {"role": "assistant", "path": r["path"], "reason": r["reason"], **out}
            except Exception as e:
                msg = {"role": "assistant", "path": "error", "reason": "",
                       "content": f"Something went wrong ({type(e).__name__}). If it's a rate limit, wait a minute and retry."}
        render(msg)
    st.session_state.messages.append(msg)