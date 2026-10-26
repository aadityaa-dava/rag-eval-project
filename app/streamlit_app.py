"""Streamlit front end for the FDA AI-device guidance RAG API."""
import os

import requests
import streamlit as st

API = os.environ.get("RAG_API", "http://localhost:8000")
EXAMPLES = [
    "What should a performance monitoring plan include?",
    "What is data drift?",
    "What is a predetermined change control plan?",
    "Which FDA guidance is the most recent premarket cybersecurity guidance?",
    "What is the FDA user fee for a 510(k) in 2026?",
]

st.set_page_config(page_title="FDA AI-device guidance Q&A", layout="wide")
st.title("FDA guidance Q&A")
st.caption("Answers come only from 10 FDA documents on AI-enabled devices, software, cybersecurity and human factors. "
           "Check every answer against its cited passages.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Passages retrieved (k)", 1, 10, 5)
    retrieval_only = st.toggle("Retrieval only (no LLM)", value=False)
    st.markdown("**Try:**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    try:
        requests.get(f"{API}/health", timeout=2).raise_for_status()
        st.success("API reachable")
    except requests.RequestException:
        st.error(f"API not reachable at {API}")

question = st.text_input("Ask a question", key="q", placeholder="e.g. What is data drift?")

if question and len(question.strip()) >= 3:
    path = "/search" if retrieval_only else "/ask"
    try:
        with st.spinner("Retrieving..." if retrieval_only else "Retrieving and generating (about 10-20 s on a laptop)..."):
            resp = requests.post(f"{API}{path}", json={"question": question, "k": k}, timeout=300)
        if resp.status_code == 503:
            st.error("The answer generator (Ollama) is not running. Switch on 'Retrieval only' or start Ollama.")
            st.stop()
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        st.stop()

    if not retrieval_only:
        if data["refused"]:
            st.warning(data["answer"])
        else:
            st.markdown(data["answer"])
        if data["truncated"]:
            st.info("This answer hit the length limit and may be cut off.")
        t, tok = data["timing"], data["tokens"]
        st.caption(f"retrieval {t['retrieve_ms']:.0f} ms | generation {t['generate_s']:.1f} s | "
                   f"tokens {tok['prompt']} in / {tok['output']} out")
        results = data["sources"]
    else:
        st.caption(f"retrieval {data['retrieve_ms']:.0f} ms")
        results = data["results"]

    st.subheader("Sources")
    for s in results:
        mark = "CITED  " if s.get("cited") else ""
        with st.expander(f"[{s['n']}] {mark}{s['doc']}, {s['pages']} | {s['section']}", expanded=bool(s.get("cited"))):
            st.write(s.get("text", s["snippet"]))
