"""Streamlit UI:  streamlit run src/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run src/app.py` (script mode) to import the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.answer import RagEngine  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.llm import LLMError  # noqa: E402

st.set_page_config(page_title="VTU Notes Q&A", page_icon="book", layout="wide")


@st.cache_resource(show_spinner="Loading index and embedding model...")
def get_engine(provider: str, top_k: int, reranker: str) -> RagEngine:
    return RagEngine(
        load_settings(llm_provider=provider, top_k=top_k, reranker=reranker)
    )


@st.cache_data(ttl=30)
def get_subjects(provider: str, top_k: int, reranker: str) -> list[str]:
    return get_engine(provider, top_k, reranker).subjects()


defaults = load_settings()

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "LLM provider",
        ["anthropic", "ollama", "extractive"],
        index=["anthropic", "ollama", "extractive"].index(defaults.llm_provider)
        if defaults.llm_provider in {"anthropic", "ollama", "extractive"}
        else 2,
        help="anthropic needs ANTHROPIC_API_KEY; ollama needs a local server; "
        "extractive needs nothing and quotes your notes directly.",
    )
    top_k = st.slider("Passages to retrieve (k)", 1, 12, defaults.top_k)
    reranker = st.selectbox(
        "Reranker", ["lexical", "cross-encoder", "none"],
        index=["lexical", "cross-encoder", "none"].index(defaults.reranker)
        if defaults.reranker in {"lexical", "cross-encoder", "none"}
        else 0,
    )

    engine = get_engine(provider, top_k, reranker)
    subjects = ["All"] + get_subjects(provider, top_k, reranker)
    subject = st.selectbox("Subject", subjects)

    exam_mode = st.toggle("Exam mode (VTU answer format)", value=False)
    marks = st.radio("Marks", [5, 10], index=1, horizontal=True, disabled=not exam_mode)

    st.divider()
    st.caption(f"{engine.retriever.count()} chunks indexed")
    if st.button("Re-index PDFs", use_container_width=True):
        with st.status("Indexing...", expanded=True) as status:
            stats = engine.ingest(report=lambda msg: status.write(msg))
            status.update(label=stats.summary(), state="complete")
        get_subjects.clear()
        st.rerun()

st.title("VTU Notes Q&A")
st.caption("Answers come only from your indexed PDFs, with file and page citations.")

if engine.retriever.count() == 0:
    st.warning(
        f"Nothing indexed yet. Drop PDFs into {defaults.pdf_dir} "
        "(one folder per subject) and press **Re-index PDFs**."
    )

question = st.text_input(
    "Question",
    placeholder="e.g. What is the difference between 3NF and BCNF?",
)
ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask and question.strip():
    try:
        with st.spinner("Searching your notes..."):
            answer = engine.ask(
                question,
                subject=None if subject == "All" else subject,
                k=top_k,
                exam_mode=exam_mode,
                marks=marks,
            )
    except LLMError as exc:
        st.error(str(exc))
        st.stop()

    if answer.not_found:
        st.warning(answer.text)
    else:
        st.markdown(answer.text)

    for warning in answer.warnings:
        st.caption(f":orange[check: {warning}]")

    st.divider()
    st.subheader(f"Sources ({len(answer.sources)})")
    for i, src in enumerate(answer.sources, start=1):
        with st.expander(
            f"{i}. {src.file} - page {src.page}  ({src.subject})  "
            f"score {src.rerank_score:.3f}"
        ):
            st.markdown(src.text)
            st.caption(f"{src.source} - page {src.page} - chunk id {src.chunk_id}")

    st.caption(
        f"provider={answer.provider} | model={answer.model} | {answer.elapsed_s:.2f}s"
    )
