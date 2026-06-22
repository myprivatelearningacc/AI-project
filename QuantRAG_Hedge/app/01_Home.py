import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import BasicRAGPipeline


load_dotenv()

RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "processed" / "basic_vector_store"


st.set_page_config(
    page_title="QuantRAG Hedge",
    layout="wide",
)


@st.cache_resource
def load_rag_pipeline(chunk_size: int, overlap: int):
    """
    Cache the RAG pipeline so Streamlit does not rebuild everything on every refresh.
    """

    pipeline = BasicRAGPipeline(
        raw_docs_dir=str(RAW_DOCS_DIR),
        vector_store_dir=str(VECTOR_STORE_DIR),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    index_info = pipeline.build_or_load_index()

    return pipeline, index_info


def render_header():
    st.title("QuantRAG Hedge")
    st.caption(
        "Interactive RAG assistant for options pricing, Greeks, delta hedging, "
        "and reinforcement learning-based hedging."
    )


def render_sidebar():
    st.sidebar.title("Settings")

    top_k = st.sidebar.slider(
        "Top-k retrieved chunks",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
    )

    chunk_size = st.sidebar.slider(
        "Chunk size",
        min_value=300,
        max_value=1200,
        value=700,
        step=100,
    )

    overlap = st.sidebar.slider(
        "Chunk overlap",
        min_value=0,
        max_value=300,
        value=120,
        step=20,
    )

    st.sidebar.divider()

    st.sidebar.markdown("### Example questions")

    examples = [
        "What is delta hedging and why is it used?",
        "What are the assumptions of the Black-Scholes model?",
        "Explain delta, gamma, and vega in simple terms.",
        "Why does gamma risk matter in delta hedging?",
        "How can reinforcement learning be used for hedging?",
    ]

    selected_example = st.sidebar.radio(
        "Choose a sample question",
        examples,
        index=0,
    )

    return top_k, chunk_size, overlap, selected_example


def render_overview(index_info):
    st.subheader("Knowledge Base Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Documents", index_info.get("num_docs", 0))

    with col2:
        st.metric("Chunks", index_info.get("num_chunks", 0))

    with col3:
        st.metric("Embedding Dim", index_info.get("embedding_dim", 0))

    with col4:
        st.metric("Vector Store", "Ready")

    sources = index_info.get("sources", [])

    if sources:
        st.markdown("#### Available source files")
        source_df = pd.DataFrame(
            {
                "Source file": sources,
            }
        )
        st.dataframe(source_df, use_container_width=True, hide_index=True)


def render_retrieved_chunks(retrieved_chunks):
    st.subheader("Retrieved Sources")

    rows = []

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        rows.append(
            {
                "Rank": rank,
                "Source": chunk["source"],
                "Chunk ID": chunk["chunk_id"],
                "Similarity Score": round(chunk["score"], 4),
            }
        )

    score_df = pd.DataFrame(rows)

    st.dataframe(score_df, use_container_width=True, hide_index=True)

    chart_df = score_df[["Rank", "Similarity Score"]].set_index("Rank")
    st.bar_chart(chart_df)

    st.markdown("#### Retrieved context")

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        with st.expander(
            f"Rank {rank} | {chunk['source']} | Score: {chunk['score']:.4f}"
        ):
            st.write(chunk["text"])


def render_rag_chat(pipeline, top_k, selected_example):
    st.subheader("RAG Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Ask a question about options, Black-Scholes, Greeks, or hedging",
            value=selected_example,
            height=100,
        )

    with col2:
        st.markdown("##### Suggested use")
        st.write(
            "Ask conceptual or technical questions. "
            "The assistant will answer using the retrieved knowledge base."
        )

    ask_button = st.button("Generate answer", type="primary")

    if ask_button:
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving context and generating answer..."):
            answer, retrieved_chunks, prompt = pipeline.answer_query(
                query=query,
                top_k=top_k,
            )

        st.session_state.chat_history.append(
            {
                "query": query,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
            }
        )

    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]

        st.markdown("### Answer")
        st.write(latest["answer"])

        render_retrieved_chunks(latest["retrieved_chunks"])

        with st.expander("Show full RAG prompt"):
            st.code(latest["prompt"], language="text")

        with st.expander("Chat history"):
            for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
                st.markdown(f"#### Question {i}")
                st.write(item["query"])
                st.markdown("**Answer:**")
                st.write(item["answer"])


def render_about():
    st.subheader("About this Demo")

    st.markdown(
        """
This Streamlit app is the Milestone 4 demo for **QuantRAG Hedge**.

The app connects the Basic RAG pipeline from Milestone 3 to an interactive web interface.

Current features:

- Load the markdown knowledge base.
- Build or load a local vector store.
- Retrieve top-k relevant chunks.
- Generate an answer using retrieved context.
- Display source files and similarity scores.
- Show retrieved context for transparency.

This is currently a RAG demo. Later milestones will add:

- Black-Scholes option pricing.
- Greeks calculation and visualization.
- Delta hedging simulation.
- Hedging P&L dashboard.
- RAG evaluation.
"""
    )


def main():
    render_header()

    top_k, chunk_size, overlap, selected_example = render_sidebar()

    pipeline, index_info = load_rag_pipeline(
        chunk_size=chunk_size,
        overlap=overlap,
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Overview",
            "RAG Chat",
            "About",
        ]
    )

    with tab1:
        render_overview(index_info)

    with tab2:
        render_rag_chat(
            pipeline=pipeline,
            top_k=top_k,
            selected_example=selected_example,
        )

    with tab3:
        render_about()


if __name__ == "__main__":
    main()