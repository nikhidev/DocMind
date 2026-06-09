import streamlit as st
import fitz
import os
from rag.chunking import create_chunk
from rag.vector_db import store_chunks
from llm.summary import summarize_text
from rag.retrival import retrieve_chunks
from llm.qa import answer_question
from rag.vector_db import index_name

st.set_page_config(
    page_title="DocMind — AI Document Assistant",
    page_icon="📄",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F1B2D;
    color: #F0F4FF;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Page wrapper */
.block-container {
    padding: 2.5rem 2rem 4rem;
    max-width: 760px;
}

/* ── Wordmark ── */
.wordmark {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.25rem;
}
.wordmark-icon {
    width: 36px; height: 36px;
    background: #4F8EF7;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.wordmark-text {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F0F4FF;
}
.wordmark-text span { color: #4F8EF7; }
.tagline {
    font-size: 0.82rem;
    color: #6B7FA3;
    margin-bottom: 2rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #1E2D45;
    border: 1.5px dashed #2E4166;
    border-radius: 12px;
    padding: 1.25rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4F8EF7;
}
[data-testid="stFileUploader"] label {
    color: #A0B0CC !important;
    font-size: 0.9rem;
}

/* ── Document passport card ── */
.passport {
    background: #1E2D45;
    border: 1px solid #2E4166;
    border-left: 3px solid #4F8EF7;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin: 1.5rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}
.passport-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: #F0F4FF;
    margin-bottom: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 360px;
}
.passport-meta {
    font-size: 0.78rem;
    color: #6B7FA3;
    letter-spacing: 0.02em;
}
.passport-badge {
    background: #4F8EF720;
    color: #4F8EF7;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    white-space: nowrap;
    letter-spacing: 0.03em;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4F8EF7;
    margin: 2rem 0 0.6rem;
}

/* ── Summary box ── */
.summary-box {
    background: #1E2D45;
    border: 1px solid #2E4166;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #C8D6F0;
}

/* ── Answer box ── */
.answer-box {
    background: #162338;
    border: 1px solid #4F8EF740;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #C8D6F0;
    margin-top: 0.75rem;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1E2D45;
    margin: 1.75rem 0;
}

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: transparent !important;
}
[data-testid="stMetricLabel"] { color: #6B7FA3 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #F0F4FF !important; font-size: 1.6rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: #4F8EF7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.02em !important;
    transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover {
    background: #3a75e0 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary button style via data attribute */
[data-secondary="true"] .stButton > button {
    background: #1E2D45 !important;
    border: 1px solid #2E4166 !important;
    color: #A0B0CC !important;
}
[data-secondary="true"] .stButton > button:hover {
    background: #253652 !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    background: #1E2D45 !important;
    border: 1px solid #2E4166 !important;
    border-radius: 8px !important;
    color: #F0F4FF !important;
    padding: 0.6rem 0.9rem !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4F8EF7 !important;
    box-shadow: 0 0 0 2px #4F8EF720 !important;
}
.stTextInput label { color: #A0B0CC !important; font-size: 0.85rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #4F8EF7; }

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Wordmark ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="wordmark">
    <div class="wordmark-icon">📄</div>
    <div class="wordmark-text">Doc<span>Mind</span></div>
</div>
<div class="tagline">AI Document Assistant</div>
""", unsafe_allow_html=True)


# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a PDF to get started",
    type=["pdf"],
    label_visibility="visible",
)

if uploaded_file is not None:

    # Reset state when a new file is uploaded
    if (
        "current_file" not in st.session_state
        or st.session_state.current_file != uploaded_file.name
    ):
        st.session_state.current_file = uploaded_file.name
        for key in ("summary", "pinecone_stored"):
            st.session_state.pop(key, None)

    # Extract text
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    word_count = len(text.split())
    page_count = doc.page_count

    # ── Document passport ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="passport">
        <div>
            <div class="passport-name">📎 {uploaded_file.name}</div>
            <div class="passport-meta">{page_count} pages &nbsp;·&nbsp; {word_count:,} words</div>
        </div>
        <div class="passport-badge">✓ Loaded</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Store in Pinecone ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Vector Index</div>', unsafe_allow_html=True)

    if not st.session_state.get("pinecone_stored"):
        if st.button("Store in Pinecone"):
            with st.spinner("Creating embeddings and indexing chunks…"):
                chunks = create_chunk(text)
                
                store_chunks(chunks, namespace=uploaded_file.name)
                st.session_state.pinecone_stored = True
                st.session_state.chunk_count = len(chunks)
            st.rerun()
    else:
        st.success(f"✓ {st.session_state.chunk_count} chunks indexed in Pinecone")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Summary ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)

    if "summary" not in st.session_state:
        with st.spinner("Reading the document…"):
            st.session_state.summary = summarize_text(text)

    st.markdown(
        f'<div class="summary-box">{st.session_state.summary}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Q&A ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Question",
        placeholder="What are the main findings of this paper?",
        label_visibility="collapsed",
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Finding the answer…"):
                chunks = retrieve_chunks(question, namespace=uploaded_file.name)
                context = "\n".join(chunks)
                answer = answer_question(question, context)
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )

else:
    # Empty state
    st.markdown("""
    <div style="margin-top:3rem; text-align:center; color:#3A5070;">
        <div style="font-size:2.5rem; margin-bottom:0.75rem;">📂</div>
        <div style="font-size:0.9rem;">Upload a PDF above to begin</div>
    </div>
    """, unsafe_allow_html=True)