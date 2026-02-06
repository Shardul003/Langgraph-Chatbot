import streamlit as st
import uuid
from dotenv import load_dotenv

from ingest import ingest_uploaded_pdfs
from vectorstore import load_vectorstore
from agents import input_filter, retriever, answer_agent, evaluation_agent
from graph import build_graph
from db import init_db, save_chat, save_feedback

# -------------------------------------------------
# App setup
# -------------------------------------------------
load_dotenv()
init_db()

st.set_page_config(page_title="LangGraph RAG Chatbot", layout="wide")
st.title("📄 LangGraph RAG Chatbot")

# -------------------------------------------------
# Session state init
# -------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "result" not in st.session_state:
    st.session_state.result = None

# -------------------------------------------------
# PDF Upload & Ingestion
# -------------------------------------------------
st.subheader("📤 Upload PDFs")

uploaded_pdfs = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_pdfs:
    with st.spinner("Indexing PDFs (chunking + embeddings)..."):
        ingest_uploaded_pdfs(uploaded_pdfs)
    st.success("PDFs indexed successfully")

# -------------------------------------------------
# Load Vector Store
# -------------------------------------------------
vectorstore = load_vectorstore()

if not vectorstore:
    st.info("Upload PDFs to start chatting.")
    st.stop()

# -------------------------------------------------
# Build LangGraph (IMPORTANT FIX)
# -------------------------------------------------
@st.cache_resource
def get_graph(_vs):  # leading underscore is REQUIRED
    return build_graph(
        input_filter,
        retriever,
        answer_agent,
        evaluation_agent,
        _vs
    )

graph = get_graph(vectorstore)

# -------------------------------------------------
# Chat UI
# -------------------------------------------------
st.subheader("💬 Ask a question")

query = st.text_input("Ask a question from your uploaded PDFs")

st.file_uploader(
    "Upload image (ignored)",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------------------------
# Ask button logic (FIXED)
# -------------------------------------------------
# Change this line in app.py
if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            # Directly assign to session state
            st.session_state.result = graph.invoke({"query": query})
            # Force a rerun to ensure the UI updates with the new state
            st.rerun()

# -------------------------------------------------
# Display result (PERSISTENT)
# -------------------------------------------------
if st.session_state.result:
    result = st.session_state.result

    save_chat(
        st.session_state.conversation_id,
        query,
        result.get("answer"),
        result.get("sources"),
        result.get("evaluation")
    )

    st.subheader("✅ Answer")
    st.write(result.get("answer", "No answer generated"))

    st.subheader("🧪 Evaluation")
    st.write(result.get("evaluation", "N/A"))

    st.subheader("📚 Sources")
    if result.get("sources"):
        for s in result["sources"]:
            st.markdown(f"- **{s['pdf']}**, page **{s['page']}**")
    else:
        st.write("No sources found")

    # -------------------------------------------------
    # Feedback
    # -------------------------------------------------
    st.subheader("📝 Feedback")

    rating = st.slider("Rate the answer", 1, 5)
    comment = st.text_input("Optional comment")

    if st.button("Submit Feedback"):
        save_feedback(
            st.session_state.conversation_id,
            rating,
            comment
        )
        st.success("Feedback saved 👍")
