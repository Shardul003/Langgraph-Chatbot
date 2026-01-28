import streamlit as st
import uuid
from dotenv import load_dotenv

from ingest import ingest_uploaded_pdfs
from vectorstore import load_vectorstore
from agents import input_filter, retriever, answer_agent, evaluation_agent
from graph import build_graph
from db import init_db, save_chat, save_feedback

load_dotenv()
init_db()

st.set_page_config("RAG Chatbot", layout="wide")
st.title("📄 LangGraph RAG Chatbot")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# -------- PDF UPLOAD --------
st.subheader("📤 Upload PDFs")

uploaded_pdfs = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_pdfs:
    with st.spinner("Indexing PDFs..."):
        ingest_uploaded_pdfs(uploaded_pdfs)
    st.success("PDFs indexed successfully")

# -------- LOAD VECTORSTORE --------
vectorstore = load_vectorstore()

if not vectorstore:
    st.info("Upload PDFs to start chatting")
    st.stop()

graph = build_graph(
    input_filter,
    retriever,
    answer_agent,
    evaluation_agent,
    vectorstore
)

# -------- CHAT --------
st.subheader("💬 Ask a question")

query = st.text_input("Ask a question from your PDFs")
st.file_uploader("Upload image (ignored)", type=["png", "jpg", "jpeg"])

if st.button("Ask") and query:
    result = graph.invoke({"query": query})

    save_chat(
        st.session_state.conversation_id,
        query,
        result["answer"],
        result["sources"],
        result["evaluation"]
    )

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Evaluation")
    st.write(result["evaluation"])

    st.subheader("Sources")
    for s in result["sources"]:
        st.markdown(f"- **{s['pdf']}**, page **{s['page']}**")

    st.subheader("Feedback")
    rating = st.slider("Rate", 1, 5)
    comment = st.text_input("Comment")

    if st.button("Submit Feedback"):
        save_feedback(
            st.session_state.conversation_id,
            rating,
            comment
        )
        st.success("Feedback saved")