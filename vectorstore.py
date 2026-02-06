# vectorstore.py
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index"

@st.cache_resource
def get_embedding_model():
    """Caches the embedding model to prevent redundant reloads."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        return None

    # Use the cached embedding model
    embeddings = get_embedding_model()

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
