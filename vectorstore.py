import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index"

def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )