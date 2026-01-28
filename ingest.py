import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

INDEX_DIR = "faiss_index"

def ingest_uploaded_pdfs(uploaded_files):
    docs = []

    for uploaded_file in uploaded_files:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        for p in pages:
            p.metadata["source"] = uploaded_file.name
            docs.append(p)

        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_DIR)