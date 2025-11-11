import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Input PDF
PDF_FILE_PATH = "Bhagavad-Gita-Hindi.pdf"

# 2. New Embedding Model
MODEL_NAME = "intfloat/multilingual-e5-base"

# 3. New FAISS Output Folder
DB_FAISS_PATH = "faiss_hindi_index_e5"

def build_db():
    print(f"Starting database build using {MODEL_NAME}")

    # Load PDF
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"Generated {len(texts)} chunks.")

    # Initialize new embeddings
    print(f"Loading embeddings: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create and save FAISS index
    print("Creating FAISS index...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ New FAISS index saved to: {DB_FAISS_PATH}")

if __name__ == "__main__":
    build_db()
