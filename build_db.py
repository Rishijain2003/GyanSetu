import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. The document file is in the working directory
PDF_FILE_PATH = "Bhagavad-Gita-Hindi.pdf"
# 2. The embedding model you specified
MODEL_NAME = 'hiiamsid/sentence_similarity_hindi'
# 3. The name of the folder to save the vector database
DB_FAISS_PATH = 'faiss_hindi_index'

def build_db():
    """
    Loads the PDF, splits the text, generates Hindi embeddings, 
    and saves the vector database using FAISS.
    """
    print(f"Starting database build for: {PDF_FILE_PATH}")

    # 1. Load the PDF Document
    try:
        print("Loading document...")
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()
    except FileNotFoundError:
        print(f"Error: The file '{PDF_FILE_PATH}' was not found in the working directory.")
        return
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    # 2. Split Text into Chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"Generated {len(texts)} text chunks.")
    
    
    print("\nDiagnostic Check: Sample of First Chunk Content")
    print("Chunk 1 Sample:")
    print(texts[0].page_content[:1000]) #
    print("-------------------------------------------\n")


    
    # 3. Initialize the Hindi Embedding Model
    print(f"Initializing Hindi embedding model: {MODEL_NAME}...")
    # The HuggingFaceEmbeddings class loads the SentenceTransformer model
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )


    print(f"Creating FAISS index and generating {len(texts)} embeddings...")
    db = FAISS.from_documents(texts, embeddings)
    
    # Save the database locally
    db.save_local(DB_FAISS_PATH)
    
    print("\nDatabase build complete!")
    print(f"The vector index is saved to the directory: **{DB_FAISS_PATH}**")

if __name__ == '__main__':

    build_db()