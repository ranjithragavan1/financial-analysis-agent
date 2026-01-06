import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents():
    # 1. Load the PDF
    print("--- 1. Loading PDF... ---")
    loader = PyPDFLoader("apple_10k.pdf") 
    documents = loader.load()
    
    # 2. Split into small chunks (Critical for RAG)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    print(f"--- 2. Split into {len(docs)} chunks ---")

    # 3. Embed Locally (Uses your GPU automatically if set up, otherwise CPU)
    print("--- 3. Embedding (This runs locally on your machine) ---")
    
    # We use a standard open-source model. It downloads once (80MB).
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Create the Database
    vector_db = FAISS.from_documents(docs, embeddings)
    
    # 5. Save to disk
    vector_db.save_local("faiss_index")
    print("--- SUCCESS! Database saved to folder 'faiss_index' ---")

if __name__ == "__main__":
    ingest_documents()