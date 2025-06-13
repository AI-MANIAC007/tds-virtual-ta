from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import json

def create_vector_store():
    print("📥 Loading Discourse JSON data...")

    with open("data/discourse/tds_kb_posts.json", "r", encoding="utf-8") as f:
        raw_posts = json.load(f)  # should be a list of strings

    # Convert each post string to a Document
    docs = [Document(page_content=post) for post in raw_posts]

    print(f"✅ Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    print("🔍 Generating embeddings using HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_documents(chunks, embeddings)

    os.makedirs("rag/index", exist_ok=True)
    vectordb.save_local("rag/index")
    print("✅ FAISS index saved successfully in rag/index")

if __name__ == "__main__":
    create_vector_store()



