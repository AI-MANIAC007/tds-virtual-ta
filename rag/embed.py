# rag/embed.py
# rag/embed.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
import os

def create_vector_store():
    print("📥 Loading Discourse JSON data...")
    loader = JSONLoader(
        file_path="data/discourse/tds_kb_posts.json",
        jq_schema=".[] | .posts[]",
        text_content=True,
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    print("🔄 Creating FAISS vector store with HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    print("💾 Saving FAISS index to rag/index/")
    db.save_local("rag/index")
    print("✅ FAISS index saved successfully.")

if __name__ == "__main__":
    create_vector_store()



