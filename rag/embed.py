# rag/embed.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Use HuggingFace local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store():
    print("📥 Loading Discourse JSON data...")
    disc_loader = JSONLoader(
        "data/discourse/tds_kb_posts.json",
        jq_schema=".[] | {text: (.posts | join(\"\\n\"))}",
        text_content=True
    )
    docs = disc_loader.load()

    print(f"✅ Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    print("🔄 Creating FAISS vector store...")
    vectordb = FAISS.from_documents(chunks, embedding_model)

    print("💾 Saving FAISS index to rag/index/")
    vectordb.save_local("rag/index")
    print("✅ FAISS index saved successfully.")

if __name__ == "__main__":
    create_vector_store()


