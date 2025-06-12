# rag/embed.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = "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"

def create_vector_store():
    course_loader = DirectoryLoader("data/course", glob="*.html")
    disc_loader = JSONLoader("data/discourse/tds_kb_posts.json", jq_schema=".[] | .posts[]", text_content=False)

    docs = course_loader.load() + disc_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    vectordb.save_local("rag/index")
