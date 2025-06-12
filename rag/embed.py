# rag/embed.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.environ["AIPIPE_TOKEN"] = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDE5MTVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.FdixSrNdglNv2jOA78f6NDwH8BxXLMmP2tYpehHhVn4"

def create_vector_store():
    course_loader = DirectoryLoader("data/course", glob="*.html")
    disc_loader = JSONLoader("data/discourse/tds_kb_posts.json", jq_schema=".[] | .posts[]", text_content=False)

    docs = course_loader.load() + disc_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    vectordb.save_local("rag/index")
