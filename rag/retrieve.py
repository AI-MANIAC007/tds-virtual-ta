# rag/retrieve.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

os.environ["AIPROXY_TOKEN"] = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDE5MTVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.yyL_AkgLzYF6sRRlwrAPZHWRpDX6qvHdBRFisQVRaf4"

def get_answer(query):
    db = FAISS.load_local("rag/index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=db.as_retriever())
    return qa.run(query)
