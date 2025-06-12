# rag/retrieve.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = "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"

def get_answer(query):
    db = FAISS.load_local("rag/index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=db.as_retriever())
    return qa.run(query)
