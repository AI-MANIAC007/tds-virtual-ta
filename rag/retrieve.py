# rag/retrieve.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key)

def get_answer(query):
    db = FAISS.load_local("rag/index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=db.as_retriever())
    return qa.run(query)
