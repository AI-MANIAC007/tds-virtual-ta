# rag/retrieve.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def get_answer(query: str) -> str:
    # Load FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("rag/index", embeddings, allow_dangerous_deserialization=True)

    # Connect to local Ollama model (e.g. mistral)
    llm = Ollama(model="mistral")

    # Build retrieval-based QA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa_chain.run(query)
