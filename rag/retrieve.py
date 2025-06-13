# rag/retrieve.py
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def get_answer(query: str) -> str:
    # Load FAISS vector store
    db = FAISS.load_local("rag/index", embeddings=None, allow_dangerous_deserialization=True)

    # Use local LLM (e.g., mistral) via Ollama
    llm = Ollama(model="mistral")

    # Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    return qa_chain.run(query)
