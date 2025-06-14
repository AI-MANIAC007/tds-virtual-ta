# rag/retrieve.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

def get_answer(query: str) -> str:
    # Load local embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index with local embedding model
    db = FAISS.load_local("rag/index", embeddings=embedding_model, allow_dangerous_deserialization=True)

    # Use local LLM (Mistral via Ollama running on host machine)
    llm = OllamaLLM(
        model="mistral",
        base_url="http://host.docker.internal:11434"  # required when running inside Docker
    )

    # Set up RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    return qa_chain.invoke({"query": query})["result"]


