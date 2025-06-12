# main.py
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from rag.retrieve import get_answer

app = FastAPI()

class Query(BaseModel):
    question: str
    image: str | None = None

@app.post("/api/")
def answer(query: Query):
    # (Optional) Handle image with OCR if needed
    final_question = query.question
    answer = get_answer(final_question)

    return {
        "answer": answer,
        "links": []  # Optional: parse links from RAG results
    }
