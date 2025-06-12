
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.retrieve import get_answer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    image: str | None = None

@app.post("/api/")
@app.post("/")  # ✅ NEW: this allows POST to root URL
def answer(query: Query):
    final_question = query.question
    answer = get_answer(final_question)
    return {
        "answer": answer,
        "links": []
    }

@app.get("/")  # Optional, helps for browser test
def root():
    return {"message": "TDS Virtual TA API is running."}

# ✅ Optional: define a GET / route to help debugging
@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running."}
