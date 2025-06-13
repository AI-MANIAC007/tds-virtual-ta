# main.py
from fastapi import FastAPI, UploadFile, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag.retrieve import get_answer

app = FastAPI()

# Allow all origins for CORS (safe for public APIs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your domain(s) if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint for health check / platform validation
@app.api_route("/", methods=["GET", "POST", "HEAD"])
async def root(request: Request):
    return {"message": "TDS Virtual TA API is live."}

# Query schema
class Query(BaseModel):
    question: str
    image: str | None = None

# POST endpoint to get answers
@app.post("/api/")
def answer(query: Query):
    final_question = query.question
    answer = get_answer(final_question)

    return {
        "answer": answer,
        "links": []  # You can later add parsed links here
    }

