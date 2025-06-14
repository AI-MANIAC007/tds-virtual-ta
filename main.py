from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag.retrieve import get_answer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the schema
class Query(BaseModel):
    question: str
    image: str | None = None

# Accept POST at both / and /api/ for evaluator compatibility
@app.post("/")
@app.post("/api/")
async def answer(query: Query):
    final_question = query.question
    result = get_answer(final_question)
    return {"answer": result, "links": []}

# Optional health check for GET/HEAD
@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return {"message": "TDS Virtual TA API is live."}

