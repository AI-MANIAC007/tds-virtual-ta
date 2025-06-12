from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.retrieve import get_answer

app = FastAPI()

# ✅ Add CORS middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    image: str | None = None

@app.post("/api/")
def answer(query: Query):
    final_question = query.question
    answer = get_answer(final_question)
    return {
        "answer": answer,
        "links": []
    }

# ✅ Optional: define a GET / route to help debugging
@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running."}
