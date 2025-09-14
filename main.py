# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client


app = FastAPI(title="Mini Chat API")
app.add_middleware (
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

HF_SPACE = "jaafar-ai/miniChatModel"


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


client = Client(HF_SPACE)

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
       
        response = client.predict(request.question, fn_index=0)
        return AnswerResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
