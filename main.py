from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

MODEL_HF = "jaafar-ai/miniChat"  
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_HF}"

HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

app = FastAPI(title="Mini Chat API")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

def query_hf_api(question: str):
    payload = {"inputs": question}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
 
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    return str(result)

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        answer = query_hf_api(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
