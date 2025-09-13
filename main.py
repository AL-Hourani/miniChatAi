# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from model.generate_answer import safe_generate_answer, allowed_keywords
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_HF = "jaafar-ai/miniChat"
model: Optional[AutoModelForSeq2SeqLM] = None
tokenizer: Optional[AutoTokenizer] = None

app = FastAPI(title="Mini Chat API")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    global model, tokenizer
    try:
        if model is None or tokenizer is None:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_HF , torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)

        answer = safe_generate_answer(model, tokenizer, request.question, allowed_keywords)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
