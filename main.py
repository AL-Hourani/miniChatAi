from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.generate_answer import safe_generate_answer, model, tokenizer, allowed_keywords 


app = FastAPI(title="Mini Chat API")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        ##
        answer = safe_generate_answer(model, tokenizer, request.question, allowed_keywords)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
