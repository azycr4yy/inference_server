from fastapi import FastAPI , HTTPException 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
from DynamicBatching import DynamicBatcher

model = None
tokenizer = None

class InputData(BaseModel):
    text: str
    id: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    BASE_PATH = Path(__file__).parent
    MODEL_PATH = BASE_PATH / "models" / "saved_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    yield
    del model
    del tokenizer
    model = None
    tokenizer = None

app = FastAPI(lifespan=lifespan)
batcher = asyncio.Queue(maxsize=100)


@app.post("/predict",response_model=InputData)
async def predict(data:InputData):
    future = asyncio.get_event_loop().create_future()
    try:
        batcher.put_nowait((data, future))
    except Exception as e:
        raise HTTPException(status_code=503,detail=str(e))
    result = await asyncio.wait_for(future,timeout=20.0)
    return result
    

@app.get("/health")
async def health():
    return {"status": "ok",
    "model_loaded": model is not None}

