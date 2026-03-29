from fastapi import FastAPI , HTTPException 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import requests
import torch
from DynamicBatching import DynamicBatcher
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge 
from prometheus_client import make_asgi_app
#from python_logging_loki import LokiHandler
from model import InputData, OutputData

REQUESTS_IN_QUEUE = Gauge(
    'app_requests_in_queue',
    'Number of requests in queue',
    ['endpoint']
)

REQUESTS_LATENCY = Histogram(
    'app_requests_latency',
    'Latency of requests',
    ['endpoint']
)

REQUESTS_SUCCESS = Counter(
    'app_requests_success',
    'Number of successful requests',
    ['endpoint']
)

REQUESTS_FAILED = Counter(
    'app_requests_failed',
    'Number of failed requests',
    ['endpoint']
)

BATCH_SIZE = Gauge(
    'app_batch_size',
    'Size of batches',
    ['endpoint']
)

EXCEPTION = Counter(
    'app_total_exceptions',
    'Total number of exceptions',
    ['endpoint', 'status_code']
)

model = None
tokenizer = None
batcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, batcher
    MODEL_PATH = Path(__file__).parent.parent / "models" / "saved_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batcher = asyncio.Queue(maxsize=100)
    dynamic_batcher = DynamicBatcher(model, tokenizer, batcher)
    task = asyncio.create_task(dynamic_batcher.run())
    
    yield
    
    task.cancel()
    del model
    del tokenizer
    model = None
    tokenizer = None


app = FastAPI(lifespan=lifespan)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
PORT = 8000
Instrumentator().instrument(app).expose(app)


@app.get("/")
def read_root():
    return {"message": "Server is alive"}

@app.post("/predict", response_model=OutputData)
async def predict(data: InputData):
    if batcher is None:
        REQUESTS_FAILED.labels(endpoint = "/predict").inc()
        EXCEPTION.labels(endpoint = "/predict", status_code = "503").inc()
        raise HTTPException(status_code=503, detail="Server is starting up")
    future = asyncio.get_running_loop().create_future()
    current_input_time = asyncio.get_event_loop().time()
    try:
        batcher.put_nowait((data, future, data.data_id))
        REQUESTS_IN_QUEUE.labels(endpoint="/predict").set(batcher.qsize())
    except asyncio.QueueFull:
        REQUESTS_FAILED.labels(endpoint = "/predict").inc()
        REQUESTS_IN_QUEUE.labels(endpoint="/predict").set(batcher.qsize())
        EXCEPTION.labels(endpoint = "/predict", status_code = "503").inc()
        raise HTTPException(status_code=503, detail="Server is too busy")
    try:
        result = await asyncio.wait_for(future, timeout=20.0)
    except asyncio.TimeoutError:
        future.cancel()
        REQUESTS_FAILED.labels(endpoint = "/predict").inc()
        EXCEPTION.labels(endpoint = "/predict", status_code = "504").inc()
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        REQUESTS_FAILED.labels(endpoint = "/predict").inc()
        EXCEPTION.labels(endpoint = "/predict", status_code = "500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    REQUESTS_SUCCESS.labels(endpoint = "/predict").inc()
    REQUESTS_IN_QUEUE.labels(endpoint="/predict").set(batcher.qsize())
    REQUESTS_LATENCY.labels(endpoint = "/predict").observe(asyncio.get_event_loop().time() - current_input_time)
    return OutputData(data_id=data.data_id, prediction=result)


@app.get("/health")
async def health():
    return {"status": "ok",
    "model_loaded": model is not None}

@app.get("/crash")
async def crash():
    EXCEPTION.labels(endpoint = "/crash", status_code = "500").inc()
    raise HTTPException(status_code=500, detail="Server crashed")
    os._exit(1)
