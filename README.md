# ML Inference Server

**1. What the project is**
A production-grade ML inference server built from scratch that serves a fine-tuned DistilBERT model with async dynamic batching, observability, and graceful degradation under load. It solves the problem of efficiently serving deep learning models by maximizing GPU utilization and maintaining predictable latency without needing heavy external tools.

---

**2. Architecture diagram**
```text
Client Request
      ↓
FastAPI /predict
      ↓
asyncio.Queue (bounded, maxsize=100)
      ↓
DynamicBatcher (collects requests for 20ms or until batch=16)
      ↓
Single GPU forward pass (DistilBERT)
      ↓
asyncio.Future resolved → response returned to client
      ↓
Prometheus scrapes /metrics every 5s → Grafana dashboard
```

---

**3. Model details**
```text
Model:     distilbert-base-uncased
Task:      Sentiment Analysis (SST-2)
F1 Score:  88.6%
Training:  3 epochs, lr=2e-5, batch_size=16
Hardware:  RTX 4050
```

---

**4. Key engineering decisions**
- **Dynamic batching (20ms window, max_batch=16)**
  Why: The GPU completes its forward pass so fast that processing requests one-by-one leaves the GPU mostly idle. A 20ms window allows enough time to gather requests under traffic spikes. `max_batch=16` is used because it optimizes compute speed within the hardware memory constraints without causing latency deterioration.

- **Bounded queue (maxsize=100)**
  Why: Provides predictable behavior and graceful degradation under load (backpressure). An unbounded queue would let requests pile up indefinitely, leading to memory saturation and extremely high latency. When full, the system actively drops incoming load (`503 Service Unavailable`), maintaining healthy operation for already queued requests.

- **`asyncio.Future` vs Celery/BackgroundTasks**
  Why: We need the client connection to be held open and immediately return the prediction response without adding the overhead of a separate worker layer or a message broker (Redis/RabbitMQ). `asyncio.Future` natively integrates exactly where we need it: waiting asynchronously inside the request handler while a background loop processes the batch.

- **`put_nowait` vs `put`**
  Why: Immediate rejection over blocking. Using `put_nowait` guarantees *Fail-Fast* behavior. If the queue is at capacity, the client is instantly served a 503 rather than holding open a connection while blocking the main thread, which would bottleneck the ASGI server.

---

**5. Benchmark results**
```text
Test config: 50 concurrent users, 60s duration, RTX 4050

Metric              Sequential    Batched    
────────────────────────────────────────────
Throughput (req/s)  126.2         139.9      
p50 latency         35ms          13ms       
p95 latency         59ms          34ms       
Failures            0             0          
```

---

**6. Observability**
- **`request_count`**: Total successful predictions (rate = QPS).
- **`request_latency`**: p95 end-to-end latency histogram.
- **`batch_size`**: Distribution of requests per batch.
- **`queue_depth`**: Current queue size (saturation signal).

![Grafana Dashboard Screenshot](<img width="1600" height="744" alt="image" src="https://github.com/user-attachments/assets/fc76da8d-68bc-45b9-b00f-87e4c29f8d52" />
) 

---

**7. Setup instructions**
```bash
# clone and install
git clone https://github.com/azycr4yy/inference_server.git
cd inference_server
pip install -r requirements.txt

# start full stack
docker-compose up

# run server
cd server
uvicorn app:app --host 0.0.0.0 --port 8000

# example request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing", "data_id": 1}'
```

---

**8. Project structure**
```text
inference_server/
├── models/
│   └── saved_model/          # Fine-tuned DistilBERT artifacts
├── observability/
│   ├── locustfile.py         # Load testing script
│   └── *.csv                 # Benchmark load test results
├── server/
│   ├── app.py                # FastAPI web server and metrics exposing
│   ├── DynamicBatching.py    # Async dynamic batching queue implementation
│   ├── model.py              # Pydantic data schemas
│   ├── docker-compose.yml    # Prometheus service definition
│   └── prometheus-config.yml # Prom scrapers
├── tests/                    # Unit and integration tests
├── requirements.txt
└── README.md
```
