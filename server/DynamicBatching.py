import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DynamicBatcher:
    def __init__(self, model, tokenizer,batcher : asyncio.Queue, max_batch=16, max_wait_ms=20):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.batcher = batcher
    async def add(self):
        batch = []
        first_item = await self.batcher.get()
        batch.append(first_item)
        deadline = asyncio.get_event_loop().time() + 0.02
        while(True):
            remaining_time = deadline - asyncio.get_event_loop().time()  
            if(remaining_time <= 0 or len(batch) >= self.max_batch):
                break
            try:
                next_item = await asyncio.wait_for(batcher.get(),timeout=remaining_time)
                batch.append(next_item)
            except asyncio.TimeoutError:
                break
    def seperating_lists(self,batch):
        texts = [item[0].text for item in batch]
        ids = [item[0].id for item in batch]
        futures = [item[1] for item in batch]
        return texts, ids, futures
    async def process_batch(self,batch):
        texts, ids, futures = self.seperating_lists(batch)
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
        for i, future in enumerate(futures):
            if not future.cancelled():
                future.set_result(predictions[i])
