import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DynamicBatcher:
    def __init__(self, model, tokenizer, batcher: asyncio.Queue, max_batch=16, max_wait_ms=20):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.batcher = batcher

    async def run(self):
        while True:
            try:
                first_item = await self.batcher.get()
            except asyncio.CancelledError:
                break

            batch = [first_item]
            deadline = asyncio.get_event_loop().time() + (self.max_wait_ms / 1000.0)

            while len(batch) < self.max_batch:
                remaining_time = deadline - asyncio.get_event_loop().time()
                if remaining_time <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self.batcher.get(), timeout=remaining_time)
                    batch.append(next_item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    break

            try:
                await self.process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                for item in batch:
                    future = item[1]
                    if not future.done():
                        future.set_exception(e)

    def seperating_lists(self, batch):
        texts = [item[0].text for item in batch]
        ids = [item[2] for item in batch]
        futures = [item[1] for item in batch]
        return texts, ids, futures

    async def process_batch(self, batch):
        texts, ids, futures = self.seperating_lists(batch)
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).tolist()
            
            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result(predictions[i])
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)
