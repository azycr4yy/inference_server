import httpx, asyncio

async def test():
    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(*[
            client.post("http://localhost:8000/predict",
                        json={"text": f"this movie was great", "data_id": i})
            for i in range(40)
        ])
        for r in responses:
            print(r.json())

asyncio.run(test())