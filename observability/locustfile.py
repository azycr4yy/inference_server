from locust import HttpUser, task, between

class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", json={
            "text": "This movie was absolutely amazing",
            "data_id": 1
        })
    