import torch 

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = r"C:\Users\Udayan\Desktop\udayan\vscode\inference_server\models\saved_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

prompt = "I didnt like the movie."

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax().item()

print(f"Predicted class: {model.config.id2label[predicted_class_id]}")