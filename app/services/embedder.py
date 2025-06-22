import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "BAAI/bge-large-en-v1.5"  # This model is publicly available and works
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = requests.post(HF_API_URL, headers=headers, json={
        "inputs": texts,
        "options": {
            "wait_for_model": True  # Important: lets Hugging Face load the model if needed
        }
    })
    response.raise_for_status()
    return response.json()
