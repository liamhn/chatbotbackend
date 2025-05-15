from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import faiss
import numpy as np
import os

# Load Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_SPI_KEY")

app = FastAPI()

# Enable CORS so your frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

# ----- Document Setup -----
DOC_URLS = [
    "https://raw.githubusercontent.com/liamhn/chatbotbackend/main/resume1text.txt"
    #"https://raw.githubusercontent.com/yourusername/yourrepo/main/docs/doc2.md"
]

documents = [requests.get(url).text for url in DOC_URLS]

# Local embedding model (MiniLM)
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed docs
doc_embeddings = embed_model.encode(documents).astype("float32")
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# ----- Chat Route -----
@app.post("/chat")
async def chat(message: Message):
    user_input = message.message

    # Embed query and search
    query_embedding = embed_model.encode([user_input]).astype("float32")
    D, I = index.search(query_embedding, k=2)
    context = "\n---\n".join([documents[i] for i in I[0]])

    # Send to Groq
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mixtral-8x7b-32768",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that uses the context provided to answer questions accurately."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {user_input}"
                }
            ]
        }
    )

    result = response.json()
    return {"reply": result["choices"][0]["message"]["content"]}
