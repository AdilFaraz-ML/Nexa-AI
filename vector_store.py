import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone (Serverless)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "iub-chatbot"
index = pc.Index(index_name)

# Load JSON data
with open("transport_schedule.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load embedding model (384 dim)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Prepare vectors
vectors = []

for i, item in enumerate(data):
    text = item["content"]
    embedding = model.encode(text).tolist()

    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {"text": text}
    })

# Upload to Pinecone
index.upsert(vectors=vectors)

print("Data uploaded successfully!")