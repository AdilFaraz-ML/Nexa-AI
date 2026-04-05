import os
import json
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "iub-chatbot"
index = pc.Index(index_name)

# Load JSON data
with open("transport_schedule.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load embedding model (384 dim)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

vectors = []

for i, item in enumerate(data):
    text = item["content"]

    # Add abbreviations + context
    page_content = f"""
Transport Schedule Info:
{text}

Route Abbreviations:
AC = Abbasia Campus
KH = Khawaja Fareed Campus
FC = Faculty Campus
BIC = Baghdad-ul-Jadeed Campus
"""

    # Extract metadata
    metadata = {
        "type": "transport",
        "source": "transport_schedule"
    }

    # --- Route detection ---
    if "AC to KH" in text:
        metadata["route"] = "AC-KH"
    elif "AC to BIC" in text:
        metadata["route"] = "AC-BIC"
    elif "BJC to AC" in text:
        metadata["route"] = "BJC-AC"
    elif "KH to AC" in text:
        metadata["route"] = "KH-AC"

    # --- Time extraction ---
    time_match = re.findall(r'\d{1,2}:\d{2}\s?(AM|PM)', text)
    if time_match:
        metadata["time"] = " ".join(time_match)

    # --- Category ---
    if "morning" in text.lower():
        metadata["category"] = "morning"
    elif "afternoon" in text.lower():
        metadata["category"] = "afternoon"
    elif "evening" in text.lower():
        metadata["category"] = "evening"

    # --- Extra helpful fields ---
    if "SB1" in text or "SB2" in text:
        metadata["special_bus"] = True

    # Generate embedding using enriched text
    embedding = model.encode(page_content).tolist()

    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {
            "text": page_content,   
            **metadata
        }
    })

# Upload to Pinecone
index.upsert(vectors=vectors)

print("Data uploaded successfully with metadata + context!")