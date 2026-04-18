import os
import json
import re
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Same embedding model you were using
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def extract_metadata(text):
    metadata = {
        "type": "transport",
        "source": "transport_schedule"
    }

    # Route detection
    if "AC to KH" in text:
        metadata["route"] = "AC-KH"
    elif "AC to BIC" in text:
        metadata["route"] = "AC-BIC"
    elif "BJC to AC" in text:
        metadata["route"] = "BJC-AC"
    elif "KH to AC" in text:
        metadata["route"] = "KH-AC"

    # Time extraction
    time_match = re.findall(r'\d{1,2}:\d{2}\s?(AM|PM)', text)
    if time_match:
        metadata["time"] = " ".join(time_match)

    # Category
    if "morning" in text.lower():
        metadata["category"] = "morning"
    elif "afternoon" in text.lower():
        metadata["category"] = "afternoon"
    elif "evening" in text.lower():
        metadata["category"] = "evening"

    # Special bus
    if "SB1" in text or "SB2" in text:
        metadata["special_bus"] = True

    return metadata


def load_json_file(filepath, source_name):
    """Load any JSON file and convert to Documents"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        text = item["content"]

        enriched = f"""
Transport Schedule Info:
{text}

Route Abbreviations:
AC = Abbasia Campus
KH = Khawaja Fareed Campus
FC = Faculty Campus
BIC = Baghdad-ul-Jadeed Campus
"""
        metadata = extract_metadata(text)
        metadata["source"] = source_name

        docs.append(Document(
            page_content=enriched,
            metadata=metadata
        ))

    return docs


def build_vectorstore():
    all_docs = []

    # Add all your JSON files here
    json_files = [
        ("transport_schedule.json", "transport_schedule"),
        # ("another_file.json", "another_source"),  # add more here
    ]

    for filepath, source_name in json_files:
        docs = load_json_file(filepath, source_name)
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} items from {filepath}")

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")

    # Embed + push to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="iub-chatbot"
    )

    print("All data uploaded to Pinecone successfully.")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()