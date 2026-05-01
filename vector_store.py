import os
import json
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# SIMPLE MERGED RETRIEVER (no external import)
class SimpleMergedRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        all_docs = []
        seen = set()
        for r in self.retrievers:
            for doc in r.invoke(query):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)
        return all_docs


# METADATA EXTRACTOR
def extract_metadata(text, source_name):
    metadata = {
        "type": "transport",
        "source": source_name
    }

    if "AC to KH" in text or "AC to KH and FC" in text:
        metadata["route"] = "AC-KH-FC"
    elif "AC to BJC" in text:
        metadata["route"] = "AC-BJC"
    elif "BJC to AC" in text:
        metadata["route"] = "BJC-AC"
    elif "KH and FC to AC" in text or "KH to AC" in text:
        metadata["route"] = "KH-FC-AC"

    time_match = re.findall(r'\d{1,2}:\d{2}\s?(?:AM|PM)', text)
    if time_match:
        metadata["timings"] = ", ".join(time_match)

    if "morning" in text.lower():
        metadata["shift"] = "morning"
    elif "afternoon" in text.lower():
        metadata["shift"] = "afternoon"
    elif "evening" in text.lower():
        metadata["shift"] = "evening"
    elif "all-day" in text.lower() or "complete" in text.lower():
        metadata["shift"] = "all"

    metadata["day"] = "saturday" if "saturday" in text.lower() else "weekday"

    if "SB1" in text or "SB2" in text:
        metadata["special_bus"] = True

    return metadata


# LOAD TRANSPORT JSON → Documents
def load_transport_json(filepath: str, source_name: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        text = item["content"]

        if "metadata" in item:
            metadata = item["metadata"]
            metadata["source"] = source_name
        else:
            metadata = extract_metadata(text, source_name)

        enriched = f"""{text}

Route Abbreviations: AC = Abbasia Campus | KH or KH.FC = Khawaja Fareed Campus | BJC = Baghdad Campus | FC = Faculty Campus | S after bus number = Staff colony bus | SB = School Bus"""

        docs.append(Document(
            page_content=enriched,
            metadata=metadata
        ))

    print(f"Loaded {len(docs)} documents from {filepath}")
    return docs


# UPLOAD TRANSPORT DATA → iub-transport
def build_transport_index():
    """Run this once to upload transport_schedule.json to iub-transport index"""

    transport_files = [
        ("transport_schedule.json", "transport_schedule"),
    ]

    all_docs = []
    for filepath, source_name in transport_files:
        docs = load_transport_json(filepath, source_name)
        all_docs.extend(docs)

    # NO splitting — each entry has full route timings in one chunk
    print(f"Uploading {len(all_docs)} documents to iub-transport index...")

    PineconeVectorStore.from_documents(
        documents=all_docs,
        embedding=embeddings,
        index_name="iub-transport-data"
    )

    print("Transport data uploaded to iub-transport successfully.")


# MERGED RETRIEVER — both indexes at once
def get_merged_retriever() -> SimpleMergedRetriever:
    """
    Returns a single retriever that queries both indexes simultaneously.
    Import and use this in app.py.
    """

    vectorstore_general = PineconeVectorStore.from_existing_index(
        index_name="iub-chatbot",
        embedding=embeddings
    )

    vectorstore_transport = PineconeVectorStore.from_existing_index(
        index_name="iub-transport-data",
        embedding=embeddings
    )

    retriever_general = vectorstore_general.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    retriever_transport = vectorstore_transport.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    print("[VectorStore] Merged retriever ready (iub-chatbot + iub-transport)")
    return SimpleMergedRetriever([retriever_general, retriever_transport])


# MAIN — run to upload transport data
if __name__ == "__main__":
    build_transport_index()