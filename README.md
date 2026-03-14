# Nexa-AI — IUB E-Portal Intelligent Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Version-3.0-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/LangChain-Enabled-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Pinecone-Vector_DB-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NLP-Transformers-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-In_Development-red?style=for-the-badge" />
</p>

<p align="center">
  A Generative AI–powered chatbot built for students of <strong>The Islamia University of Bahawalpur (IUB)</strong> to quickly access information about the IUB E-Portal — covering scholarships, admissions, transportation, and general queries.
</p>

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [GenAI Pipeline](#genai-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Functional Requirements](#functional-requirements)
- [Non-Functional Requirements](#non-functional-requirements)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [Supervisor](#supervisor)
- [References](#references)

---

## 📖 About the Project

**Nexa-AI** is a university-specific intelligent chatbot designed to assist IUB students by answering queries related to the IUB E-Portal in real time. Instead of navigating through multiple portal pages, students can simply ask questions and receive accurate, direct answers — along with relevant page links.

> **Example:** A student asks: *"Where can I find my transcript?"*  
> Nexa-AI responds: *"Your transcript is available on Page 5 of the E-Portal under Academic Records. [Direct Link]"*

The system uses **LangChain**, **Pinecone**, and **Generative AI** techniques, including chunking, indexing, and metadata tagging, to build a powerful knowledge retrieval pipeline over IUB's official data.

Developed as a **Final Year Project** at the Department of Computer Science, IUB — Session Fall 2022–2026.

---

## ✨ Key Features

- 🎓 **Scholarship Inquiry** — Lists available scholarships, eligibility criteria, and application procedures
- 🚌 **Transportation Assistance** — Provides bus routes, timings, pickup/drop points, and schedule details
- 📋 **Admission Guidance** — Step-by-step support for NAT test registration, merit lists, and required documents
- 💬 **General Queries** — Answers questions about departments, contact numbers, and key university personnel
- 🔗 **E-Portal Page Navigation** — Directs students to exact portal pages with direct links
- ⚡ **Real-Time Responses** — Processes and returns answers within 2 seconds
- 🧠 **Intent Detection** — NLP-based understanding of user query intent using Transformers
- 📚 **RAG Pipeline** — Retrieval-Augmented Generation using LangChain + Pinecone for context-aware answers

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| NLP / ML | Transformers, Scikit-learn, NLTK |
| GenAI Orchestration | LangChain |
| Vector Database | Pinecone |
| Embeddings | Sentence Transformers / OpenAI Embeddings |
| Data Format | JSON, CSV |
| Development Methodology | Agile Scrum |

---

## 🏛️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   User Interaction Layer                  │
│         Web Interface (HTML / CSS / JavaScript)           │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTP Request
┌─────────────────────────▼────────────────────────────────┐
│                    Flask Backend                          │
│              (Route Handler / Session Manager)            │
└──────────┬──────────────────────────────┬────────────────┘
           │                              │
┌──────────▼──────────┐      ┌────────────▼───────────────┐
│   NLP Processing    │      │    LangChain RAG Pipeline   │
│  • Tokenizer        │◄────►│  • Query Embedding          │
│  • Intent Detector  │      │  • Pinecone Vector Search   │
│  • Entity Extractor │      │  • Context Retrieval        │
└──────────┬──────────┘      └────────────┬───────────────┘
           │                              │
┌──────────▼──────────────────────────────▼───────────────┐
│                  Response Generator                       │
│         (ML Models + Retrieved Context Fusion)            │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                      Data Layer                           │
│   Knowledge Base │ Pinecone Index │ Config / FAQs Files  │
└─────────────────────────────────────────────────────────┘
```

---

## 🧬 GenAI Pipeline

Nexa-AI uses a full **Retrieval-Augmented Generation (RAG)** pipeline powered by LangChain and Pinecone. Here's how data flows:

### 1. 📄 Document Ingestion & Chunking
University data (scholarships, transport, admissions, FAQs) is loaded and split into semantic chunks using LangChain's `RecursiveCharacterTextSplitter`.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
```

### 2. 🏷️ Metadata Tagging
Each chunk is enriched with metadata for precise filtering during retrieval.

```python
for chunk in chunks:
    chunk.metadata = {
        "source": "iub_scholarships.pdf",
        "category": "scholarship",
        "page": 3,
        "university": "IUB"
    }
```

### 3. 🗂️ Indexing into Pinecone
Chunks are embedded and stored in a Pinecone vector index for semantic search.

``` python
from langchain. vectorstores import Pinecone
from langchain. embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="nexa-ai-index")
```

### 4. 🔍 Query & Retrieval
At runtime, the user's query is embedded and matched against the Pinecone index to retrieve the most relevant chunks.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.get_relevant_documents(user_query)
```

### 5. 💬 Response Generation
Retrieved context is passed to the ML/NLP response generator to produce a final, grounded answer.

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
response = qa_chain({"query": user_query})
```

---

## 📁 Project Structure

```
nexa-ai/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   ├── app.py                  # Flask entry point
│   ├── routes.py               # API routes
│   ├── session_manager.py      # Conversation context handling
│   └── response_router.py      # Intent → response routing
│
├── nlp/
│   ├── intent_classifier.py    # ML-based intent detection
│   ├── tokenizer.py            # Text preprocessing
│   ├── entity_extractor.py     # NER for query understanding
│   └── nlp_pipeline.py         # Full NLP workflow
│
├── genai/
│   ├── chunker.py              # Document chunking logic
│   ├── metadata_tagger.py      # Metadata enrichment
│   ├── indexer.py              # Pinecone indexing
│   ├── retriever.py            # Vector search & retrieval
│   └── rag_chain.py            # LangChain RAG pipeline
│
├── data/
│   ├── scholarships.json
│   ├── transportation.json
│   ├── admissions.json
│   └── general_faqs.json
│
├── models/
│   ├── intent_model.pkl        # Trained intent classifier
│   └── config.json             # Model hyperparameters
│
├── docs/
│   ├── SRS_Document_of_Nexa_AI.pdf
│   └── SDD_of_Nexa_AI.pdf
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- Pinecone account (free tier works)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/nexa-ai.git
cd nexa-ai

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Fill in your Pinecone API key and other config in .env

# 5. Index your data into Pinecone (run once)
python genai/indexer.py

# 6. Run the Flask backend
python backend/app.py
```

Open your browser and go to `http://localhost:5000`

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=nexa-ai-index
FLASK_SECRET_KEY=your_flask_secret_key
FLASK_DEBUG=True
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 💡 Usage

Once the app is running, open the web interface. You can ask questions like:

| Example Query | Domain |
|---|---|
| `What scholarships are available for BS students?` | Scholarship |
| `What are the bus timings for One Unit stop?` | Transportation |
| `How do I apply for the NAT entrance test?` | Admission |
| `What is the contact number of the Admissions office?` | General |
| `Where can I find my transcript on the portal?` | E-Portal Navigation |

---

## ✅ Functional Requirements

| ID | Feature | Status |
|---|---|---|
| FR-1 | Scholarship Inquiry Handling | ✅ Implemented |
| FR-2 | Transportation Information Assistance | ✅ Implemented |
| FR-3 | Admission Guidance & Support | ✅ Implemented |
| FR-4 | General Query Handling | ✅ Implemented |
| FR-5 | E-Portal Page Navigation Links | ✅ Implemented |
| FR-6 | Real-Time Output Display | ✅ Implemented |
| FR-7 | Conversation Context Maintenance | ✅ Implemented |

---

## 📊 Non-Functional Requirements

| Requirement | Target |
|---|---|
| Response Time | ≤ 2 seconds under normal load |
| Query Success Rate | ≥ 99% without failure |
| Availability | 24/7 uptime |
| Security | HTTPS, no permanent storage of user data |
| Maintainability | Modular architecture, full documentation |

---

## ⚠️ Current Limitations

- Text-based input only (no voice support yet)
- English language only (no multilingual support yet)
- No live connection to the IUB E-Portal database (integration pending approval)
- Covers 4 domains: Scholarships, Transportation, Admissions, General Queries
- Response accuracy depends entirely on the completeness of the training dataset
- No external LLM API integrated — responses are grounded solely in the local knowledge base

---

## 🔮 Future Enhancements

- [ ] Voice command input support
- [ ] Multilingual support (Urdu and other languages)
- [ ] Live integration with IUB E-Portal database (once approved)
- [ ] Admin panel for updating the knowledge base without redeployment
- [ ] Personalized responses based on student profile (BS / MPhil / PhD)
- [ ] Mobile app version

---

## 👨‍💻 Author

**Adil Faraz**  
BS Computer Science — Session Fall 2022–2026  
Department of Computer Science  
The Islamia University of Bahawalpur

---

## 👨‍🏫 Supervisor

**Dr. Muhammad Omar**  
Department of Computer Science & IT  
The Islamia University of Bahawalpur

---

## 📚 References

1. Schwaber, Ken; Beedle, Mike. *Agile Software Development with Scrum*. 1st ed., Pearson, 2001. ISBN 9780130676344.
2. Staffordshire University. *Beacon: Your digital guide*. Retrieved November 23, 2025, from https://www.staffs.ac.uk/students/digital-services/beacon
3. LangChain Documentation. https://docs.langchain.com
4. Pinecone Documentation. https://docs.pinecone.io

---

<p align="center">Made with ❤️ for IUB Students — Nexa-AI © 2026</p>
