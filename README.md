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


Here's the full README in markdown — ready to copy-paste:

```markdown
# Nexa AI — IUB Intelligent Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for students of The Islamia University of Bahawalpur (IUB), providing instant answers on scholarships, admissions, transportation, and general university queries.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-blue)
![Pinecone](https://img.shields.io/badge/Pinecone-latest-blue)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange)
![FYP](https://img.shields.io/badge/FYP-Session_2022--2026-green)

---

## About the project

Nexa AI is a university-specific intelligent chatbot that lets IUB students ask natural-language questions about the IUB E-Portal and get precise, direct answers — without navigating through multiple portal pages.

The system is built on a full **RAG pipeline**: university knowledge (scholarships, transport schedules, admission procedures, FAQs) is chunked, embedded, and stored in Pinecone. At runtime, the student's query is embedded, the most relevant chunks are retrieved, and a Groq-hosted LLM synthesizes a focused, contextual response — guided by a structured system prompt that ensures the LLM answers only what was asked.

Developed as a Final Year Project at the Department of Computer Science, IUB — Session Fall 2022–2026.

---

## Key features

- 🎓 **Scholarship inquiry** — Lists available scholarships, eligibility criteria, and application procedures per query intent.
- 🚌 **Transport assistance** — Bus routes, departure times, nearest alternatives, and pickup/drop points with route metadata.
- 📋 **Admission guidance** — NAT registration, merit list process, required documents, and fee installment steps.
- 💬 **General queries** — Departments, contact numbers, university personnel, library timings, WiFi, and E-portal help.
- 🧠 **Intent-aware prompt** — Structured system prompt routes listing, yes/no, timing, procedure, and eligibility questions to the right response shape.
- 📚 **RAG pipeline** — LangChain + Pinecone (MMR retrieval, k=3) with rich metadata tagging per chunk for filtered, precise retrieval.
- ⚡ **Real-time responses** — Groq inference (Llama 3.3 70B Versatile) delivers sub-second LLM responses via the Groq API.
- 🗄️ **SQLite fast-path** — Department contacts and hardcoded FAQs are served instantly from SQLite before hitting the RAG pipeline.

---

## Tech stack

| Layer | Technology | Version / Notes |
|---|---|---|
| Frontend | HTML, CSS, JavaScript | Vanilla — no framework |
| Backend | [Flask](https://flask.palletsprojects.com/) | 3.x · Python web framework |
| GenAI Orchestration | [LangChain](https://docs.langchain.com/) | langchain-core, langchain-community, langchain-classic |
| LLM | [Groq API](https://console.groq.com/docs) · [Llama 3.3 70B Versatile](https://huggingface.co/meta-llama/Llama-3.3-70B-Versatile) | langchain-groq · temperature=0 |
| Vector Database | [Pinecone](https://docs.pinecone.io/) | langchain-pinecone · index: iub-chatbot · Dense · us-east-1 |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | langchain-huggingface · HuggingFaceEmbeddings · 384-dim |
| Data Storage | SQLite3 | Departments · hardcoded FAQs · fast-path lookup |
| Environment | python-dotenv | .env for API keys |
| Dev Environment | Python venv · Git | Windows / PowerShell |

---

## System architecture

Three-tier flow: the student's browser sends a POST request to Flask, which routes through a SQLite fast-path first, then falls through to the full LangChain RAG pipeline on Pinecone.

```
┌─────────────────────────────────────────────┐
│            Student browser                   │
│         (HTML / CSS / JavaScript)            │
└──────────────────┬──────────────────────────┘
                   │ POST /get_response
┌──────────────────▼──────────────────────────┐
│           Flask route handler                │
└──────────┬───────────────────┬──────────────┘
           │ keyword match     │ fallthrough
┌──────────▼──────────┐  ┌────▼─────────────────────────┐
│  SQLite fast-path   │  │  LangChain RetrievalQA        │
│  Depts · FAQs       │  │  chain_type=stuff · MMR k=3   │
└─────────────────────┘  └────┬──────────────┬───────────┘
                              │ embed query  │ generate
                    ┌─────────▼──────┐  ┌────▼──────────────┐
                    │    Pinecone    │  │    Groq API        │
                    │  iub-chatbot   │  │  Llama 3.3 70B     │
                    │  Dense 384-dim │  │  temp=0            │
                    └────────────────┘  └───────────────────┘
                         ▲
              ┌──────────┴──────────┐
              │ HuggingFace Embeds  │
              │  all-MiniLM-L6-v2  │
              └─────────────────────┘
```

---

## GenAI & RAG pipeline

Nexa AI uses a full Retrieval-Augmented Generation pipeline. Here is how data flows from raw university content to a student answer.

### 1. Document ingestion & chunking
University data (scholarships, transport schedule, admissions, FAQs) is structured into QA-pair chunks. Each chunk is formatted as `Q: {question}\n\nA: {answer}` so the embedding captures both the user query signal and the answer content. Data is pre-structured into meaningful semantic units to avoid mid-answer splits.

### 2. Metadata tagging
Each chunk is enriched with structured metadata before indexing: `source`, `type` (faq / transport), `category` (scholarship / admission / hostel / academics / transport / library / general), `tags` (comma-separated keywords), `question` (original Q for citation display), and a deterministic `chunk_id` (UUID5) to prevent duplicate upserts.

### 3. Embedding & indexing into Pinecone
Chunks are embedded using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dimensional dense vectors) via `HuggingFaceEmbeddings` from [langchain-huggingface](https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub). Vectors plus metadata are upserted into the `iub-chatbot` Pinecone index (Dense, us-east-1) using [langchain-pinecone](https://python.langchain.com/docs/integrations/vectorstores/pinecone/).

### 4. Query retrieval (MMR)
At runtime the user's message is embedded with the same model. The retriever uses **Maximal Marginal Relevance (MMR)** with `k=3` to return the most relevant non-redundant chunks. MMR balances similarity to the query with diversity of results, preventing three near-identical chunks from flooding the context.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
```

### 5. Response generation (Groq + Llama 3.3 70B)
Retrieved chunks are injected into a structured `PromptTemplate` alongside the student's question. The prompt contains intent-routing rules that tell the LLM to answer only what was asked — list queries return only names, yes/no queries start with YES/NO, timing queries suggest nearest alternatives if no exact match. The LLM is served via [Groq](https://console.groq.com/) with `temperature=0` for deterministic answers.

### 6. SQLite fast-path (pre-RAG)
Before hitting the RAG pipeline, the Flask route does a keyword match against department names and static FAQs stored in SQLite. Matched queries return instantly without an LLM call — saving latency and Groq API tokens for simple lookups like "accounts department contact number".

---

## RAG pipeline diagram

```
┌──────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Student  │───▶│    Embed     │───▶│    Pinecone    │───▶│     Prompt      │───▶│    Groq LLM     │
│  query   │    │ MiniLM-L6-v2│    │  MMR · k=3     │    │ context+question│    │  Llama 3.3 70B  │
└──────────┘    └──────────────┘    └────────────────┘    └─────────────────┘    └────────┬────────┘
     ▲                                      ▲                                              │
     └──────────────────────────────────────┼──────────────────────────────────────────────┘
                                            │              focused answer
                              ┌─────────────┴──────────────────────────────────┐
                              │  Scholarships · Transport · Admissions · FAQs  │
                              └────────────────────────────────────────────────┘
```

---

## Project structure

```
Nexa-AI/
├── app.py                            # Flask entry point · routes · QA chain · SQLite fast-path
├── pinecone_data_upload_script.py    # FAQ chunking · metadata tagging · Pinecone upsert
├── templates/
│   └── index.html                   # Chat UI
├── static/
│   ├── style.css
│   └── script.js
├── university.db                    # SQLite — departments · static FAQs (auto-created)
├── .env                             # API keys (not committed)
├── .env.example                     # Template for env vars
├── requirements.txt                 # Pinned dependencies
└── README.md
```

---

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/AdilFaraz-ML/Nexa-AI.git
cd Nexa-AI

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Fill in PINECONE_API_KEY and GROQ_API_KEY in .env

# 5. Upload knowledge data to Pinecone (run once)
python pinecone_data_upload_script.py

# 6. Start the Flask app
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Environment variables

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | Your Pinecone project API key |
| `GROQ_API_KEY` | Groq Console API key for Llama 3.3 70B |

The Pinecone index name (`iub-chatbot`) and embedding model (`all-MiniLM-L6-v2`) are hardcoded in `app.py`. Update them there if you change either.

---

## Functional requirements

| ID | Feature | Status |
|---|---|---|
| FR-1 | Scholarship inquiry — list, eligibility, application procedure per query intent | ✅ Implemented |
| FR-2 | Transport assistance — route timings, nearest alternative if no exact match | ✅ Implemented |
| FR-3 | Admission guidance — NAT, merit list, documents, fee installment | ✅ Implemented |
| FR-4 | General queries — library, WiFi, student card, LMS, contact numbers | ✅ Implemented |
| FR-5 | Intent-aware response shaping via structured system prompt | ✅ Implemented |
| FR-6 | Real-time output with sub-second Groq inference | ✅ Implemented |
| FR-7 | SQLite fast-path for department contacts and static FAQs | ✅ Implemented |

---

## Current limitations

- Text-based input only — no voice support
- English language only — no Urdu or multilingual support
- No live connection to the IUB E-Portal database — data is static and manually curated
- Response accuracy depends on completeness of the indexed knowledge base
- No user session or conversation memory — each query is stateless

---

## Future enhancements

- Urdu language support
- Voice command input via Web Speech API
- Live integration with IUB E-Portal database (pending university approval)
- Conversation memory for multi-turn context
- Admin panel for updating the knowledge base without redeployment
- Personalized responses based on student level (BS / MPhil / PhD)
- Mobile app version

---

## Author & supervisor

**Author — Adil Faraz**
BS Computer Science — Session Fall 2022–2026
Department of Computer Science, The Islamia University of Bahawalpur
[github.com/AdilFaraz-ML](https://github.com/AdilFaraz-ML) · [LinkedIn](https://linkedin.com/in/adil-faraz-b3407a271)

**Supervisor — Dr. Muhammad Omar**
Department of Computer Science & IT, The Islamia University of Bahawalpur

---

## References

1. [LangChain Documentation](https://docs.langchain.com/) — GenAI orchestration framework
2. [Pinecone Documentation](https://docs.pinecone.io/) — Vector database and similarity search
3. [Groq API Documentation](https://console.groq.com/docs) — LLM inference platform
4. [all-MiniLM-L6-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Sentence embedding model
5. [Flask Documentation](https://flask.palletsprojects.com/) — Python web framework
6. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020.*
7. Schwaber, K.; Beedle, M. *Agile Software Development with Scrum.* Pearson, 2001.

---

*Made for IUB students — Nexa AI © 2026*
```
