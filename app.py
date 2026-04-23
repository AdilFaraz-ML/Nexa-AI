import os
import sqlite3
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)

# Same embedding model — must match what you used in vector_store.py
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Pinecone vectorstore
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="iub-chatbot",
    embedding=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# System Prompt

PROMPT_TEMPLATE = """
You are Nexa AI, the official AI assistant of Islamia University of Bahawalpur (IUB).
Your job is to answer student queries accurately and concisely using ONLY the context below.
Never make up information. Never paste context as-is.

━━━ RESPONSE RULES ━━━

1. LISTING QUESTIONS ("which", "what are", "how many", "list"):
   - Return ONLY the count (if asked) and names.
   - Do NOT include eligibility, procedure, or details unless the student explicitly asks.
   - Example: "How many scholarships for BS?" → "There are 5 departmental scholarships: Needy Fee Remission, Kinship, Hafiz-e-Quran, Position Holder, and Orphan."

2. YES/NO QUESTIONS ("is there", "are admissions open", "is any bus available"):
   - Always start with YES or NO.
   - Then add only the directly relevant detail (date, time, route).
   - Do NOT elaborate beyond what was asked.
   - Example: "Are admissions open?" → "Yes, Fall admissions are open. Last date to apply is [date]."

3. BUS / TRANSPORT TIMING QUESTIONS:
   - If a bus exists at the requested time and route → say YES and give that timing.
   - If NO exact match → say NO, then suggest the nearest available timing (before or after).
   - Example: "Is there a bus at 3 PM from BJC to AC?" → "No bus at exactly 3:00 PM on that route. The nearest option is at 3:30 PM / 2:30 PM."

4. PROCEDURE QUESTIONS ("how to", "how can I", "steps to"):
   - Give a clean numbered step-by-step answer.
   - Keep each step short and actionable.

5. ELIGIBILITY QUESTIONS ("eligibility", "criteria", "who can apply", "requirements"):
   - Return ONLY the eligibility criteria for what was asked.
   - Do not mix in the application procedure.

6. UNKNOWN / OUT OF SCOPE:
   - If the answer is not in the context, respond:
     "I don't have that information right now. Please contact IUB official website or E-portal."

━━━ STRICT RULES ━━━
- NEVER dump the full chunk as a response.
- NEVER answer in more detail than the question requires.
- ALWAYS respond in plain natural language, not raw text from context.
- If the student asks a follow-up (e.g., "what is the eligibility for that?"), answer only the follow-up.

━━━ CONTEXT ━━━
{context}

━━━ STUDENT QUESTION ━━━
{question}

━━━ YOUR ANSWER ━━━
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)


# Database Setup

def init_db():
    if not os.path.exists("university.db"):
        conn = sqlite3.connect("university.db")
        c = conn.cursor()

        c.execute("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department_name TEXT,
                contact_number TEXT,
                email TEXT,
                location TEXT
            )
        """)

        c.execute("""
            CREATE TABLE faqs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                category TEXT
            )
        """)

        c.execute("""
            INSERT INTO departments (department_name, contact_number, email, location)
            VALUES ('Accounts Department', '+92-62-9250123', 'accounts@iub.edu.pk', 'Admin Block')
        """)

        c.execute("""
            INSERT INTO departments (department_name, contact_number, email, location)
            VALUES ('Admission Office', '+92-62-9250456', 'admissions@iub.edu.pk', 'Main Campus')
        """)

        c.execute("""
            INSERT INTO faqs (question, answer, category)
            VALUES ('What is the last date for fee submission?', 'The last date for fee submission is 10th November 2025.', 'Finance')
        """)

        c.execute("""
            INSERT INTO faqs (question, answer, category)
            VALUES ('How can I reset my portal password?', 'Go to "Forgot Password" on the portal login page.', 'Portal Help')
        """)

        conn.commit()
        conn.close()
        print("Database initialized")

init_db()

# Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_msg = request.json.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please enter a message."})

    # 1. Department keyword match
    conn = sqlite3.connect("university.db")
    c = conn.cursor()
    c.execute("SELECT department_name, contact_number FROM departments")
    for dept, contact in c.fetchall():
        if dept.lower() in user_msg.lower():
            conn.close()
            return jsonify({"reply": f"{dept} contact number is {contact}."})

    # 2. FAQ keyword match
    c.execute("SELECT question, answer FROM faqs")
    for question, answer in c.fetchall():
        if question.lower() in user_msg.lower():
            conn.close()
            return jsonify({"reply": answer})

    conn.close()

    # 3. LangChain RAG — Pinecone + Groq
    try:
        response = qa_chain.invoke({"query": user_msg})
        return jsonify({"reply": response["result"]})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"reply": "Something went wrong. Please try again."})


if __name__ == "__main__":
    app.run(debug=True)