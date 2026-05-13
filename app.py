import os
import re
import json
import sqlite3
import threading
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from vector_store import get_merged_retriever

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nexa-secret-123")


# LLM + MERGED RETRIEVER
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

merged_retriever = get_merged_retriever()


# GREETINGS
GREETINGS = [
    "hi", "hello", "hey", "salam", "assalam", "assalamualaikum",
    "good morning", "good afternoon", "good evening",
    "howdy", "whats up", "what's up", "greetings"
]


# SESSION HISTORY STORE
MAX_SESSIONS = 500
session_histories = {}

def get_history(session_id: str) -> list:
    if len(session_histories) > MAX_SESSIONS:
        keys = list(session_histories.keys())
        for k in keys[:MAX_SESSIONS // 2]:
            del session_histories[k]
    if session_id not in session_histories:
        session_histories[session_id] = []
    return session_histories[session_id]

def append_history(session_id: str, user_msg: str, ai_reply: str):
    history = get_history(session_id)
    history.append(HumanMessage(content=user_msg))
    history.append(AIMessage(content=ai_reply))
    if len(history) > 10:
        session_histories[session_id] = history[-10:]


# STEP 1+2 COMBINED — REWRITE + CLASSIFY IN ONE GROQ CALL
REWRITER_ROUTER_PROMPT = """
You are a query processor for Nexa AI, the chatbot of Islamia University of Bahawalpur (IUB).

You have TWO jobs — do both in ONE response:

JOB 1 — REWRITE:
Rewrite the user's latest message into a complete, standalone search query
using conversation history to resolve pronouns and references.
If already clear, return as-is.

JOB 2 — CLASSIFY:
Classify the rewritten query into exactly one category:
- IUB: anything about IUB (admissions, transport, fees, departments, hostel, exams, scholarships, portal, campus)
- EDUCATION: general education/career advice, degree comparisons — NOT specific to IUB
- OUT_OF_SCOPE: weather, cricket, jokes, cooking, politics — nothing to do with IUB or education

OUTPUT FORMAT (strict JSON, nothing else, no markdown, no explanation):
{"rewritten": "<rewritten query here>", "category": "IUB"}

Examples:
History: User asked about bus at 2 PM, bot said nearest is 3 PM
User: "before it"
Output: {"rewritten": "bus timing before 3 PM from BJC to AC route", "category": "IUB"}

User: "what is scope of BSCS?"
Output: {"rewritten": "what is scope of BSCS?", "category": "EDUCATION"}

User: "tell me a joke"
Output: {"rewritten": "tell me a joke", "category": "OUT_OF_SCOPE"}
"""

def rewrite_and_classify(user_msg: str, history: list) -> tuple:
    """Single Groq call: returns (rewritten_query, category)"""
    try:
        messages = [SystemMessage(content=REWRITER_ROUTER_PROMPT)]
        if history:
            messages.extend(history[-6:])
        messages.append(HumanMessage(content=f"User message: {user_msg}"))
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        rewritten = parsed.get("rewritten", user_msg)
        category = parsed.get("category", "IUB").upper()
        if category not in ["IUB", "EDUCATION", "OUT_OF_SCOPE"]:
            category = "IUB"
        print(f"[REWRITER] '{user_msg}' → '{rewritten}'")
        print(f"[ROUTER] Category: {category}")
        return rewritten, category
    except Exception as e:
        print(f"Rewrite+classify error: {e}")
        return user_msg, "IUB"


# STEP 3A — IUB RAG ANSWER (with parallel retrieval)
IUB_SYSTEM_PROMPT = """
You are Nexa AI, the official AI assistant of Islamia University of Bahawalpur (IUB).
Answer student queries using the retrieved context below as your primary source.
Use the conversation history to understand the full picture.

━━━ WHEN TO USE CONTEXT ONLY (strict) ━━━
For these query types, answer ONLY from context — no assumptions:
- Bus/transport timings and routes
- Fee amounts, deadlines, last dates
- Admission status (open/closed) — NEVER say "admissions are open" unless context explicitly says so
- Scholarship amounts or eligibility criteria
- Exam schedules or results

━━━ WHEN YOU CAN ADD HELPFUL CONTEXT (flexible) ━━━
For these query types, use context as base but you may add brief helpful explanation:
- "What is LMS?" → explain what LMS is generally + mention IUB uses it
- "What is NAT?" → explain what NAT is + how it relates to IUB admissions
- "What is a transcript?" → explain + mention how to get it at IUB
- "How does hostel work?" → explain generally + give IUB-specific steps from context
- Procedural or explanatory questions where a little background helps the student understand

━━━ RESPONSE RULES ━━━

1. LISTING QUESTIONS ("which", "what are", "how many", "list"):
   - Return count + names only. No extra detail unless asked.

2. YES/NO QUESTIONS:
   - Start with YES or NO, then only the relevant detail.

3. ADMISSION QUESTIONS:
   - NEVER say admissions are open unless context explicitly confirms it.
   - If context gives a schedule, return the schedule only.
   - Example: "Fall admissions are announced in June/July, Spring in November/December."

4. BUS / TRANSPORT TIMING:
   - Exact match → YES + timing.
   - No exact match → NO + nearest timing before and after.
   - For "before it" → timing just before the last mentioned time.
   - For "after it"  → timing just after the last mentioned time.

5. PROCEDURE QUESTIONS ("how to", "steps to", "how can I"):
   - Give numbered steps, short and actionable.
   - You may add a brief one-line explanation if it helps the student understand why.

6. EXPLANATORY QUESTIONS ("what is", "explain", "tell me about"):
   - Use context as the base.
   - You may add 1-2 sentences of general helpful background if context is thin.
   - Keep it concise and student-friendly.

7. UNKNOWN / NOT IN CONTEXT:
   - "I don't have that information right now. Please contact IUB official website or E-portal."
   - Do NOT guess or make up facts like dates, amounts, or names.

━━━ STRICT RULES ━━━
- NEVER dump raw context.
- NEVER fabricate specific facts (dates, fees, names, timings).
- NEVER say admissions are open/closed unless context confirms it.
- Respond in plain, friendly, student-appropriate language.

━━━ RETRIEVED CONTEXT ━━━
{context}
"""

def answer_iub_question(rewritten_query: str, original_msg: str, session_id: str) -> str:
    try:
        history = get_history(session_id)

        # Run Pinecone retrieval in a background thread
        docs_result = []
        def fetch_docs():
            docs_result.extend(merged_retriever.invoke(rewritten_query))

        retrieval_thread = threading.Thread(target=fetch_docs)
        retrieval_thread.start()

        # While retrieval runs, build the message list (CPU only, no API call)
        messages_base = []
        messages_base.extend(history[-10:])
        messages_base.append(HumanMessage(content=original_msg))

        # Wait for retrieval to finish
        retrieval_thread.join(timeout=10)

        # Deduplicate docs
        seen = set()
        unique_docs = []
        for doc in docs_result:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        context = "\n\n".join(doc.page_content for doc in unique_docs)
        if not context.strip():
            context = "No relevant information found in the knowledge base."

        messages = [SystemMessage(content=IUB_SYSTEM_PROMPT.format(context=context))]
        messages.extend(messages_base)

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        print(f"IUB answer error: {e}")
        return "Something went wrong. Please try again."


# STEP 3B — EDUCATION DIRECT ANSWER
EDUCATION_SYSTEM_PROMPT = """
You are Nexa AI, the assistant of Islamia University of Bahawalpur (IUB).
Answer general education and career advice questions from your own knowledge.
You may mention IUB programs if relevant, but don't fabricate IUB-specific details.
Keep answers concise, student-friendly, and practical.
"""

def answer_education_question(rewritten_query: str, original_msg: str, session_id: str) -> str:
    try:
        history = get_history(session_id)
        messages = [SystemMessage(content=EDUCATION_SYSTEM_PROMPT)]
        messages.extend(history[-10:])
        messages.append(HumanMessage(content=original_msg))
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Education answer error: {e}")
        return "I couldn't process that. Please try again."


# DATABASE
def init_db():
    if not os.path.exists("university.db"):
        conn = sqlite3.connect("university.db")
        c = conn.cursor()
        c.execute("""CREATE TABLE departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_name TEXT, contact_number TEXT,
            email TEXT, location TEXT)""")
        c.execute("""CREATE TABLE faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT, answer TEXT, category TEXT)""")
        c.execute("INSERT INTO departments VALUES (NULL,'Accounts Department','+92-62-9250123','accounts@iub.edu.pk','Admin Block')")
        c.execute("INSERT INTO departments VALUES (NULL,'Admission Office','+92-62-9250456','admissions@iub.edu.pk','Main Campus')")
        c.execute("INSERT INTO faqs VALUES (NULL,'What is the last date for fee submission?','The last date for fee submission is 10th November 2025.','Finance')")
        c.execute("INSERT INTO faqs VALUES (NULL,'How can I reset my portal password?','Go to \"Forgot Password\" on the portal login page.','Portal Help')")
        conn.commit()
        conn.close()

init_db()


# ROUTES
OUT_OF_SCOPE_REPLY = (
    "I'm Nexa AI, designed specifically to assist with IUB-related queries and general education guidance. "
    "Feel free to ask me about admissions, courses, transport, scholarships, or career advice!"
)

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = os.urandom(16).hex()
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "Please enter a message."})

    if "session_id" not in session:
        session["session_id"] = os.urandom(16).hex()
    session_id = session["session_id"]

    # 1. Greeting check
    if any(user_msg.lower().strip("!.,?") == g for g in GREETINGS):
        reply = "Hello! I'm Nexa AI, the official assistant of Islamia University of Bahawalpur. How can I help you today? Feel free to ask me about admissions, courses, transport, scholarships, or career advice!"
        append_history(session_id, user_msg, reply)
        return jsonify({"reply": reply})

    # 2. Department keyword match
    conn = sqlite3.connect("university.db")
    c = conn.cursor()
    c.execute("SELECT department_name, contact_number FROM departments")
    for dept, contact in c.fetchall():
        if dept.lower() in user_msg.lower():
            conn.close()
            reply = f"{dept} contact number is {contact}."
            append_history(session_id, user_msg, reply)
            return jsonify({"reply": reply})

    # 3. FAQ keyword match
    c.execute("SELECT question, answer FROM faqs")
    for question, answer in c.fetchall():
        if question.lower() in user_msg.lower():
            conn.close()
            append_history(session_id, user_msg, answer)
            return jsonify({"reply": answer})
    conn.close()

    # 4. Rewrite + Classify in ONE Groq call
    history = get_history(session_id)
    rewritten, category = rewrite_and_classify(user_msg, history)

    # 5. Route and answer
    if category == "OUT_OF_SCOPE":
        return jsonify({"reply": OUT_OF_SCOPE_REPLY})

    elif category == "EDUCATION":
        reply = answer_education_question(rewritten, user_msg, session_id)
        append_history(session_id, user_msg, reply)
        return jsonify({"reply": reply})

    else:  # IUB
        reply = answer_iub_question(rewritten, user_msg, session_id)
        append_history(session_id, user_msg, reply)
        return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)