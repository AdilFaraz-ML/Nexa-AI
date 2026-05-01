import os
import sqlite3
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

# ─────────────────────────────────────────
# LLM + MERGED RETRIEVER
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

merged_retriever = get_merged_retriever()

# ─────────────────────────────────────────
# GREETINGS
# ─────────────────────────────────────────
GREETINGS = [
    "hi", "hello", "hey", "salam", "assalam", "assalamualaikum",
    "good morning", "good afternoon", "good evening",
    "howdy", "whats up", "what's up", "greetings"
]

# ─────────────────────────────────────────
# SESSION HISTORY STORE
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# STEP 1 — QUERY REWRITER
# ─────────────────────────────────────────
REWRITER_PROMPT = """
You are a query rewriter for a university chatbot.

Your job: Given the conversation history and the user's latest message,
rewrite the latest message into a COMPLETE, STANDALONE search query
that can be understood WITHOUT any prior context.

Rules:
- Resolve all pronouns and references ("it", "that", "before it", "after it", "the previous one")
- Use the history to figure out what "it" refers to
- Output ONLY the rewritten query — no explanation, no quotes, nothing else

Examples:
History: User asked "is there a bus at 2 PM from BJC to AC?", Bot said "No bus at 2 PM, nearest is 3 PM"
User says: "before it"
Rewritten: "bus timing before 3 PM from BJC to AC route"

History: User asked "what scholarships are available for BS students?"
User says: "what is the eligibility for that?"
Rewritten: "eligibility criteria for BS student scholarships at IUB"

History: User asked "Should I do BSCS or SE?"
User says: "which has more scope?"
Rewritten: "which has more job scope BSCS or Software Engineering"

If the message is already clear and standalone, return it as-is.
"""

def rewrite_query(user_msg: str, history: list) -> str:
    if not history:
        return user_msg
    try:
        messages = [SystemMessage(content=REWRITER_PROMPT)]
        messages.extend(history[-6:])
        messages.append(HumanMessage(content=f"Rewrite this: {user_msg}"))
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        print(f"[REWRITER] '{user_msg}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"Rewriter error: {e}")
        return user_msg

# ─────────────────────────────────────────
# STEP 2 — QUESTION ROUTER
# ─────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """
You are a query classifier for Nexa AI, the chatbot of Islamia University of Bahawalpur (IUB).

Classify the user's question into exactly ONE of these three categories:

1. IUB — Question is specifically about IUB: admissions, fee, transport, departments,
   scholarships, hostel, timetable, exams, faculty, campus, portal, courses at IUB, etc.

2. EDUCATION — Question is about general education, career advice, degree comparisons,
   academic fields, study tips — but NOT specific to IUB.

3. OUT_OF_SCOPE — Nothing to do with IUB or education.
   Examples: weather, cricket, jokes, poems, cooking, politics.

Reply with ONLY one word: IUB, EDUCATION, or OUT_OF_SCOPE.
No explanation. No punctuation. Just the category word.
"""

def classify_question(rewritten_query: str) -> str:
    try:
        response = llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=rewritten_query)
        ])
        category = response.content.strip().upper()
        if category not in ["IUB", "EDUCATION", "OUT_OF_SCOPE"]:
            return "IUB"
        return category
    except Exception:
        return "IUB"

# ─────────────────────────────────────────
# STEP 3A — IUB RAG ANSWER
# ─────────────────────────────────────────
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

        docs = merged_retriever.invoke(rewritten_query)

        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        context = "\n\n".join(doc.page_content for doc in unique_docs)

        if not context.strip():
            context = "No relevant information found in the knowledge base."

        messages = [SystemMessage(content=IUB_SYSTEM_PROMPT.format(context=context))]
        messages.extend(history[-10:])
        messages.append(HumanMessage(content=original_msg))

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        print(f"IUB answer error: {e}")
        return "Something went wrong. Please try again."

# ─────────────────────────────────────────
# STEP 3B — EDUCATION DIRECT ANSWER
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
# ✅ Fix 2 — removed "I'm not able to help with that topic."
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

    # 1. Greeting check ✅ Fix 1
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

    # 4. Rewrite vague follow-ups into full standalone queries
    history = get_history(session_id)
    rewritten = rewrite_query(user_msg, history)

    # 5. Classify
    category = classify_question(rewritten)
    print(f"[ROUTER] Category: {category} | Original: '{user_msg}' | Rewritten: '{rewritten}'")

    # 6. Route and answer
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