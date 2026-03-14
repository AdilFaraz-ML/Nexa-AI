import json
import os
import sqlite3
import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer


# Initialize Flask app
app = Flask(__name__)

# Initialize Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Transport JSON & Create Embeddings
with open("transport_schedule.json", "r", encoding="utf-8") as f:
    transport_data = json.load(f)

transport_texts = [item["content"] for item in transport_data]
transport_embeddings = model.encode(transport_texts, convert_to_numpy=True)

dimension = transport_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(transport_embeddings)

# Database setup
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
            VALUES ('How can I reset my portal password?', 'Go to “Forgot Password” on the portal login page.', 'Portal Help')
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
    user_msg = request.json.get("message", "").lower()

    # Departments
    conn = sqlite3.connect("university.db")            
    c = conn.cursor()
    c.execute("SELECT department_name, contact_number FROM departments")
    for dept, contact in c.fetchall():
        if dept.lower() in user_msg:
            conn.close()
            return jsonify({"reply": f" {dept} contact number is {contact}."})

    # FAQs
    c.execute("SELECT question, answer FROM faqs")
    for question, answer in c.fetchall():
        if question.lower() in user_msg:
            conn.close()
            return jsonify({"reply": answer})

    conn.close()

    # Transport (Embeddings of Transport) 
    query_embedding = model.encode([user_msg], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k=2)

    if distances[0][0] < 1.2: 
        answers = [transport_texts[i] for i in indices[0]]
        return jsonify({"reply": " " + " ".join(answers)})

    # Fallback 
    return jsonify({
        "reply": "I'm still learning. Please ask about admission, fees, or bus schedules details."
    })

# Run App
if __name__ == "__main__":
    app.run(debug=True)
