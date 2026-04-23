"""
Nexa AI — FAQ Chunking & Pinecone Upload Script
This script processes a structured list of FAQs (questions + answers) about Islamia University of Bahawalpur (IUB),
builds LangChain Documents with relevant metadata, and uploads them to an existing Pinecone index for use in the Nexa AI chatbot.
Make sure to set  PINECONE_API_KEY in the .env file before running this script.
"""

import os
import uuid
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# FAQ's Data  (Q + A pairs, structured)

FAQ_DATA = [
    {
        "question": "How to apply for university hostel?",
        "answer": (
            "To apply for university hostel accommodation:\n\n"
            "If you are a 1st semester student:\n"
            "- Visit the Abbasia Campus at the time of your admission.\n"
            "- Apply for hostel accommodation directly from the Admission Department.\n"
            "- Submit: Paid fee challan of Rs. 20,000 (per semester), Attested affidavit, CNIC copy.\n"
            "- Upload all required documents on the IUB E-portal.\n\n"
            "If you are a 2nd semester or above student:\n"
            "- Visit the Executive Hostel at Baghdad-ul-Jadeed (BJC) Campus in person.\n"
            "- Submit a physical application with: Paid fee challan of Rs. 20,000, Attested affidavit, CNIC copy.\n"
            "- Upload all required documents on the IUB E-portal.\n"
            "- Accommodation is allotted subject to availability."
        ),
        "category": "hostel",
        "tags": ["hostel", "accommodation", "apply", "1st semester", "BJC", "Abbasia", "fee challan"],
    },
    {
        "question": "How to apply for a scholarship from the portal?",
        "answer": (
            "To apply for a scholarship through the IUB E-portal:\n"
            "1. Sign in to your My-IUB account.\n"
            "2. Navigate to the Scholarship tab from the menu.\n"
            "3. Fill in the required information and upload the necessary documents.\n"
            "4. Submit your application.\n\n"
            "If shortlisted, the department publishes a list of selected candidates. "
            "You may be called for an interview, after which scholarships are awarded at varying percentages "
            "(e.g., 30%, 70%, or 100%). If awarded, the scholarship amount will reflect on your portal and "
            "will be deducted from your fee challan in the following semester."
        ),
        "category": "scholarship",
        "tags": ["scholarship", "portal", "My-IUB", "apply", "interview", "fee deduction"],
    },
    {
        "question": "Which scholarships are available for BS students?",
        "answer": (
            "Scholarships available for BS students:\n\n"
            "Departmental Scholarships (Faculty of Computing):\n"
            "- Needy Fee Remission\n"
            "- Kinship (sibling currently enrolled at IUB)\n"
            "- Hafiz-e-Quran\n"
            "- Position Holder\n"
            "- Orphan\n\n"
            "External / HEC Scholarships:\n"
            "- EHSAAS / BISP Scholarship (HEC & BISP)\n"
            "- HEC Need-Based Scholarship (HEC)\n"
            "- PEEF Scholarship (Punjab Government)\n"
            "- HEC-USAID MNBSP (HEC)\n"
            "- National Endowment Scholarship for Talented — NEST (Ministry of Planning)\n"
            "- Pakistan Scottish Scholarships Scheme (British Council)\n"
            "- Diya Scholarship (Private)\n"
            "- Rehmat-ul-lil-Alameen (Punjab Government)\n"
            "- Karwan-e-Ilm Scholarship (Private)\n"
            "- Gilgit Baltistan (GB) Scholarship (HEC)\n"
            "- Honhaar Scholarship (PHEC)"
        ),
        "category": "scholarship",
        "tags": ["scholarship", "BS", "list", "EHSAAS", "HEC", "PEEF", "Honhaar", "Kinship", "departmental"],
    },
    {
        "question": "How to apply for a scholarship? What is the application procedure?",
        "answer": (
            "Application procedure varies by scholarship type:\n\n"
            "Departmental Scholarships (Kinship, Needy Fee Remission, Orphan, Hafiz-e-Quran, Position Holder):\n"
            "- Collect and fill out the application form in hard copy.\n"
            "- Attach required documents including an income certificate.\n"
            "- Submit the form in person to the designated scholarship coordinator in your department.\n\n"
            "E-portal / HEC / Private Scholarships (Karwan-e-Ilm, Diya, Honhaar, PEEF):\n"
            "- Log in to My-IUB and navigate to the Scholarship section.\n"
            "- Fill in personal details and upload required documents.\n"
            "- If shortlisted, submit physical copies along with income certificate to the department.\n\n"
            "International Scholarships (US, UK, Scotland, etc.):\n"
            "- The department or official university social media will share the application link.\n"
            "- Visit the link and fill in your information directly on the respective scholarship website."
        ),
        "category": "scholarship",
        "tags": ["scholarship", "apply", "procedure", "departmental", "HEC", "international", "income certificate"],
    },
    {
        "question": "What is the eligibility criteria for scholarships?",
        "answer": (
            "Eligibility criteria for scholarships:\n\n"
            "Departmental Scholarships:\n"
            "- Needy Fee Remission: Family income below Rs. 40,000 | Paid challan | CNIC | Attested income certificate\n"
            "- Orphan: Father's death certificate | Paid challan | CNIC\n"
            "- Kinship: Both siblings enrolled at IUB | 2 paid challan copies | 2 Matric degree copies | 2 CNIC copies\n"
            "- Hafiz-e-Quran: Sanad (Hafiz-e-Quran degree) | Paid challan | CNIC\n"
            "- Position Holder: Next semester challan copy | Controller-signed transcript | CNIC\n\n"
            "External / HEC Scholarships (income limits):\n"
            "- EHSAAS / BISP: Rs. 45,000\n"
            "- HEC Need-Based: Rs. 60,000\n"
            "- PEEF (Bachelor & Master): Rs. 60,000\n"
            "- HEC-USAID MNBSP: Rs. 60,000\n"
            "- Rehmat-ul-lil-Alameen: Rs. 60,000\n"
            "- Honhaar Scholarship Phase-I: Rs. 350,000\n"
            "- NEST, Pakistan Scottish, Diya, Karwan-e-Ilm, Gilgit Baltistan: As per policy"
        ),
        "category": "scholarship",
        "tags": ["scholarship", "eligibility", "criteria", "income limit", "documents", "Needy", "Kinship", "Orphan"],
    },
    {
        "question": "How to apply for clearance for 8th semester students?",
        "answer": (
            "Students who have completed all degree requirements must complete the clearance process.\n\n"
            "Step 1 — Generate and Pay Challans:\n"
            "- Log in to IUB E-portal → Generate Online Challan.\n"
            "- Search 'Alumni' and generate Alumni Challan (Rs. 500).\n"
            "- Search 'Degree', select Normal Degree Fee challan (Rs. 5,000).\n"
            "- Pay via HBL App, JazzCash, Women Bank, or any HBL branch.\n\n"
            "Step 2 — Upload Documents & Apply for Clearance:\n"
            "- Go to My-IUB portal → Menu → My Documents.\n"
            "- Upload scanned documents in JPG/PNG format.\n"
            "- Go to My Clearance → Apply for Clearance.\n"
            "- Enter remarks as 'Degree Completed' and submit.\n"
            "- Wait 24–48 hours for IT Department verification.\n\n"
            "Step 3 — Obtain Physical Signatures (in order):\n"
            "Library Office → Security Office (Main Gate) → Alumni Office (Taabish Alwari Building) "
            "→ Chairman (Department Office) → Accounts Branch (Old / Abbasia Campus)\n\n"
            "Step 4 — Submit & Collect Result Card:\n"
            "- Submit signed clearance form to the Department Office.\n"
            "- Collect your result card in person with your clearance form.\n"
            "Note: Only the concerned student can collect their own result card."
        ),
        "category": "clearance",
        "tags": ["clearance", "8th semester", "degree", "result card", "challan", "signatures", "alumni"],
    },
    {
        "question": "If I score less in course improvement than before, which marks will be considered?",
        "answer": (
            "If you attempt a course improvement and score lower than your previous attempt, "
            "your original (higher) marks will be retained. The new lower marks will not replace the previous result."
        ),
        "category": "academics",
        "tags": ["course improvement", "marks", "grade", "result", "lower score"],
    },
    {
        "question": "What is Nexa AI?",
        "answer": (
            "Nexa AI is the AI-powered chatbot of Islamia University of Bahawalpur (IUB). "
            "It answers queries about transport, scholarships, admissions, general queries, and more. "
            "It is developed by Adil Faraz as a Final Year Project (FYP) at IUB."
        ),
        "category": "general",
        "tags": ["Nexa AI", "chatbot", "FYP", "IUB", "Adil Faraz"],
    },
    {
        "question": "How can I obtain a library card?",
        "answer": (
            "To obtain a library card at IUB, visit the Sir Sadiq Library in person. "
            "Present your official student ID card for registration. "
            "After registration, a library card will be issued granting access to digital and physical resources. "
            "In case of loss or damage to any borrowed book, you must compensate per university library policies."
        ),
        "category": "library",
        "tags": ["library", "library card", "Sir Sadiq Library", "student ID"],
    },
    {
        "question": "What are the university transport timings?",
        "answer": (
            "University transport schedule:\n"
            "- Buses run from 7:30 AM to 6:30 PM.\n"
            "- Buses arrive at intervals of 30 minutes to 1 hour.\n"
            "- Timings may vary during examination periods."
        ),
        "category": "transport",
        "tags": ["transport", "bus", "timings", "schedule", "7:30 AM", "6:30 PM"],
    },
    {
        "question": "How can I contact IUB administration?",
        "answer": (
            "You can contact IUB at +92-62-9250235 or visit the main campus at Bahawalpur."
        ),
        "category": "general",
        "tags": ["contact", "IUB", "administration", "phone number", "Bahawalpur"],
    },
    {
        "question": "Which documents are required for admission?",
        "answer": (
            "Documents required for IUB admission:\n"
            "- Verified result cards of Matric (10th) and Intermediate (12th).\n"
            "- CNIC or B-Form.\n"
            "- Recent passport-size photographs.\n"
            "- Any additional documents required by the university.\n"
            "All documents must be submitted with an affidavit at the IUB Old Campus for verification."
        ),
        "category": "admission",
        "tags": ["admission", "documents", "Matric", "Intermediate", "CNIC", "affidavit", "Old Campus"],
    },
    {
        "question": "How can I download my transcript?",
        "answer": (
            "To download your transcript:\n"
            "1. Sign in to your My-IUB account.\n"
            "2. Navigate to the Transcript tab on the homepage.\n"
            "3. Click on the Download PDF button."
        ),
        "category": "academics",
        "tags": ["transcript", "download", "My-IUB", "PDF"],
    },
    {
        "question": "What is LMS?",
        "answer": (
            "The Learning Management System (LMS) is an online platform provided by IUB through its E-portal. "
            "Instructors use it to upload lectures, assignments, and course materials. "
            "Students submit assignments through this platform within given deadlines. "
            "LMS is accessible through the My-IUB portal."
        ),
        "category": "general",
        "tags": ["LMS", "Learning Management System", "assignments", "E-portal", "My-IUB"],
    },
    {
        "question": "Does Nexa AI work 24/7?",
        "answer": (
            "Yes, Nexa AI is available around the clock. "
            "For urgent administrative matters, we recommend contacting university staff directly during office hours."
        ),
        "category": "general",
        "tags": ["Nexa AI", "availability", "24/7", "office hours"],
    },
    {
        "question": "Which campuses does IUB have?",
        "answer": (
            "IUB operates multiple campuses including:\n"
            "- Baghdad-ul-Jadeed Campus (BJC)\n"
            "- Abbasia Campus\n"
            "- Khawaja Fareed Campus (KH)\n"
            "- Rahim Yar Khan Campus"
        ),
        "category": "general",
        "tags": ["campus", "BJC", "Abbasia", "Khawaja Fareed", "Rahim Yar Khan", "IUB campuses"],
    },
    {
        "question": "How can I download my student card?",
        "answer": (
            "To download your student card:\n"
            "1. Log in to your My-IUB account.\n"
            "2. Navigate to the My Student Card tab.\n"
            "3. Click on the option to download your student card.\n"
            "If you encounter issues, contact the IT Department of IUB."
        ),
        "category": "general",
        "tags": ["student card", "ID card", "download", "My-IUB", "IT Department"],
    },
    {
        "question": "How to create an account on IUB E-portal?",
        "answer": (
            "To create an account on the IUB E-portal:\n"
            "1. Open your browser and search for IUB E-portal.\n"
            "2. Open the official portal website.\n"
            "3. Fill in required information: Name, CNIC, Address, Educational details, Contact information.\n"
            "4. Submit the registration form.\n"
            "5. Sign in using your credentials after registration.\n"
            "You can then apply for the NAT (National Aptitude Test) through the portal.\n"
            "For issues, contact the IT Department of IUB."
        ),
        "category": "admission",
        "tags": ["E-portal", "account", "register", "My-IUB", "NAT", "IT Department"],
    },
    {
        "question": "How can I apply for course repeat or improvement?",
        "answer": (
            "To apply for a course repeat or improvement:\n"
            "1. Log in to your My-IUB account.\n"
            "2. Navigate to the Course Repeat Challan tab.\n"
            "3. Select Course Repeat or Course Improvement.\n"
            "4. Choose the subject(s) from your semester list.\n"
            "5. Download the fee challan.\n"
            "6. Pay the required fee.\n"
            "7. Submit the paid challan copy to the relevant clerk.\n"
            "After submission, you will be assigned to a class for the selected course.\n"
            "Note: Course repeat fee is generally higher than course improvement fee."
        ),
        "category": "academics",
        "tags": ["course repeat", "course improvement", "challan", "My-IUB", "fee", "clerk"],
    },
    {
        "question": "Is full payment required after getting name in a merit list?",
        "answer": (
            "Yes, once your name appears in the IUB merit list, you are required to pay the full admission fee initially. "
            "Afterward, semester fees may be paid in two installments, subject to university policy."
        ),
        "category": "admission",
        "tags": ["merit list", "admission fee", "payment", "installment"],
    },
    {
        "question": "How can I apply for fee installments?",
        "answer": (
            "To apply for fee installment:\n"
            "1. Log in to your My-IUB account.\n"
            "2. Navigate to the Financial Voucher tab.\n"
            "3. Select the challan for which you want installments.\n"
            "4. Click on the Installment Option (highlighted in orange).\n"
            "5. Submit your request.\n"
            "Note: This facility is available for a limited time each semester."
        ),
        "category": "admission",
        "tags": ["fee", "installment", "Financial Voucher", "challan", "My-IUB"],
    },
    {
        "question": "Who is the Vice Chancellor of IUB?",
        "answer": (
            "Dr. Muhammad Kamran is the current Vice Chancellor of The Islamia University of Bahawalpur (IUB). "
            "He can be contacted via email at: vc@iub.edu.pk"
        ),
        "category": "general",
        "tags": ["Vice Chancellor", "VC", "Dr. Muhammad Kamran", "IUB", "contact"],
    },
    {
        "question": "What are the library timings?",
        "answer": (
            "The Sir Sadiq Library is generally open from 9:00 AM to 5:00 PM on weekdays."
        ),
        "category": "library",
        "tags": ["library", "timings", "Sir Sadiq Library", "hours", "weekdays"],
    },
    {
        "question": "When are spring and fall admissions announced?",
        "answer": (
            "IUB typically announces admissions as follows:\n"
            "- Fall Admissions: Announced in June/July (classes usually begin in October).\n"
            "- Spring Admissions: Announced in November/December (classes usually begin in January/February).\n"
            "Applications are submitted online through the official IUB E-portal."
        ),
        "category": "admission",
        "tags": ["admission", "fall", "spring", "announcement", "schedule", "E-portal"],
    },
    {
        "question": "How many programs and departments are available in IUB?",
        "answer": (
            "IUB offers a wide range of academic programs across approximately 300 disciplines, "
            "delivered through 135 teaching departments. "
            "These include undergraduate (BS), postgraduate (MS/MPhil), and PhD levels, "
            "covering fields such as arts, sciences, engineering, and professional studies."
        ),
        "category": "general",
        "tags": ["programs", "departments", "BS", "MS", "PhD", "IUB", "disciplines"],
    },
    {
        "question": "How can I apply for NAT?",
        "answer": (
            "To apply for the NAT:\n"
            "1. Register on the IUB E-portal (if you don't have an account).\n"
            "2. Log in to your account.\n"
            "3. Navigate to the NAT application section.\n"
            "4. Fill out the required details.\n"
            "5. Generate and download the test challan (approximately Rs. 1,000).\n"
            "6. Pay the challan fee at the designated bank.\n"
            "7. Upload the paid challan receipt on the portal.\n"
            "After verification, you will receive a roll number slip and test date."
        ),
        "category": "admission",
        "tags": ["NAT", "National Aptitude Test", "apply", "challan", "roll number slip", "E-portal"],
    },
    {
        "question": "What should I do if someone is harassing me?",
        "answer": (
            "If someone is harassing you via messages or phone calls, you can:\n"
            "- Contact your class CR (Class Representative).\n"
            "- Reach out directly to the Student Affairs Director "
            "(for female students, this is typically a female official).\n"
            "- Contact your department coordinator."
        ),
        "category": "general",
        "tags": ["harassment", "student affairs", "CR", "department coordinator", "female students"],
    },
    {
        "question": "How to connect to IUB WiFi?",
        "answer": (
            "Two ways to connect to IUB WiFi:\n\n"
            "For Students:\n"
            "- Connect to 'IUB Smart University' WiFi network.\n"
            "- Enter username as rollno@iub.edu.pk and your My-IUB password.\n\n"
            "For Guests:\n"
            "- Connect to 'IUB Guest' WiFi network.\n"
            "- When prompted for a password, enter 'IUB GUEST' (uppercase or lowercase both work)."
        ),
        "category": "general",
        "tags": ["WiFi", "internet", "IUB Smart University", "IUB Guest", "connect", "password"],
    },
]


# Build LangChain Documents with rich metadata

def build_documents(faq_data: list[dict]) -> list[Document]:
    """
    Each chunk = Question + Answer combined as page_content.
    Metadata carries category, tags, source, and a stable chunk_id.
    """
    docs = []
    for item in faq_data:
        # Q+A combined — improves semantic match for user queries
        page_content = f"Q: {item['question']}\n\nA: {item['answer']}"

        metadata = {
            "source": "iub_faq",
            "category": item["category"],
            "tags": ", ".join(item["tags"]),   # Pinecone metadata must be str/int/float/bool/list
            "question": item["question"],       # stored for easy display in citations
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, item["question"])),
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs



# UPSERT Data to Pinecone

def upload_to_pinecone(docs: list[Document]):
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    INDEX_NAME = "iub-chatbot"   # your existing index

    print(f"Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print(f"Connecting to Pinecone index: {INDEX_NAME}")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    print(f"Upserting {len(docs)} FAQ chunks...")
    vector_store = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )

    print(f" Done. {len(docs)} chunks uploaded to '{INDEX_NAME}'.")
    return vector_store


# OPTIONAL: Preview Chunks (Just for dry run)
def preview_chunks(docs: list[Document], n: int = 3):
    print(f"\n── PREVIEW: first {n} chunks ──\n")
    for doc in docs[:n]:
        print(f"[category: {doc.metadata['category']}]")
        print(f"[tags: {doc.metadata['tags']}]")
        print(f"[chunk_id: {doc.metadata['chunk_id']}]")
        print(doc.page_content[:300])
        print("─" * 60)



# Main execution of chunks building and uploading

if __name__ == "__main__":
    docs = build_documents(FAQ_DATA)

    # Preview before uploading (comment out when ready)
    preview_chunks(docs, n=3)

    # Uncomment to upload
    upload_to_pinecone(docs)

    print(f"\nTotal chunks ready: {len(docs)}")
    print("Categories:", set(d.metadata["category"] for d in docs))
