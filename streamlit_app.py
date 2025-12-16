import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from datetime import datetime
from fpdf import FPDF

# Load environment variables
load_dotenv()

MODEL = "gemini-2.0-flash"

st.set_page_config(
    page_title="AyurGenix AI - Ayurvedic Medicine Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CSS ------------------
st.markdown("""
<style>
    @media (min-width: 768px) {
        [data-testid="stSidebar"] { min-width: 350px; max-width: 400px; }
    }
    [data-testid="stChatMessage"] {
        max-width: 100%;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ DATASET ------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_excel("AyurGenixAI_Dataset (1).xlsx")
    except Exception as e:
        st.error(e)
        return None

# ------------------ GEMINI ------------------
@st.cache_resource
def init_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

# ------------------ SEARCH ------------------
def search_dataset(df, query):
    if df is None:
        return []
    q = query.lower()
    matches = []
    for _, row in df.iterrows():
        if q in str(row.get("Disease","")).lower() or q in str(row.get("Symptoms","")).lower():
            matches.append(row)
    return matches[:5]

def format_matches_for_context(matches):
    if not matches:
        return "No direct matches found in the Ayurvedic database."
    context = ""
    for m in matches:
        context += f"""
Disease: {m.get('Disease')}
Symptoms: {m.get('Symptoms')}
Ayurvedic Herbs: {m.get('Ayurvedic Herbs')}
Formulation: {m.get('Formulation')}
Doshas: {m.get('Doshas')}
Diet: {m.get('Diet and Lifestyle Recommendations')}
Yoga: {m.get('Yoga & Physical Therapy')}
Prevention: {m.get('Prevention')}
---
"""
    return context

# ------------------ AI RESPONSE ------------------
def generate_response(client, prompt, history):
    contents = []

    for h in history[-6:]:
        role = "user" if h["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(h["content"])]
        ))

    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(prompt)]
    ))

    config = types.GenerateContentConfig(
        system_instruction="""
You are AyurGenix AI, a compassionate Ayurvedic medicine assistant.

MANDATORY RESPONSE ORDER:

1️⃣ 🩺 Possible Diseases (Based on Symptoms)
   - Use database FIRST
   - List up to 3 diseases
   - One-line reasoning
   - NOT a diagnosis

2️⃣ 🌿 Ayurvedic Herbs & Formulations
3️⃣ 🥗 Diet & Lifestyle
4️⃣ 🧘 Yoga & Therapy
5️⃣ 🛡️ Prevention

Rules:
- Database is PRIMARY
- Google search only for dosage/safety
- Use emojis and clear headings
- Always add disclaimer
"""
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config
    )
    return response.text if hasattr(response, "text") else "No response"

# ------------------ PDF ------------------
def sanitize_for_pdf(text):
    return str(text).encode("ascii", "ignore").decode("ascii")

def generate_pdf_report(user_info, conversation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AyurGenix AI Consultation Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 11)
    for k, v in user_info.items():
        pdf.cell(0, 7, f"{k}: {sanitize_for_pdf(v)}", ln=True)

    pdf.ln(5)
    for c in conversation:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"{c['role'].upper()}:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_for_pdf(c["content"]))
        pdf.ln(2)

    return bytes(pdf.output())

# ------------------ SESSION ------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "profile_saved" not in st.session_state:
    st.session_state.profile_saved = False

df = load_dataset()
client = init_client()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("🌿 User Profile")
    with st.form("profile"):
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120, 25)
        dosha = st.selectbox("Dosha", ["Unknown", "Vata", "Pitta", "Kapha", "Tridosha"])
        stress = st.selectbox("Stress", ["Low", "Moderate", "High"])
        if st.form_submit_button("Save"):
            st.session_state.user_info = {
                "Name": name,
                "Age": age,
                "Dosha": dosha,
                "Stress": stress
            }
            st.session_state.profile_saved = True
            st.session_state.conversation = []
            st.rerun()

    if st.session_state.conversation:
        st.download_button(
            "📥 Download PDF",
            generate_pdf_report(st.session_state.user_info, st.session_state.conversation),
            "ayurgenix_report.pdf",
            "application/pdf"
        )

# ------------------ MAIN ------------------
st.title("🌿 AyurGenix AI")

for msg in st.session_state.conversation:
    with st.chat_message("assistant" if msg["role"] == "model" else "user"):
        st.write(msg["content"])

# ------------------ CHAT INPUT (ONLY ONE) ------------------
if prompt := st.chat_input("Describe your symptoms or ask about Ayurvedic remedies..."):
    if not st.session_state.profile_saved:
        st.warning("Please save your profile first.")
    elif not client:
        st.error("API key not set.")
    else:
        st.session_state.conversation.append({
            "role": "user",
            "content": prompt
        })

        matches = search_dataset(df, prompt)
        context = format_matches_for_context(matches)

        full_prompt = f"""
User Profile:
{st.session_state.user_info}

=== VERIFIED AYURVEDIC DATABASE ===
{context}
=== END DATABASE ===

User Symptoms:
{prompt}
"""

        with st.spinner("🌿 Consulting ancient wisdom..."):
            response = generate_response(
                client,
                full_prompt,
                st.session_state.conversation
            )

        st.session_state.conversation.append({
            "role": "model",
            "content": response
        })

        st.rerun()

# ------------------ WELCOME ------------------
if not st.session_state.profile_saved:
    st.info("""
👋 **Welcome to AyurGenix AI**

1️⃣ Fill your profile  
2️⃣ Save profile  
3️⃣ Start consultation  
""")
