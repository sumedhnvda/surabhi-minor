import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from datetime import datetime
from fpdf import FPDF
import io

# Load environment variables
load_dotenv()

# Model configuration
MODEL = "gemini-2.0-flash"

# Page configuration
st.set_page_config(
    page_title="AyurGenix AI - Ayurvedic Medicine Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
/* Desktop: wider sidebar */
@media (min-width: 768px) {
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 400px;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 350px;
    }
}
/* Mobile: default sidebar behavior */
@media (max-width: 767px) {
    [data-testid="stSidebar"] {
        min-width: 0 !important;
        max-width: 100vw !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 100% !important;
    }
    /* Better mobile spacing */
    .stMainBlockContainer {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Improve chat input on mobile */
    [data-testid="stChatInput"] {
        max-width: 100% !important;
    }
    /* Better form spacing on mobile */
    [data-testid="stForm"] {
        padding: 0.5rem !important;
    }
    /* Adjust title size on mobile */
    h1 {
        font-size: 1.75rem !important;
    }
}
/* Ensure chat messages don't overflow */
[data-testid="stChatMessage"] {
    max-width: 100%;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

# Load the Ayurvedic dataset
@st.cache_data
def load_dataset():
    try:
        df = pd.read_excel("AyurGenixAI_Dataset (1).xlsx")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Initialize Gemini client
@st.cache_resource
def init_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        return None

# Search dataset for matching conditions
def search_dataset(df, query):
    if df is None:
        return []
    
    query_lower = query.lower()
    matches = []
    for idx, row in df.iterrows():
        disease = str(row.get('Disease', '')).lower()
        symptoms = str(row.get('Symptoms', '')).lower()
        
        if query_lower in disease or query_lower in symptoms:
            matches.append(row)
        else:
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and (word in disease or word in symptoms):
                    matches.append(row)
                    break
    return matches[:5]

# Format dataset matches for context - uses Excel columns
def format_matches_for_context(matches):
    if not matches:
        return "No direct matches found in the Ayurvedic database."
    
    context = "**Relevant data from Ayurvedic database:**\n\n"
    for i, match in enumerate(matches, 1):
        context += f"""
{i}. {match.get('Disease', 'N/A')}
   Symptoms: {match.get('Symptoms', 'N/A')}
   Duration of Treatment: {match.get('Duration of Treatment', 'N/A')}
   Medical History: {match.get('Medical History', 'N/A')}
   Current Medications: {match.get('Current Medications', 'N/A')}
   Risk Factors: {match.get('Risk Factors', 'N/A')}
   Stress Levels: {match.get('Stress Levels', 'N/A')}
   Herbal/Alternative Remedies: {match.get('Herbal/Alternative Remedies', 'N/A')}
   Ayurvedic Herbs: {match.get('Ayurvedic Herbs', 'N/A')}
   Formulation: {match.get('Formulation', 'N/A')}
   Doshas: {match.get('Doshas', 'N/A')}
   Constitution/Prakriti: {match.get('Constitution/Prakriti', 'N/A')}
   Diet and Lifestyle Recommendations: {match.get('Diet and Lifestyle Recommendations', 'N/A')}
   Yoga & Physical Therapy: {match.get('Yoga & Physical Therapy', 'N/A')}
   Prevention: {match.get('Prevention', 'N/A')}
   Patient Recommendations: {match.get('Patient Recommendations', 'N/A')}
"""
    return context

# Generate response
def generate_response(client, prompt, conversation_history):
    contents = []
    
    for msg in conversation_history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    )
    
    tools = [types.Tool(google_search=types.GoogleSearch())]
    
    # Updated System Instruction to include the "Possible Diseases" section requirement
    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction="""You are AyurGenix AI, a compassionate Ayurvedic medicine assistant. 
        
        CRITICAL: The DATABASE INFORMATION provided in the prompt is from a VERIFIED, 100% ACCURATE Ayurvedic medical database.

        ALWAYS use the database information as your PRIMARY foundation.

        HOW TO STRUCTURE YOUR RESPONSE:

        1. **Possible Diseases Analysis**: 
           - Start by analyzing the user's described symptoms.
           - Compare these symptoms with the "Relevant data from Ayurvedic database".
           - Create a clear section titled "### 🔍 Possible Ayurvedic Conditions".
           - List the diseases from the database that match the symptoms. 
           - If the symptoms are vague, suggest the most likely conditions based on Ayurvedic principles (Vata/Pitta/Kapha imbalance).

        2. **Treatment & Recommendations**:
           - Use the DATABASE to quote specific herbs, formulations, diet, and yoga.
           - Use GOOGLE SEARCH to add dosages, preparation methods, and modern research.

        Guidelines:
        - Be empathetic and supportive.
        - Start with the "Possible Ayurvedic Conditions" section to show you understand the issue.
        - Follow with the treatment plan using the database.
        - Consider user's dosha type in recommendations.
        - Always remind this is informational, not medical advice.
        - Focus ONLY on health and Ayurveda - redirect other topics politely.

        When responding:
        - Start with "Based on our verified Ayurvedic database..." for database info.
        - Add "Additionally, from current research..." for Google Search supplementary info.
        - Format with clear headings and bullet points. Use emoji (🌿 herbs, 🧘 yoga, 🥗 diet)."""
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            text_parts = []
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
            return ''.join(text_parts)
        return "Unable to generate response."
    except Exception as e:
        return f"Error: {str(e)}"

# Helper function to sanitize text for PDF (ASCII-safe)
def sanitize_for_pdf(text):
    """Remove non-ASCII characters and clean text for PDF generation."""
    if not text:
        return ""
    # Convert to string if not already
    text = str(text)
    # Remove markdown formatting
    text = text.replace('**', '').replace('*', '')
    text = text.replace('###', '').replace('##', '').replace('#', '')
    # Encode to ASCII, replacing non-ASCII chars with closest equivalent or removing them
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Clean up any extra whitespace
    text = ' '.join(text.split())
    return text

# Generate PDF report
def generate_pdf_report(user_info, conversation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "AyurGenix AI - Consultation Report", ln=True, align="C")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)
    
    # User Profile Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "User Profile", ln=True)
    
    profile_items = [
        ("Name", user_info.get('name', 'Not provided')),
        ("Age", str(user_info.get('age', 'Not provided'))),
        ("Gender", user_info.get('gender', 'Not provided')),
        ("Dosha Type", user_info.get('dosha', 'Unknown')),
        ("Stress Level", user_info.get('stress', 'Not provided')),
        ("Existing Conditions", user_info.get('conditions', 'None mentioned')),
        ("Current Medications", user_info.get('medications', 'None mentioned')),
    ]
    
    for label, value in profile_items:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"{label}:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, sanitize_for_pdf(value))
        pdf.ln(2)
    
    pdf.ln(5)
    
    # Consultation Summary Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Consultation Summary", ln=True)
    pdf.ln(5)
    
    for msg in conversation:
        if msg["role"] == "user":
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(0, 0, 128) # Blue for user
            pdf.cell(0, 8, "You:", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(0, 100, 0) # Green for AI
            pdf.cell(0, 8, "AyurGenix AI:", ln=True)
            pdf.set_text_color(0, 0, 0) # Reset to black
            pdf.set_font("Helvetica", "", 10)
            
        # Sanitize content for PDF
        content = sanitize_for_pdf(msg['content'])
        pdf.multi_cell(0, 6, content)
        pdf.ln(5)
        
    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "Disclaimer: This report is for informational purposes only. Please consult a qualified healthcare provider for medical concerns.")
    
    # Return PDF as bytes
    return bytes(pdf.output())

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "profile_saved" not in st.session_state:
    st.session_state.profile_saved = False
if "greeting_sent" not in st.session_state:
    st.session_state.greeting_sent = False

# Load dataset and client
df = load_dataset()
client = init_client()

if not client:
    st.error("⚠️ Please set GOOGLE_API_KEY in your .env file")

# Sidebar - User Profile
with st.sidebar:
    st.header("🌿 Your Profile")
    
    with st.form("user_info_form"):
        name = st.text_input("👤 Your Name", value=st.session_state.user_info.get('name', ''))
        age = st.number_input("🎂 Age", min_value=1, max_value=120, value=st.session_state.user_info.get('age', 25))
        
        gender_options = ["Select", "Male", "Female", "Other"]
        stored_gender = st.session_state.user_info.get('gender', 'Select')
        gender_index = gender_options.index(stored_gender) if stored_gender in gender_options else 0
        gender = st.selectbox("⚧ Gender", gender_options, index=gender_index)
        
        st.subheader("🔮 Ayurvedic Profile")
        dosha_options = ["Unknown", "Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Tridosha"]
        stored_dosha = st.session_state.user_info.get('dosha', 'Unknown')
        dosha_index = dosha_options.index(stored_dosha) if stored_dosha in dosha_options else 0
        dosha = st.selectbox("🧬 Your Dosha Type (if known)", dosha_options, index=dosha_index)
        
        stress = st.select_slider("😰 Stress Level", options=["Low", "Moderate", "High", "Very High"], value=st.session_state.user_info.get('stress', 'Moderate'))
        
        st.subheader("🏥 Health Info")
        conditions = st.text_area("📋 Health Conditions", value=st.session_state.user_info.get('conditions', ''), placeholder="E.g., Diabetes, Hypertension")
        medications = st.text_area("💊 Current Medications", value=st.session_state.user_info.get('medications', ''), placeholder="List medications")
        
        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        
        if submitted:
            st.session_state.user_info = {
                'name': name if name else 'Guest',
                'age': age,
                'gender': gender if gender != "Select" else "Not provided",
                'dosha': dosha,
                'stress': stress,
                'conditions': conditions if conditions else "None mentioned",
                'medications': medications if medications else "None mentioned"
            }
            st.session_state.profile_saved = True
            st.session_state.greeting_sent = False
            st.session_state.conversation = []
            st.success("✅ Profile saved!")
            st.rerun()

    st.divider()
    
    if st.session_state.conversation:
        pdf_bytes = generate_pdf_report(st.session_state.user_info, st.session_state.conversation)
        st.download_button(
            label="📥 Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"ayurgenix_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    if st.button("🔄 New Consultation", use_container_width=True):
        st.session_state.conversation = []
        st.session_state.greeting_sent = False
        st.rerun()

    # Show dataset status
    st.divider()
    if df is not None:
        st.success(f"📊 Dataset loaded: {len(df)} conditions")
    else:
        st.error("❌ Dataset not loaded")

# Main content
st.title("🌿 AyurGenix AI")
st.caption("Your Personal Ayurvedic Medicine Assistant")

st.divider()

# Auto-generate greeting after profile save
if st.session_state.profile_saved and not st.session_state.greeting_sent and client:
    user_info = st.session_state.user_info
    
    greeting_prompt = f"""You are AyurGenix AI. The user just saved their profile. 
    
    IMPORTANT: Your ONLY job right now is to greet the user and ASK QUESTIONS to understand their health problem.
    DO NOT give any recommendations, suggestions, herbs, yoga, or diet advice yet.
    DO NOT provide any Ayurvedic guidance until you understand what problem they are facing.
    
    User Profile:
    Name: {user_info.get('name')}
    Age: {user_info.get('age')}
    Gender: {user_info.get('gender')}
    Dosha: {user_info.get('dosha')}
    Stress Level: {user_info.get('stress')}
    Existing Conditions: {user_info.get('conditions')}
    Current Medications: {user_info.get('medications')}
    
    Your response should:
    1. Greet them warmly by name (1-2 sentences max)
    2. Briefly acknowledge their dosha type if known
    3. Ask 2-3 specific questions to understand: 
       - What health issues or symptoms are they experiencing?
       - How long have they been experiencing these issues?
       - What brings them to seek Ayurvedic help today?
       
    Keep your response SHORT and focused on asking questions. Do NOT provide any advice yet."""
    
    with st.spinner("🌿 Preparing your personalized consultation..."):
        full_response = generate_response(client, greeting_prompt, [])
        st.session_state.conversation.append({"role": "model", "content": full_response})
        st.session_state.greeting_sent = True
        st.rerun()

# Display conversation using Streamlit's native chat
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🌿"):
            st.write(msg["content"])

# Welcome message
if not st.session_state.profile_saved:
    st.info("""
    👋 Welcome to AyurGenix AI!
    
    To begin your personalized Ayurvedic consultation:
    1. 📝 Fill in your profile in the sidebar
    2. 💾 Click "Save Profile" to start
    3. 💬 I'll guide you through your health needs
    """)

# Chat input
if prompt := st.chat_input("Describe your symptoms or ask about Ayurvedic remedies..."):
    if not st.session_state.profile_saved:
        st.warning("Please save your profile first.")
    elif not client:
        st.error("API key not set.")
    else:
        # Show user message immediately
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get dataset context
        matches = search_dataset(df, prompt)
        dataset_context = format_matches_for_context(matches)
        
        # Build full prompt with context and request for "Possible Diseases" section
        user_info = st.session_state.user_info
        full_prompt = f"""User Profile: 
        Name: {user_info.get('name')}
        Age: {user_info.get('age')}
        Dosha: {user_info.get('dosha')}
        Stress: {user_info.get('stress')}
        Conditions: {user_info.get('conditions')}
        Medications: {user_info.get('medications')}
        
        === VERIFIED AYURVEDIC DATABASE (100% ACCURATE - USE THIS FIRST) ===
        {dataset_context}
        === END DATABASE ===
        
        User Question: {prompt}
        
        CRITICAL INSTRUCTIONS:
        
        1. **Possible Diseases Section**:
           - Based on the user's symptoms and the database context above, analyze and list "Possible Diseases".
           - Look for matches between the user's symptoms and the database 'Disease' or 'Symptoms' columns.
           - If no direct database match is clear, use your Ayurvedic knowledge to suggest likely imbalances/conditions.
           - Present this as the FIRST section of your response.
        
        2. **Recommendations**:
           - YOU MUST use the database information as your PRIMARY source for recommendations.
           - Quote the specific herbs, formulations, diet, yoga recommendations from the database.
           - Only use Google Search to supplement or verify safety/interactions.
           - Start your response with "Based on our verified Ayurvedic database..." if database matches found.
        """
        
        # Generate response
        with st.spinner("🌿 Consulting ancient wisdom..."):
            full_response = generate_response(client, full_prompt, st.session_state.conversation)
            st.session_state.conversation.append({"role": "model", "content": full_response})
            st.rerun()

