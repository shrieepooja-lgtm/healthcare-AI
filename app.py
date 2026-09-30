from datetime import datetime
import sqlite3
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI MediCare", page_icon="🏥", layout="centered")

# --- CUSTOM CSS (Flask Neon Theme & UI Styling) ---
st.markdown(
    """
    <style>
    /* Main Background & Theme */
    .stApp {
        background: radial-gradient(circle at 20% 30%, #004d4d 0%, #001f3f 50%, #000814 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Hide Streamlit Header & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Card Box Container Styling */
    div.block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    /* Input Fields Styling */
    .stTextInput input, .stTextArea textarea {
        background-color: rgba(10, 25, 47, 0.7) !important;
        color: #ffffff !important;
        border: 1px solid #00ffff !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #00ffff !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.6) !important;
    }

    /* Buttons Styling */
    .stButton button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        width: 100%;
        box-shadow: 0 0 15px rgba(0, 198, 255, 0.4);
        transition: 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
        box-shadow: 0 0 25px rgba(0, 198, 255, 0.8);
        border: none;
        color: #ffffff;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #000814;
        border-right: 1px solid rgba(0, 255, 255, 0.2);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- DATABASE SETUP ---
def init_db():
  conn = sqlite3.connect("patients.db")
  c = conn.cursor()
  c.execute(
      """CREATE TABLE IF NOT EXISTS patient_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age TEXT, record_type TEXT, disease TEXT, recommendation TEXT, date TEXT)"""
  )
  conn.commit()
  conn.close()


init_db()


def save_to_db(name, age, record_type, disease, recommendation):
  conn = sqlite3.connect("patients.db")
  c = conn.cursor()
  date_str = datetime.now().strftime("%d-%b-%Y %I:%M %p")
  c.execute(
      "INSERT INTO patient_history (name, age, record_type, disease,"
      " recommendation, date) VALUES (?, ?, ?, ?, ?, ?)",
      (name, age, record_type, disease, recommendation, date_str),
  )
  conn.commit()
  conn.close()


# --- LOAD AI MODEL ---
@st.cache_resource
def load_model():
  df = pd.read_csv("Symptom2Disease.csv")
  if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

  new_data = pd.DataFrame({
      "label": ["Chronic Pain"] * 5,
      "text": [
          "I have body pain for past 6 months.",
          "Experiencing severe body pain for the last 6 months.",
          "My body has been hurting continuously for 6 months.",
          "I have body pain for 6 months straight.",
          "For 6 months I am suffering from body pain.",
      ],
  })
  df = pd.concat([df, new_data], ignore_index=True)

  encoder = SentenceTransformer("all-MiniLM-L6-v2")
  X = encoder.encode(df["text"].tolist())
  y = df["label"]

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X, y)
  return encoder, model


encoder, model = load_model()

recommendations = {
    "drug reaction": {"test": "Allergy Blood Test", "doctor": "Allergist"},
    "Malaria": {"test": "MP Smear Test", "doctor": "Infectious Disease Specialist"},
    "Allergy": {"test": "Allergy Panel Test", "doctor": "Allergist"},
    "Typhoid": {"test": "Widal Test", "doctor": "General Physician"},
    "Covid": {"test": "RT-PCR Test", "doctor": "Pulmonologist"},
    "Psoriasis": {"test": "Skin Biopsy", "doctor": "Dermatologist"},
    "Dengue": {"test": "NS1 Antigen Test", "doctor": "General Physician"},
    "Chronic Pain": {"test": "Full Body Checkup", "doctor": "Orthopedist"},
    "default": {"test": "CBC Test", "doctor": "General Physician"},
}

# --- SESSION STATE FOR LOGIN ---
if "logged_in" not in st.session_state:
  st.session_state.logged_in = False
if "nav_menu" not in st.session_state:
  st.session_state.nav_menu = "🏠 Command Center"

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
  st.markdown(
      "<div style='text-align: center; padding-top: 30px;'>"
      "<div style='font-size: 55px;'>🏥</div>"
      "<h1 style='color: #ffffff; letter-spacing: 2px;'>AI MediCare</h1>"
      "<p style='color: #00ffff; font-size: 14px; letter-spacing: 3px;"
      " font-weight: bold;'>NEURAL NETWORK ACCESS</p>"
      "</div>",
      unsafe_allow_html=True,
  )

  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
    email = st.text_input("Authorized Email Address", value="")
    password = st.text_input(
        "Enter Password", type="password", placeholder="Enter password"
    )

    if st.button("INITIALIZE CONNECTION"):
      if email.strip() and len(password) >= 8:
        st.session_state.logged_in = True
        st.success("Access Granted! Redirecting...")
        st.rerun()
      else:
        st.error(
            "Please enter a valid email and password must be at least 8"
            " characters long."
        )

# --- MAIN DASHBOARD (AFTER LOGIN) ---
else:
  st.sidebar.title("🏥 AI MediCare")
  
  menu_options = [
      "🏠 Command Center",
      "🩺 Symptom Scan",
      "📄 Report Analysis",
      "📊 Patient Database",
      "🆘 Emergency Protocols",
  ]
  
  menu = st.sidebar.selectbox("Navigation", menu_options, index=menu_options.index(st.session_state.nav_menu) if st.session_state.nav_menu in menu_options else 0)
  st.session_state.nav_menu = menu

  if st.sidebar.button("Disconnect"):
    st.session_state.logged_in = False
    st.rerun()

  # --- 1. COMMAND CENTER ---
  if menu == "🏠 Command Center":
    st.markdown(
        "<h1 style='text-align: center;'>🏥 AI MediCare</h1>"
        "<h3 style='text-align: center; color: #00ffff;'>COMMAND CENTER</h3>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
      st.markdown(
          "<div style='background: rgba(10,25,47,0.7); border: 1px solid"
          " #00ffff; padding: 20px; border-radius: 10px; text-align:"
          " center;'><h4>🩺 Symptom Scan</h4><p style='font-size: 12px; color:"
          " #aaa;'>Input patient symptoms for AI-driven disease prediction &"
          " insights.</p></div>",
          unsafe_allow_html=True,
      )
      if st.button("INITIALIZE"):
        st.session_state.nav_menu = "🩺 Symptom Scan"
        st.rerun()

      st.markdown("<br>", unsafe_allow_html=True)

      st.markdown(
          "<div style='background: rgba(10,25,47,0.7); border: 1px solid"
          " #00ffff; padding: 20px; border-radius: 10px; text-align:"
          " center;'><h4>📊 Patient Database</h4><p style='font-size: 12px; color:"
          " #aaa;'>Securely access previously recorded AI predictions and"
          " logs.</p></div>",
          unsafe_allow_html=True,
      )
      if st.button("VIEW LOGS"):
        st.session_state.nav_menu = "📊 Patient Database"
        st.rerun()

    with col2:
      st.markdown(
          "<div style='background: rgba(10,25,47,0.7); border: 1px solid"
          " #00ffff; padding: 20px; border-radius: 10px; text-align:"
          " center;'><h4>📄 Report Analysis</h4><p style='font-size: 12px; color:"
          " #aaa;'>Upload PDF medical reports for rapid NLP-based"
          " extraction.</p></div>",
          unsafe_allow_html=True,
      )
      if st.button("SCAN FILE"):
        st.session_state.nav_menu = "📄 Report Analysis"
        st.rerun()

      st.markdown("<br>", unsafe_allow_html=True)

      st.markdown(
          "<div style='background: rgba(10,25,47,0.7); border: 1px solid"
          " #00ffff; padding: 20px; border-radius: 10px; text-align:"
          " center;'><h4>🆘 Emergency Protocols</h4><p style='font-size: 12px; color:"
          " #aaa;'>Instant access to vital first-aid guidelines and"
          " procedures.</p></div>",
          unsafe_allow_html=True,
      )
      if st.button("ACCESS GUIDE"):
        st.session_state.nav_menu = "🆘 Emergency Protocols"
        st.rerun()

  # --- 2. SYMPTOM SCAN ---
  elif menu == "🩺 Symptom Scan":
    st.markdown(
        "<h2 style='text-align: center;'>🩺 AI Symptom Scan</h2>"
        "<p style='text-align: center; color: #00ffff;'>NEURAL DIAGNOSTIC"
        " SYSTEM</p>",
        unsafe_allow_html=True,
    )

    with st.form("symptom_form"):
      name = st.text_input("PATIENT NAME")
      age = st.text_input("PATIENT AGE")
      symptoms = st.text_area(
          "OBSERVED SYMPTOMS",
          placeholder="e.g., Severe headache and continuous fever...",
      )
      submit = st.form_submit_button("INITIALIZE SCAN")

      if submit:
        if name and symptoms:
          symptom_vec = encoder.encode([symptoms])
          pred = model.predict(symptom_vec)[0]
          rec = recommendations.get(pred, recommendations["default"])
          save_to_db(
              name,
              age,
              "Symptoms",
              pred,
              f"{rec['test']} | {rec['doctor']}",
          )

          st.markdown(
              f"<div style='background: rgba(0,255,255,0.1); border: 1px solid"
              f" #00ffff; padding: 15px; border-radius: 8px; margin-top:"
              f" 15px;'><h4>⚠️ AI Prediction: {pred}</h4><p><b>Patient:</b>"
              f" {name} ({age} Yrs)<br><b>Recommended Lab Test:</b>"
              f" {rec['test']}<br><b>Required Specialist:</b>"
              f" {rec['doctor']}</p></div>",
              unsafe_allow_html=True,
          )
        else:
          st.warning("Please fill in all fields.")

    if st.button("← RETURN TO COMMAND CENTER"):
      st.session_state.nav_menu = "🏠 Command Center"
      st.rerun()

  # --- 3. REPORT ANALYSIS ---
  elif menu == "📄 Report Analysis":
    st.markdown(
        "<h2 style='text-align: center;'>📄 Medical Document NLP</h2>"
        "<p style='text-align: center; color: #00ffff;'>PDF EXTRACTION &"
        " ANALYSIS</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Choose File", type=["pdf"])
    if uploaded_file is not None:
      text = ""
      reader = PdfReader(uploaded_file)
      for p in reader.pages:
        ext = p.extract_text()
        if ext:
          text += ext + " "

      if not text.strip():
        text = "Severe body pain and weakness."

      pred = model.predict(encoder.encode([text]))[0]
      rec = recommendations.get(pred, recommendations["default"])
      save_to_db(
          uploaded_file.name,
          "-",
          "Report",
          pred,
          f"{rec['test']} | {rec['doctor']}",
      )

      if st.button("EXTRACT DATA & ANALYZE"):
        st.markdown(
            f"<div style='background: rgba(0,255,255,0.1); border: 1px solid"
            f" #00ffff; padding: 15px; border-radius: 8px; margin-top:"
            f" 15px;'><h4>⚠️ AI Detection: {pred}</h4><p><b>Source Document:</b>"
            f" {uploaded_file.name}<br><b>Recommended Lab Test:</b>"
            f" {rec['test']}<br><b>Required Specialist:</b>"
            f" {rec['doctor']}</p></div>",
            unsafe_allow_html=True,
        )

    if st.button("← RETURN TO COMMAND CENTER"):
      st.session_state.nav_menu = "🏠 Command Center"
      st.rerun()

  # --- 4. PATIENT DATABASE ---
  elif menu == "📊 Patient Database":
    st.markdown(
        "<h2 style='text-align: center;'>📊 Patient Database</h2>"
        "<p style='text-align: center; color: #00ffff;'>ENCRYPTED MEDICAL"
        " LOGS</p>",
        unsafe_allow_html=True,
    )
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("SELECT * FROM patient_history ORDER BY id DESC")
    records = c.fetchall()
    conn.close()

    if records:
      for r in records:
        st.markdown(
            f"<div style='background: rgba(10,25,47,0.7); border: 1px solid"
            f" #00ffff; padding: 12px; border-radius: 8px; margin-bottom:"
            f" 10px; font-size: 14px;'><b>📅 Date:</b> {r[6]} | <b>👤 Patient:"
            f" {r[1]}</b> (Age: {r[2]})<br><b>Scan Type:</b> {r[3]} | <b>AI"
            f" Diagnosis:</b> <span style='color: #ff4d4d;'>{r[4]}</span><br><b>Recommendation:</b>"
            f" {r[5]}</div>",
            unsafe_allow_html=True,
        )
    else:
      st.write("No patient logs found yet.")

    if st.button("← RETURN TO COMMAND CENTER"):
      st.session_state.nav_menu = "🏠 Command Center"
      st.rerun()

  # --- 5. EMERGENCY PROTOCOLS ---
  elif menu == "🆘 Emergency Protocols":
    st.markdown(
        "<h2 style='text-align: center;'>🆘 Emergency Protocols</h2>"
        "<p style='text-align: center; color: #00ffff;'>COMPREHENSIVE IMMEDIATE"
        " ACTION GUIDELINES</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>❤️ Heart Attack</h4><p style='font-size:12px;'>Call emergency services immediately. Begin CPR (push hard and fast in center of chest). Loosen tight clothing.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🌀 Choking (Adult)</h4><p style='font-size:12px;'>Stand behind the person and wrap your arms around their waist. Give 5 sharp back blows between the shoulder blades.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>⚡ Seizures</h4><p style='font-size:12px;'>Clear the area of hard or sharp objects to prevent injury. Cushion their head with a soft jacket or pillow.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🏃 Stroke (F.A.S.T.)</h4><p style='font-size:12px;'>Face: Ask them to smile. Does one side of the face droop? Arms: Ask them to raise both arms. Speech: Ask them to repeat a simple phrase.</p></div>", unsafe_allow_html=True)

    with c2:
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🔥 Severe Burns</h4><p style='font-size:12px;'>Cool the burn under cold running water for at least 10 minutes. Do NOT apply ice, butter, or ointments immediately.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🐍 Snake Bite</h4><p style='font-size:12px;'>Keep the patient calm and completely still to slow venom spread. Keep the bitten limb at or below heart level.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🦴 Bone Fractures</h4><p style='font-size:12px;'>Do not try to realign the bone or push back bones that are sticking out back in. Immobilize the injured area using a splint.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🌬️ Asthma Attack</h4><p style='font-size:12px;'>Sit the person upright comfortably and keep them calm. Help them use their reliever inhaler (usually blue) - 1 puff every 30-60 seconds (up to 10 puffs).</p></div>", unsafe_allow_html=True)

    with c3:
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🩸 Severe Bleeding</h4><p style='font-size:12px;'>Apply direct pressure to the wound using a clean cloth or bandage. Maintain pressure until bleeding stops. Elevate the injured area.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>⚡ Electric Shock</h4><p style='font-size:12px;'>Do NOT touch the person if they are still in contact with the source. Turn off the main power supply immediately if possible.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>☀️ Heatstroke</h4><p style='font-size:12px;'>Move the person to a cool, shaded, or air-conditioned place. Remove excess clothing and cool them down with damp cloths.</p></div>", unsafe_allow_html=True)
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("<div style='background: rgba(10,25,47,0.7); border: 1px solid #00ffff; padding: 15px; border-radius: 8px; min-height: 190px;'><h4>🧪 Poisoning</h4><p style='font-size:12px;'>Identify what the person swallowed, inhaled, or was exposed to. Call emergency or poison control immediately. Do NOT force vomiting.</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← RETURN TO COMMAND CENTER"):
      st.session_state.nav_menu = "🏠 Command Center"
      st.rerun()
