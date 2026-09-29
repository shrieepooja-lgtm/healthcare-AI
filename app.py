from flask import Flask, render_template, request, session, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import sqlite3
from datetime import datetime
from flask_mail import Mail, Message
import random

app = Flask(__name__)
app.secret_key = 'super_secret_ai_medicare_key' 

# ================= MAIL SETUP =================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'shrieepooja@gmail.com'   
app.config['MAIL_PASSWORD'] = 'qohsmvgpplzcipij' 
mail = Mail(app)
# ==============================================

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patient_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age TEXT, record_type TEXT, disease TEXT, recommendation TEXT, date TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(name, age, record_type, disease, recommendation):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    c.execute("INSERT INTO patient_history (name, age, record_type, disease, recommendation, date) VALUES (?, ?, ?, ?, ?, ?)",
              (name, age, record_type, disease, recommendation, date_str))
    conn.commit()
    conn.close()

print("Loading AI Model (BERT)...")
df = pd.read_csv('Symptom2Disease.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

new_data = pd.DataFrame({'label': ['Chronic Pain']*5, 'text': ["I have body pain for past 6 months.", "Experiencing severe body pain for the last 6 months.", "My body has been hurting continuously for 6 months.", "I have body pain for 6 months straight.", "For 6 months I am suffering from body pain."]})
df = pd.concat([df, new_data], ignore_index=True)

encoder = SentenceTransformer('all-MiniLM-L6-v2')
X = encoder.encode(df['text'].tolist())
y = df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model Ready! 🚀")

recommendations = {
    "drug reaction": {"test": "Allergy Blood Test", "doctor": "Allergist"},
    "Malaria": {"test": "MP Smear Test", "doctor": "Infectious Disease Specialist"},
    "Allergy": {"test": "Allergy Panel Test", "doctor": "Allergist"},
    "Typhoid": {"test": "Widal Test", "doctor": "General Physician"},
    "Covid": {"test": "RT-PCR Test", "doctor": "Pulmonologist"},
    "Psoriasis": {"test": "Skin Biopsy", "doctor": "Dermatologist"},
    "Dengue": {"test": "NS1 Antigen Test", "doctor": "General Physician"},
    "Chronic Pain": {"test": "Full Body Checkup", "doctor": "Orthopedist"},
    "default": {"test": "CBC Test", "doctor": "General Physician"}
}

# --- REAL OTP ROUTES ---
@app.route('/send_otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    email = data.get('email')
    
    otp = str(random.randint(1000, 9999))
    session['otp'] = otp 
    
    try:
        msg = Message('AI MediCare - Login OTP', sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Hello,\n\nYour secret OTP for AI MediCare login is: {otp}\n\nDo not share this with anyone."
        mail.send(msg)
        return jsonify({'success': True, 'message': 'OTP Sent Successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': 'Failed to send email. Check App Password.'})

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    user_otp = data.get('otp')
    if 'otp' in session and session['otp'] == user_otp:
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    result = None
    if request.method == 'POST':
        name, age, symp = request.form['name'], request.form['age'], request.form['symptoms']
        symptom_vec = encoder.encode([symp])
        pred = model.predict(symptom_vec)[0]
        rec = recommendations.get(pred, recommendations["default"])
        save_to_db(name, age, "Symptoms", pred, f"{rec['test']} | {rec['doctor']}")
        result = {"name": name, "age": age, "disease": pred, "test": rec["test"], "doctor": rec["doctor"]}
    return render_template('index.html', result=result)

@app.route('/report', methods=['GET', 'POST'])
def report():
    result = None
    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            text = ""
            if f.filename.endswith('.pdf'):
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    ext = p.extract_text()
                    if ext: text += ext + " "
            if not text.strip(): text = "Severe body pain and weakness."
            pred = model.predict(encoder.encode([text]))[0]
            rec = recommendations.get(pred, recommendations["default"])
            save_to_db(f.filename, "-", "Report", pred, f"{rec['test']} | {rec['doctor']}")
            result = {"filename": f.filename, "disease": pred, "test": rec["test"], "doctor": rec["doctor"]}
    return render_template('report.html', result=result)

@app.route('/history')
def history():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute("SELECT * FROM patient_history ORDER BY id DESC")
    records = c.fetchall()
    conn.close()
    return render_template('history.html', records=records)

@app.route('/firstaid')
def firstaid():
    return render_template('firstaid.html')

if __name__ == '__main__':
    app.run(debug=True)
    import os
os.environ['HF_HOME'] = 'D:/hf_cache'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)