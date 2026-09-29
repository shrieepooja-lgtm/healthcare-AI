from flask import Flask, render_template, request
import os

app = Flask(__name__)

# --- Unga app-oda routes and prediction logic ellam inga irukatum ---
@app.route('/')
def home():
    return render_template('index.html')  # Unga index/home template iruntha inga varum

@app.route('/predict', methods=['POST'])
def predict():
    # Unga model prediction code ellam inga irukalam
    return "Prediction Results"


# --- Render-kku thevayana Port binding code (Never remove this!) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
