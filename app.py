from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

USER_CSV = "user_data.csv"

app = Flask(__name__)

# Load Q&A data
df = pd.read_csv("data.csv")

# Train ML model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])
model = LogisticRegression()
model.fit(X, df["answer"])

# Mood prediction
def predict_mood(message):
    message = message.lower()
    if any(word in message for word in ["sad","depressed","lonely","hopeless"]):
        return "sad"
    elif any(word in message for word in ["stressed","anxious","overwhelmed","worried"]):
        return "stressed"
    elif any(word in message for word in ["happy","good","motivated","excited"]):
        return "happy"
    elif any(word in message for word in ["tired","lazy","unmotivated"]):
        return "tired"
    else:
        return "neutral"

# Log messages
def log_message(message, mood):
    write_header = not os.path.exists(USER_CSV)
    with open(USER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "message", "mood"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, mood])

# API to accept mood + journal entry (from client)
@app.route("/api/log-mood", methods=["POST"])
def api_log_mood():
    # expects JSON: { "mood": "<emoji or label>", "journal": "<text>" }
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    journal = data.get("journal", "").strip()

    if not mood:
        return jsonify({"ok": False, "error": "mood required"}), 400

    # reuse log_message to append to CSV
    # ensure header is present when logging (log_message handles this if you used the fixed version)
    log_message(journal if journal else "-", mood)
    return jsonify({"ok": True})

# API to return aggregated mood data (for chart)
@app.route("/api/mood-data")
def api_mood_data():
    USER_CSV = "user_data.csv"
    if not os.path.exists(USER_CSV):
        return jsonify({"labels": [], "counts": []})

    try:
        df = pd.read_csv(USER_CSV)
    except Exception as e:
        print("Error reading user_data.csv:", e)
        return jsonify({"labels": [], "counts": []})

    if "mood" not in df.columns:
        return jsonify({"labels": [], "counts": []})

    # produce ordered data: unique moods and counts
    counts = df["mood"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    return jsonify({"labels": labels, "counts": values})

# route to serve the mood tracker page (render UI)
@app.route("/mood-tracker-page")
def mood_tracker_page():
    return render_template("mood_tracker.html")


# Routes
@app.route("/")
def landing():
    return render_template("login.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]
    user_vect = vectorizer.transform([user_message])
    reply = model.predict(user_vect)[0]
    mood = predict_mood(user_message)
    log_message(user_message, mood)
    return jsonify({"reply": reply})

import matplotlib.pyplot as plt
from fpdf import FPDF
import os

@app.route("/report")
def report():
    # Read logged data
    if not os.path.exists("user_data.csv"):
        return "No data yet!"

    df = pd.read_csv("user_data.csv")
    
    # Count moods
    mood_counts = df["mood"].value_counts()

    # Generate chart
    plt.figure(figsize=(6,4))
    mood_counts.plot(kind='bar', color=['#007BFF','#FF5733','#28B463','#F1C40F','#A569BD'])
    plt.title("Your Mood Summary")
    plt.xlabel("Mood")
    plt.ylabel("Number of Messages")
    plt.tight_layout()
    chart_path = "static/mood_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # Create PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MindMate Mood Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Total messages: {len(df)}", ln=True)
    pdf.ln(5)
    pdf.image(chart_path, x=25, w=160)
    pdf_file = "static/mood_report.pdf"
    pdf.output(pdf_file)

    return f'<h2>Report Generated!</h2><p><a href="/{pdf_file}" target="_blank">Download PDF</a></p><img src="/{chart_path}" width="500">'


@app.route("/dashboard")
def dashboard():
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    if not os.path.exists("user_data.csv"):
        return "No data yet!"

    df = pd.read_csv("user_data.csv")

    # Count moods
    mood_counts = df["mood"].value_counts()

    # Generate chart
    plt.figure(figsize=(6,4))
    mood_counts.plot(kind='bar', color=['#007BFF','#FF5733','#28B463','#F1C40F','#A569BD'])
    plt.title("Your Mood Summary")
    plt.xlabel("Mood")
    plt.ylabel("Number of Messages")
    plt.tight_layout()
    
    if not os.path.exists("static"):
        os.makedirs("static")
    chart_path = "static/mood_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("dashboard.html", chart_path=chart_path, total_messages=len(df))

if __name__ == "__main__":
    app.run(debug=True)

