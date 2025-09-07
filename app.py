from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    with open("user_data.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, mood])

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]

    # Predict answer
    user_vect = vectorizer.transform([user_message])
    reply = model.predict(user_vect)[0]

    # Predict mood and log
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

