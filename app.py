import os
import pandas as pd
import random
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, current_app
import logging
import threading
import time
import functools

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask is working!"

if __name__ == "__main__":
    app.run(debug=True)

# If using transformers, import and check availability
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Try importing your real classes; if not available, provide safe stubs so app can run for testing.
try:
    # Replace 'your_module' with the actual module path if needed
    # from your_module import MindMateBot, EmpatheticResponseGenerator
    raise ImportError  # remove this line if you have real imports
except ImportError:
    class EmpatheticResponseGenerator:
        def __init__(self):
            pass

        def generate_empathetic_response(self, text, emotion_analysis=None):
            return "I hear you. Tell me more." if text else "I'm here."

        def analyze_emotion_depth(self, text):
            # simple placeholder analysis
            return {
                'dominant_emotion': {'label': 'NEUTRAL', 'score': 0.5},
                'intensity': 'medium'
            }

        def add_therapeutic_elements(self, response, emotion_analysis):
            return response + " (If you like, we can try a grounding exercise.)"

    class MindMateBot:
        def __init__(self, data_file="data.csv"):
            self.user_data_file = data_file
            self.conversation_memory = {}
            # simple placeholder for response_feedback used in routes
            self.response_feedback = {}

        def log_enhanced_message(self, message, emotion_analysis, tag):
            # simple logging stub
            logging.getLogger("mindmate").info(f"Logged: {tag} - {message[:100]}")

        def get_fallback_response(self, user_message):
            return "Sorry, something went wrong. I'm here with you."

        def generate_mood_report(self):
            # stub: return (chart_path, pdf_path) or (None, None)
            # In your real bot you would implement actual chart/pdf generation and return file paths
            return (None, None)

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logger = logging.getLogger("mindmate")
logging.basicConfig(level=logging.INFO)

# Make bot global but start as None (routes will check)
bot = None

# EnhancedResponse & Bot classes from your prompt (adapted slightly)
class EnhancedEmpatheticResponseGenerator(EmpatheticResponseGenerator):
    def __init__(self):
        super().__init__()
        self.conversation_patterns = {
            'first_interaction': [
                "Hello there! I'm so glad you decided to reach out today. {response}",
                "Welcome! Thank you for trusting me with your thoughts. {response}"
            ],
            'return_user': [
                "It's good to hear from you again. {response}",
                "I remember our last conversation. {response}"
            ],
            'crisis_detected': [
                "I'm deeply concerned about what you're sharing. {response} Please consider reaching out to a crisis hotline: 988 (US) or emergency services.",
            ]
        }

        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'want to die',
            'not worth living', 'better off dead', 'harm myself'
        ]

    def detect_crisis(self, text):
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)

    def generate_contextual_response(self, text, emotion_analysis, user_history=None):
        # Check crisis first
        if self.detect_crisis(text):
            base_response = self.generate_empathetic_response(text, emotion_analysis)
            crisis_response = random.choice(self.conversation_patterns['crisis_detected'])
            return crisis_response.format(response=base_response)

        pattern_type = 'first_interaction' if not user_history else 'return_user'
        base_response = self.generate_empathetic_response(text, emotion_analysis)

        if random.random() < 0.3:
            contextual_wrapper = random.choice(self.conversation_patterns[pattern_type])
            return contextual_wrapper.format(response=base_response)

        return base_response

class ImprovedMindMateBot(MindMateBot):
    def __init__(self, data_file="data.csv"):
        super().__init__(data_file)
        self.response_generator = EnhancedEmpatheticResponseGenerator()
        self.response_feedback = {}

    def get_enhanced_response(self, user_message, user_id="default"):
        try:
            user_history = self.conversation_memory.get(user_id, [])
            emotion_analysis = self.response_generator.analyze_emotion_depth(user_message)
            response = self.response_generator.generate_contextual_response(
                user_message, emotion_analysis, user_history
            )
            response = self.response_generator.add_therapeutic_elements(response, emotion_analysis)

            if user_id not in self.conversation_memory:
                self.conversation_memory[user_id] = []

            self.conversation_memory[user_id].append({
                'message': user_message,
                'emotion': emotion_analysis,
                'response': response,
                'timestamp': datetime.now()
            })

            self.log_enhanced_message(user_message, emotion_analysis, "enhanced_empathetic")
            return response, emotion_analysis
        except Exception as e:
            logger.error(f"Enhanced response error: {e}")
            return self.get_fallback_response(user_message), self.get_default_emotion_analysis()

    def get_default_emotion_analysis(self):
        return {
            'dominant_emotion': {'label': 'NEUTRAL', 'score': 0.5},
            'intensity': 'medium'
        }

class OptimizedMindMateBot(ImprovedMindMateBot):
    def __init__(self, data_file="data.csv"):
        super().__init__(data_file)
        self.max_conversation_history = 10
        self.max_users_in_memory = 100
        self.response_cache = {}
        self.cache_size_limit = 1000

    def cleanup_memory(self):
        current_time = datetime.now()
        for user_id in list(self.conversation_memory.keys()):
            history = self.conversation_memory[user_id]
            recent_history = [
                conv for conv in history
                if (current_time - conv.get('timestamp', current_time)).total_seconds() < 86400
            ]
            if recent_history:
                self.conversation_memory[user_id] = recent_history[-self.max_conversation_history:]
            else:
                del self.conversation_memory[user_id]

        if len(self.conversation_memory) > self.max_users_in_memory:
            sorted_users = sorted(
                self.conversation_memory.items(),
                key=lambda x: max((conv.get('timestamp') or datetime.min) for conv in x[1]),
                reverse=True
            )
            self.conversation_memory = dict(sorted_users[:self.max_users_in_memory])

# Enhanced response templates (as provided)
ENHANCED_RESPONSE_TEMPLATES = {
    'sadness': [
        "I can really hear the sadness in what you're sharing. {context} It takes courage to open up about these feelings. 💙",
        "That sounds incredibly difficult to go through. {context} Your feelings make complete sense given what you're experiencing.",
        "I'm sitting here with you in this sadness. {context} Sometimes we need to feel these emotions fully before we can move through them."
    ],
    'anxiety': [
        "Anxiety can be so overwhelming - like your mind is racing in a hundred directions. {context} What you're feeling is really common and understandable.",
        "That anxious energy sounds exhausting. {context} Let's take this one moment at a time together.",
        "I can sense how much your mind is churning right now. {context} Anxiety has a way of making everything feel urgent, but you're safe in this moment."
    ],
    'conversation_starters': [
        "What's been weighing on your heart lately?",
        "I'm here and ready to listen. What would be helpful to talk about?",
        "How are you really feeling today - beyond just 'fine'?",
        "What's one thing that's been on your mind recently?"
    ]
}

# safe_route decorator with functools.wraps
def safe_route(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Route {f.__name__} error: {e}", exc_info=True)
            response = {
                "error": "Something went wrong, but I'm still here for you."
            }
            if app.debug:
                response["technical_error"] = str(e)
            return jsonify(response), 500
    return wrapper

# Routes
@app.route("/dashboard")
@safe_route
def dashboard():
    global bot
    if bot is None:
        return render_template("dashboard.html", error="Server initializing, no bot instance available.", no_data=True, total_messages=0)

    try:
        if not os.path.exists(bot.user_data_file):
            return render_template("dashboard.html", no_data=True, total_messages=0)

        df = pd.read_csv(bot.user_data_file)
        if df.empty:
            return render_template("dashboard.html", no_data=True, total_messages=0)

        # Defensive access to expected columns
        emotion_col = 'emotion' if 'emotion' in df.columns else None
        intensity_col = 'intensity' if 'intensity' in df.columns else None
        sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else None

        stats = {
            'total_messages': len(df),
            'mood_distribution': df[emotion_col].value_counts().to_dict() if emotion_col else {},
            'intensity_breakdown': df[intensity_col].value_counts().to_dict() if intensity_col else {},
            'recent_sentiment': float(df.tail(10)[sentiment_col].mean()) if sentiment_col and len(df) > 0 else 0.5,
            'most_common_emotion': (df[emotion_col].mode().iloc[0] if (emotion_col and not df[emotion_col].mode().empty) else 'neutral')
        }

        return render_template("dashboard.html", stats=stats, no_data=False)

    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        if app.debug:
            return render_template("dashboard.html", error=str(e))
        return render_template("dashboard.html", error="Unable to load dashboard at this time.")

@app.route("/report")
@safe_route
def generate_report():
    global bot
    if bot is None:
        return jsonify({"error": "Bot not initialized"}), 503

    try:
        # Some implementations of generate_mood_report may return (chart_path, pdf_path)
        chart_path, pdf_path = (None, None)
        if hasattr(bot, "generate_mood_report"):
            try:
                chart_path, pdf_path = bot.generate_mood_report() or (None, None)
            except TypeError:
                # if generate_mood_report returns a single value or different shape
                result = bot.generate_mood_report()
                if isinstance(result, tuple) and len(result) >= 2:
                    chart_path, pdf_path = result[0], result[1]
                else:
                    chart_path = result

        return jsonify({
            "chart_available": bool(chart_path),
            "pdf_available": bool(pdf_path),
            "chart_url": (f"/static/{os.path.basename(chart_path)}" if chart_path else None),
            "pdf_url": (f"/static/{os.path.basename(pdf_path)}" if pdf_path else None)
        })
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate report."}), 500

@app.route("/conversation_starters")
@safe_route
def get_conversation_starters():
    starters = ENHANCED_RESPONSE_TEMPLATES['conversation_starters']
    return jsonify({"starters": random.sample(starters, min(3, len(starters)))})

@app.route("/feedback", methods=["POST"])
@safe_route
def collect_feedback():
    global bot
    if bot is None:
        return jsonify({"error": "Bot not initialized"}), 503

    try:
        data = request.get_json(force=True, silent=True) or {}
        response_id = data.get('response_id')
        rating = data.get('rating')
        if response_id:
            bot.response_feedback[response_id] = {
                'rating': rating,
                'timestamp': datetime.now()
            }
        return jsonify({"status": "feedback_recorded"})
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}", exc_info=True)
        return jsonify({"error": "Unable to record feedback"}), 500

# Example chat route (apply safe_route)
@app.route("/chat", methods=["POST"])
@safe_route
@app.route("/chat", methods=["POST"])
@safe_route
def chat():
    """Main chat endpoint for frontend chatbot"""
    data = request.get_json()  # frontend sends JSON { "message": "..." }
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"reply": "I didn’t catch that — could you say it again?"})

    try:
        # Get bot reply
        reply, analysis = bot.get_enhanced_response(user_message, user_id="default")

        return jsonify({
            "reply": reply,
            "emotion": analysis.get("dominant_emotion", {}).get("label", "neutral"),
            "intensity": analysis.get("intensity", "medium")
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"reply": "Sorry, I ran into an issue processing your message."}), 500


# Transformer setup (unchanged but safe)
def setup_advanced_transformers():
    models = {}
    if TRANSFORMERS_AVAILABLE:
        try:
            models['sentiment'] = pipeline("sentiment-analysis", return_all_scores=True)
            models['emotion'] = pipeline("text-classification", return_all_scores=True)
            try:
                models['mental_health'] = pipeline("text-classification", return_all_scores=True)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error setting up transformers: {e}")
    return models

if __name__ == "__main__":
    # instantiate bot before the server starts so routes can use it
    bot = OptimizedMindMateBot(data_file="data.csv")

    # start background cleanup thread
    def periodic_cleanup():
        while True:
            time.sleep(3600)  # Every hour
            try:
                bot.cleanup_memory()
                logger.info("Memory cleanup completed")
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}", exc_info=True)

    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

    # Optionally preload transformer models in the background (non-blocking)
    if TRANSFORMERS_AVAILABLE:
        def load_models_async():
            try:
                models = setup_advanced_transformers()
                logger.info(f"Transformers loaded: {list(models.keys())}")
            except Exception as e:
                logger.error(f"Transformers preload failed: {e}", exc_info=True)
        threading.Thread(target=load_models_async, daemon=True).start()

    app.run(debug=True, host='0.0.0.0', port=5000)




