"""
Enhanced MindMate Backend with Advanced Sentiment Analysis
A comprehensive mental health companion with sophisticated ML models
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime, timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import plotly.graph_objs as go
import plotly.utils
import json
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

# Import our custom modules
from models.database import db, User, MoodEntry, ChatSession, ChatMessage, Recommendation, MoodInsight, DatabaseUtils
from models.sentiment_analyzer import AdvancedSentimentAnalyzer
from services.chat_service import IntelligentChatService

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///mindmate.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)
CORS(app)

# Initialize services
sentiment_analyzer = AdvancedSentimentAnalyzer()
chat_service = IntelligentChatService()

# Create tables
with app.app_context():
    db.create_all()

# Utility functions
def require_auth(f):
    """Decorator to require authentication"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def get_current_user():
    """Get current user from session"""
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

# Authentication Routes
@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        
        if not all([username, email, password]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create new user
        user = User(
            username=username,
            email=email
        )
        # Note: In production, you'd hash the password
        # user.password_hash = generate_password_hash(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Log user in
        session['user_id'] = user.id
        
        return jsonify({
            'message': 'User created successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Log user in
        session['user_id'] = user.id
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'}), 200

# Mood Tracking Routes
@app.route('/api/log-mood', methods=['POST'])
@require_auth
def api_log_mood():
    """Log mood entry with advanced sentiment analysis"""
    try:
        data = request.get_json()
        mood_label = data.get('mood', '').strip()
        journal_text = data.get('journal', '').strip()
        
        if not mood_label:
            return jsonify({'error': 'Mood is required'}), 400
        
        user_id = session['user_id']
        
        # Perform comprehensive sentiment analysis
        analysis = sentiment_analyzer.comprehensive_analysis(journal_text)
        
        # Create mood entry
        mood_entry = MoodEntry(
            user_id=user_id,
            mood_label=mood_label,
            journal_text=journal_text,
            sentiment_score=analysis['sentiment_score'],
            emotion_category=analysis['emotion'],
            confidence_score=analysis['confidence'],
            stress_level=analysis['stress_level'],
            mental_health_concern=analysis['mental_health_concern']
        )
        
        # Store analysis details
        mood_entry.set_analysis_details(analysis['analysis_details'])
        
        db.session.add(mood_entry)
        
        # Store recommendations
        for rec_text in analysis['recommendations']:
            category = 'general'
            if analysis['mental_health_concern']:
                category = 'mental_health'
            elif analysis['stress_level'] in ['high', 'medium']:
                category = 'stress'
            elif analysis['emotion'] != 'neutral':
                category = 'emotion'
            
            recommendation = Recommendation(
                user_id=user_id,
                mood_entry_id=mood_entry.id,
                recommendation_text=rec_text,
                category=category
            )
            db.session.add(recommendation)
        
        db.session.commit()
        
        # Create daily insight
        DatabaseUtils.create_mood_insight(user_id, datetime.utcnow().date(), [mood_entry])
        
        return jsonify({
            'message': 'Mood logged successfully',
            'analysis': analysis,
            'mood_entry_id': mood_entry.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/mood-data')
@require_auth
def api_mood_data():
    """Get comprehensive mood data for charts and analysis"""
    try:
        user_id = session['user_id']
        days = request.args.get('days', 30, type=int)
        
        # Get mood summary
        summary = DatabaseUtils.get_user_mood_summary(user_id, days)
        
        # Get mood entries for the period
        start_date = datetime.utcnow() - timedelta(days=days)
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.desc()).all()
        
        # Prepare data for charts
        chart_data = {
            'timeline': [],
            'emotion_distribution': summary['emotion_distribution'],
            'stress_levels': summary['stress_levels'],
            'sentiment_trend': []
        }
        
        # Timeline data
        for entry in entries:
            chart_data['timeline'].append({
                'date': entry.timestamp.isoformat(),
                'mood': entry.mood_label,
                'emotion': entry.emotion_category,
                'sentiment_score': entry.sentiment_score,
                'stress_level': entry.stress_level
            })
        
        # Sentiment trend (daily averages)
        daily_sentiments = {}
        for entry in entries:
            date = entry.timestamp.date()
            if date not in daily_sentiments:
                daily_sentiments[date] = []
            if entry.sentiment_score is not None:
                daily_sentiments[date].append(entry.sentiment_score)
        
        for date, scores in daily_sentiments.items():
            chart_data['sentiment_trend'].append({
                'date': date.isoformat(),
                'avg_sentiment': sum(scores) / len(scores),
                'entry_count': len(scores)
            })
        
        return jsonify({
            'summary': summary,
            'chart_data': chart_data,
            'total_entries': len(entries)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Chat Routes
@app.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    """Enhanced chat with sentiment analysis"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        user_id = session['user_id']
        
        # Create session if not provided
        if not session_id:
            session_id = chat_service.create_chat_session(user_id)
            if not session_id:
                return jsonify({'error': 'Failed to create chat session'}), 500
        
        # Get intelligent response
        response_data = chat_service.get_contextual_response(
            message, user_id, session_id
        )
        
        return jsonify({
            'reply': response_data['response'],
            'session_id': session_id,
            'sentiment_analysis': response_data['sentiment_analysis'],
            'recommendations': response_data['recommendations']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-sessions')
@require_auth
def get_chat_sessions():
    """Get user's chat sessions"""
    try:
        user_id = session['user_id']
        sessions = chat_service.get_user_sessions(user_id)
        
        return jsonify({'sessions': sessions}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-history/<int:session_id>')
@require_auth
def get_chat_history(session_id):
    """Get chat history for a session"""
    try:
        # Verify session belongs to user
        session_obj = ChatSession.query.get(session_id)
        if not session_obj or session_obj.user_id != session['user_id']:
            return jsonify({'error': 'Session not found'}), 404
        
        history = chat_service.get_chat_history(session_id)
        
        return jsonify({'history': history}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Analytics and Insights Routes
@app.route('/api/insights')
@require_auth
def get_insights():
    """Get comprehensive mood insights and analytics"""
    try:
        user_id = session['user_id']
        days = request.args.get('days', 30, type=int)
        
        # Get mood summary
        summary = DatabaseUtils.get_user_mood_summary(user_id, days)
        
        # Get recommendations
        recommendations = DatabaseUtils.get_user_recommendations(user_id, 10)
        
        # Calculate insights
        insights = {
            'mood_summary': summary,
            'recommendations': recommendations,
            'patterns': _analyze_patterns(user_id, days),
            'wellbeing_score': _calculate_wellbeing_score(summary)
        }
        
        return jsonify(insights), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _analyze_patterns(user_id, days):
    """Analyze mood patterns and trends"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    entries = MoodEntry.query.filter(
        MoodEntry.user_id == user_id,
        MoodEntry.timestamp >= start_date
    ).order_by(MoodEntry.timestamp.asc()).all()
    
    if len(entries) < 3:
        return {'message': 'Insufficient data for pattern analysis'}
    
    patterns = {
        'weekly_pattern': _analyze_weekly_pattern(entries),
        'time_pattern': _analyze_time_pattern(entries),
        'stress_trend': _analyze_stress_trend(entries),
        'emotion_transitions': _analyze_emotion_transitions(entries)
    }
    
    return patterns

def _analyze_weekly_pattern(entries):
    """Analyze mood patterns by day of week"""
    weekly_data = {}
    for entry in entries:
        day = entry.timestamp.strftime('%A')
        if day not in weekly_data:
            weekly_data[day] = []
        if entry.sentiment_score is not None:
            weekly_data[day].append(entry.sentiment_score)
    
    weekly_avg = {}
    for day, scores in weekly_data.items():
        weekly_avg[day] = sum(scores) / len(scores)
    
    return weekly_avg

def _analyze_time_pattern(entries):
    """Analyze mood patterns by time of day"""
    time_data = {}
    for entry in entries:
        hour = entry.timestamp.hour
        time_period = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening'
        
        if time_period not in time_data:
            time_data[time_period] = []
        if entry.sentiment_score is not None:
            time_data[time_period].append(entry.sentiment_score)
    
    time_avg = {}
    for period, scores in time_data.items():
        time_avg[period] = sum(scores) / len(scores)
    
    return time_avg

def _analyze_stress_trend(entries):
    """Analyze stress level trends"""
    stress_levels = {'low': 0, 'medium': 0, 'high': 0}
    for entry in entries:
        if entry.stress_level:
            stress_levels[entry.stress_level] += 1
    
    return stress_levels

def _analyze_emotion_transitions(entries):
    """Analyze transitions between emotions"""
    transitions = {}
    for i in range(len(entries) - 1):
        current_emotion = entries[i].emotion_category or 'unknown'
        next_emotion = entries[i + 1].emotion_category or 'unknown'
        transition = f"{current_emotion} -> {next_emotion}"
        transitions[transition] = transitions.get(transition, 0) + 1
    
    return transitions

def _calculate_wellbeing_score(summary):
    """Calculate overall wellbeing score"""
    if summary['total_entries'] == 0:
        return 50  # Neutral score for no data
    
    # Base score from average sentiment
    base_score = (summary['avg_sentiment'] + 1) * 50  # Convert -1,1 to 0,100
    
    # Adjustments
    stress_penalty = summary['stress_levels'].get('high', 0) * 5
    mental_health_penalty = summary['mental_health_concerns'] * 10
    
    # Trend bonus
    trend_bonus = 0
    if summary['recent_trend'] == 'improving':
        trend_bonus = 10
    elif summary['recent_trend'] == 'declining':
        trend_bonus = -10
    
    wellbeing_score = base_score - stress_penalty - mental_health_penalty + trend_bonus
    return max(0, min(100, wellbeing_score))

# Report Generation Routes
@app.route('/api/generate-report')
@require_auth
def generate_report():
    """Generate comprehensive mood report"""
    try:
        user_id = session['user_id']
        days = request.args.get('days', 30, type=int)
        
        # Get data
        summary = DatabaseUtils.get_user_mood_summary(user_id, days)
        recommendations = DatabaseUtils.get_user_recommendations(user_id, 20)
        
        # Generate charts
        chart_paths = _generate_report_charts(user_id, days)
        
        # Generate PDF report
        pdf_path = _generate_pdf_report(user_id, summary, recommendations, chart_paths)
        
        return jsonify({
            'message': 'Report generated successfully',
            'pdf_path': pdf_path,
            'chart_paths': chart_paths,
            'summary': summary
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _generate_report_charts(user_id, days):
    """Generate charts for the report"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    entries = MoodEntry.query.filter(
        MoodEntry.user_id == user_id,
        MoodEntry.timestamp >= start_date
    ).order_by(MoodEntry.timestamp.asc()).all()
    
    if not entries:
        return {}
    
    # Create charts directory
    charts_dir = 'static/charts'
    os.makedirs(charts_dir, exist_ok=True)
    
    chart_paths = {}
    
    # Mood distribution pie chart
    mood_counts = {}
    for entry in entries:
        mood = entry.emotion_category or entry.mood_label
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
    
    if mood_counts:
        plt.figure(figsize=(8, 6))
        plt.pie(mood_counts.values(), labels=mood_counts.keys(), autopct='%1.1f%%')
        plt.title('Emotion Distribution')
        chart_paths['emotion_distribution'] = f'{charts_dir}/emotion_distribution.png'
        plt.savefig(chart_paths['emotion_distribution'])
        plt.close()
    
    # Sentiment trend line chart
    daily_sentiments = {}
    for entry in entries:
        date = entry.timestamp.date()
        if date not in daily_sentiments:
            daily_sentiments[date] = []
        if entry.sentiment_score is not None:
            daily_sentiments[date].append(entry.sentiment_score)
    
    if daily_sentiments:
        dates = sorted(daily_sentiments.keys())
        avg_sentiments = [sum(daily_sentiments[date]) / len(daily_sentiments[date]) for date in dates]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, avg_sentiments, marker='o')
        plt.title('Sentiment Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_paths['sentiment_trend'] = f'{charts_dir}/sentiment_trend.png'
        plt.savefig(chart_paths['sentiment_trend'])
        plt.close()
    
    return chart_paths

def _generate_pdf_report(user_id, summary, recommendations, chart_paths):
    """Generate PDF report"""
    user = User.query.get(user_id)
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MindMate Mental Health Report", ln=True, align="C")
    pdf.ln(10)
    
    # User info
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated for: {user.username}", ln=True)
    pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(10)
    
    # Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Total Entries: {summary['total_entries']}", ln=True)
    pdf.cell(0, 10, f"Average Sentiment: {summary['avg_sentiment']:.2f}", ln=True)
    pdf.cell(0, 10, f"Recent Trend: {summary['recent_trend']}", ln=True)
    pdf.cell(0, 10, f"Mental Health Concerns: {summary['mental_health_concerns']}", ln=True)
    pdf.ln(10)
    
    # Charts
    if chart_paths:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Charts", ln=True)
        
        for chart_name, chart_path in chart_paths.items():
            if os.path.exists(chart_path):
                pdf.image(chart_path, x=25, w=160)
                pdf.ln(5)
    
    # Recommendations
    if recommendations:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Recommendations", ln=True)
        pdf.set_font("Arial", "", 12)
        
        for i, rec in enumerate(recommendations[:10], 1):
            pdf.cell(0, 10, f"{i}. {rec['recommendation_text']}", ln=True)
    
    # Save PDF
    reports_dir = 'static/reports'
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = f'{reports_dir}/report_{user_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    pdf.output(pdf_path)
    
    return pdf_path

# Frontend Routes (keeping existing templates)
@app.route("/")
def landing():
    return render_template("login_enhanced.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/mood-tracker-page")
def mood_tracker_page():
    return render_template("mood_tracker.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/report")
def report():
    return redirect(url_for('generate_report'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
