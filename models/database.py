"""
Database models and configuration for MindMate
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import func, desc
import json

db = SQLAlchemy()

class User(db.Model):
    """User model for storing user information"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    mood_entries = db.relationship('MoodEntry', backref='user', lazy=True, cascade='all, delete-orphan')
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class MoodEntry(db.Model):
    """Mood tracking entries"""
    __tablename__ = 'mood_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    mood_label = db.Column(db.String(50), nullable=False)  # happy, sad, angry, etc.
    journal_text = db.Column(db.Text)
    
    # Advanced sentiment analysis results
    sentiment_score = db.Column(db.Float)  # -1 to 1
    emotion_category = db.Column(db.String(50))  # joy, sadness, anger, etc.
    confidence_score = db.Column(db.Float)  # 0 to 1
    stress_level = db.Column(db.String(20))  # low, medium, high
    mental_health_concern = db.Column(db.Boolean, default=False)
    
    # Analysis details as JSON
    analysis_details = db.Column(db.Text)  # JSON string
    
    def set_analysis_details(self, details_dict):
        """Set analysis details as JSON string"""
        self.analysis_details = json.dumps(details_dict)
    
    def get_analysis_details(self):
        """Get analysis details as dictionary"""
        if self.analysis_details:
            return json.loads(self.analysis_details)
        return {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'mood_label': self.mood_label,
            'journal_text': self.journal_text,
            'sentiment_score': self.sentiment_score,
            'emotion_category': self.emotion_category,
            'confidence_score': self.confidence_score,
            'stress_level': self.stress_level,
            'mental_health_concern': self.mental_health_concern,
            'analysis_details': self.get_analysis_details()
        }

class ChatSession(db.Model):
    """Chat sessions for conversation tracking"""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_name': self.session_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'message_count': len(self.messages)
        }

class ChatMessage(db.Model):
    """Individual chat messages"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    message_type = db.Column(db.String(20), nullable=False)  # 'user' or 'bot'
    content = db.Column(db.Text, nullable=False)
    
    # Sentiment analysis for user messages
    sentiment_score = db.Column(db.Float)
    emotion_category = db.Column(db.String(50))
    confidence_score = db.Column(db.Float)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'message_type': self.message_type,
            'content': self.content,
            'sentiment_score': self.sentiment_score,
            'emotion_category': self.emotion_category,
            'confidence_score': self.confidence_score
        }

class Recommendation(db.Model):
    """Stored recommendations for users"""
    __tablename__ = 'recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    mood_entry_id = db.Column(db.Integer, db.ForeignKey('mood_entries.id'))
    recommendation_text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))  # emotion, stress, mental_health, general
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'mood_entry_id': self.mood_entry_id,
            'recommendation_text': self.recommendation_text,
            'category': self.category,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_read': self.is_read
        }

class MoodInsight(db.Model):
    """Aggregated mood insights and patterns"""
    __tablename__ = 'mood_insights'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    
    # Daily aggregated metrics
    avg_sentiment_score = db.Column(db.Float)
    dominant_emotion = db.Column(db.String(50))
    stress_level = db.Column(db.String(20))
    entry_count = db.Column(db.Integer, default=0)
    mental_health_concerns = db.Column(db.Integer, default=0)
    
    # Weekly/Monthly patterns
    mood_trend = db.Column(db.String(20))  # improving, declining, stable
    stress_trend = db.Column(db.String(20))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'date': self.date.isoformat() if self.date else None,
            'avg_sentiment_score': self.avg_sentiment_score,
            'dominant_emotion': self.dominant_emotion,
            'stress_level': self.stress_level,
            'entry_count': self.entry_count,
            'mental_health_concerns': self.mental_health_concerns,
            'mood_trend': self.mood_trend,
            'stress_trend': self.stress_trend,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Database utility functions
class DatabaseUtils:
    """Utility functions for database operations"""
    
    @staticmethod
    def get_user_mood_summary(user_id, days=30):
        """Get mood summary for a user over specified days"""
        from datetime import datetime, timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(desc(MoodEntry.timestamp)).all()
        
        if not entries:
            return {
                'total_entries': 0,
                'avg_sentiment': 0.0,
                'emotion_distribution': {},
                'stress_levels': {},
                'mental_health_concerns': 0,
                'recent_trend': 'stable'
            }
        
        # Calculate metrics
        total_entries = len(entries)
        avg_sentiment = sum(e.sentiment_score for e in entries if e.sentiment_score) / total_entries
        
        emotion_distribution = {}
        stress_levels = {}
        mental_health_concerns = 0
        
        for entry in entries:
            # Emotion distribution
            emotion = entry.emotion_category or 'unknown'
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            # Stress levels
            stress = entry.stress_level or 'unknown'
            stress_levels[stress] = stress_levels.get(stress, 0) + 1
            
            # Mental health concerns
            if entry.mental_health_concern:
                mental_health_concerns += 1
        
        # Calculate trend (comparing first half vs second half)
        if total_entries >= 4:
            mid_point = total_entries // 2
            first_half_avg = sum(e.sentiment_score for e in entries[:mid_point] if e.sentiment_score) / mid_point
            second_half_avg = sum(e.sentiment_score for e in entries[mid_point:] if e.sentiment_score) / (total_entries - mid_point)
            
            if second_half_avg > first_half_avg + 0.1:
                recent_trend = 'improving'
            elif second_half_avg < first_half_avg - 0.1:
                recent_trend = 'declining'
            else:
                recent_trend = 'stable'
        else:
            recent_trend = 'insufficient_data'
        
        return {
            'total_entries': total_entries,
            'avg_sentiment': avg_sentiment,
            'emotion_distribution': emotion_distribution,
            'stress_levels': stress_levels,
            'mental_health_concerns': mental_health_concerns,
            'recent_trend': recent_trend,
            'entries': [entry.to_dict() for entry in entries[:10]]  # Last 10 entries
        }
    
    @staticmethod
    def get_user_recommendations(user_id, limit=10):
        """Get recent recommendations for a user"""
        recommendations = Recommendation.query.filter(
            Recommendation.user_id == user_id
        ).order_by(desc(Recommendation.created_at)).limit(limit).all()
        
        return [rec.to_dict() for rec in recommendations]
    
    @staticmethod
    def create_mood_insight(user_id, date, entries):
        """Create or update daily mood insight"""
        insight = MoodInsight.query.filter(
            MoodInsight.user_id == user_id,
            MoodInsight.date == date
        ).first()
        
        if not insight:
            insight = MoodInsight(user_id=user_id, date=date)
            db.session.add(insight)
        
        if entries:
            # Calculate aggregated metrics
            sentiment_scores = [e.sentiment_score for e in entries if e.sentiment_score is not None]
            insight.avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Dominant emotion
            emotions = [e.emotion_category for e in entries if e.emotion_category]
            if emotions:
                insight.dominant_emotion = max(set(emotions), key=emotions.count)
            
            # Stress level
            stress_levels = [e.stress_level for e in entries if e.stress_level]
            if stress_levels:
                insight.stress_level = max(set(stress_levels), key=stress_levels.count)
            
            insight.entry_count = len(entries)
            insight.mental_health_concerns = sum(1 for e in entries if e.mental_health_concern)
        
        db.session.commit()
        return insight
