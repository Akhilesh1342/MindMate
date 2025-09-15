"""
Configuration file for MindMate
Contains all configuration settings and constants
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///mindmate.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Application Settings
    APP_NAME = 'MindMate'
    APP_VERSION = '2.0.0'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Sentiment Analysis Settings
    SENTIMENT_MODEL_PATH = 'models/sentiment_models.pkl'
    MIN_CONFIDENCE_THRESHOLD = 0.3
    MAX_RECOMMENDATIONS = 5
    
    # Data Processing Settings
    MAX_JOURNAL_LENGTH = 5000
    MIN_JOURNAL_LENGTH = 5
    MAX_MOOD_ENTRIES_PER_DAY = 50
    
    # Analytics Settings
    DEFAULT_ANALYTICS_DAYS = 30
    MAX_ANALYTICS_DAYS = 365
    MIN_ENTRIES_FOR_ANALYTICS = 3
    
    # Report Generation Settings
    REPORTS_DIR = 'static/reports'
    CHARTS_DIR = 'static/charts'
    MAX_REPORT_DAYS = 90
    
    # Security Settings
    SESSION_TIMEOUT = timedelta(hours=24)
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=30)
    
    # File Upload Settings
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt'}
    
    # Email Settings (for future notifications)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # External API Settings
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GOOGLE_ANALYTICS_ID = os.environ.get('GOOGLE_ANALYTICS_ID')
    
    # Feature Flags
    ENABLE_ADVANCED_ANALYTICS = True
    ENABLE_PREDICTIVE_INSIGHTS = True
    ENABLE_EMAIL_NOTIFICATIONS = False
    ENABLE_EXTERNAL_APIS = False
    
    # Emotional Categories Configuration
    EMOTION_CATEGORIES = {
        'joy': {
            'keywords': ['happy', 'excited', 'elated', 'cheerful', 'content', 'pleased', 'thrilled', 'grateful', 'optimistic'],
            'color': '#28B463',
            'icon': '😊'
        },
        'sadness': {
            'keywords': ['sad', 'depressed', 'melancholy', 'gloomy', 'down', 'blue', 'miserable', 'lonely', 'hopeless'],
            'color': '#3498DB',
            'icon': '😢'
        },
        'anger': {
            'keywords': ['angry', 'furious', 'irritated', 'annoyed', 'mad', 'rage', 'frustrated', 'enraged'],
            'color': '#E74C3C',
            'icon': '😠'
        },
        'fear': {
            'keywords': ['anxious', 'worried', 'scared', 'terrified', 'nervous', 'afraid', 'panicked', 'fearful'],
            'color': '#9B59B6',
            'icon': '😰'
        },
        'surprise': {
            'keywords': ['surprised', 'shocked', 'amazed', 'astonished', 'startled', 'bewildered'],
            'color': '#F39C12',
            'icon': '😲'
        },
        'disgust': {
            'keywords': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled'],
            'color': '#95A5A6',
            'icon': '🤢'
        },
        'neutral': {
            'keywords': ['neutral', 'calm', 'peaceful', 'serene', 'balanced', 'content'],
            'color': '#BDC3C7',
            'icon': '😐'
        }
    }
    
    # Stress Level Configuration
    STRESS_LEVELS = {
        'low': {
            'color': '#27AE60',
            'description': 'Minimal stress, feeling relaxed',
            'recommendations': ['Maintain current routine', 'Continue self-care practices']
        },
        'medium': {
            'color': '#F39C12',
            'description': 'Moderate stress, some tension',
            'recommendations': ['Practice relaxation techniques', 'Take regular breaks', 'Consider time management']
        },
        'high': {
            'color': '#E74C3C',
            'description': 'High stress, feeling overwhelmed',
            'recommendations': ['Seek support', 'Reduce workload', 'Practice stress management', 'Consider professional help']
        }
    }
    
    # Mental Health Concern Indicators
    MENTAL_HEALTH_INDICATORS = [
        'depression', 'depressed', 'hopeless', 'worthless', 'suicidal', 'self-harm',
        'therapy', 'counseling', 'medication', 'mental health', 'psychiatrist',
        'psychologist', 'anxiety disorder', 'panic attack', 'eating disorder',
        'substance abuse', 'addiction', 'trauma', 'ptsd', 'bipolar', 'ocd'
    ]
    
    # Stress Indicators
    STRESS_INDICATORS = [
        'stressed', 'overwhelmed', 'pressure', 'deadline', 'exhausted', 'burnout',
        'anxiety', 'panic', 'worry', 'concern', 'tension', 'strain', 'burden',
        'demanding', 'hectic', 'rushed', 'urgent', 'critical', 'important'
    ]
    
    # Recommendation Categories
    RECOMMENDATION_CATEGORIES = {
        'emotion': {
            'name': 'Emotional Support',
            'description': 'Recommendations for emotional well-being',
            'icon': '💝'
        },
        'stress': {
            'name': 'Stress Management',
            'description': 'Recommendations for managing stress',
            'icon': '🧘'
        },
        'mental_health': {
            'name': 'Mental Health',
            'description': 'Recommendations for mental health support',
            'icon': '🏥'
        },
        'general': {
            'name': 'General Wellness',
            'description': 'General wellness recommendations',
            'icon': '🌟'
        }
    }
    
    # Chart Colors
    CHART_COLORS = [
        '#007BFF', '#FF5733', '#28B463', '#F1C40F', '#A569BD',
        '#E74C3C', '#3498DB', '#9B59B6', '#F39C12', '#95A5A6'
    ]
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE = 60
    RATE_LIMIT_PER_HOUR = 1000
    
    # Database Settings
    DATABASE_POOL_SIZE = 10
    DATABASE_POOL_TIMEOUT = 20
    DATABASE_POOL_RECYCLE = 3600

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///mindmate_dev.db'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://user:password@localhost/mindmate'
    
    # Security settings for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
