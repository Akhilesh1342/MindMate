"""
Advanced Sentiment Analysis Models for MindMate
Includes multiple sentiment analysis approaches for comprehensive emotional understanding
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer combining multiple approaches for robust emotional analysis
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Multiple ML models for ensemble prediction
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        self.trained_models = {}
        self.is_trained = False
        
        # Emotional categories with more nuanced classifications
        self.emotion_categories = {
            'joy': ['happy', 'excited', 'elated', 'cheerful', 'content', 'pleased', 'thrilled'],
            'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'down', 'blue', 'miserable'],
            'anger': ['angry', 'furious', 'irritated', 'annoyed', 'mad', 'rage', 'frustrated'],
            'fear': ['anxious', 'worried', 'scared', 'terrified', 'nervous', 'afraid', 'panicked'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'startled'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened'],
            'neutral': ['neutral', 'calm', 'peaceful', 'serene', 'balanced']
        }
        
        # Stress and mental health indicators
        self.stress_indicators = [
            'stressed', 'overwhelmed', 'pressure', 'deadline', 'exhausted', 'burnout',
            'anxiety', 'panic', 'worry', 'concern', 'tension', 'strain'
        ]
        
        self.mental_health_indicators = [
            'depression', 'depressed', 'hopeless', 'worthless', 'suicidal', 'self-harm',
            'therapy', 'counseling', 'medication', 'mental health', 'psychiatrist',
            'kill myself', 'end my life', 'not worth living', 'want to die', 'better off dead',
            'harm myself', 'hurt myself', 'cut myself', 'overdose', 'take pills',
            'no friends', 'no one cares', 'completely alone', 'nobody loves me', 'no one understands',
            'give up', 'can\'t go on', 'no point', 'worthless'
        ]

    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)

    def extract_features(self, text):
        """Extract comprehensive features from text"""
        features = {}
        
        # Basic text features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Sentiment scores from different libraries
        blob = TextBlob(text)
        features['textblob_polarity'] = blob.sentiment.polarity
        features['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # VADER sentiment scores
        vader_scores = self.vader_analyzer.polarity_scores(text)
        features['vader_positive'] = vader_scores['pos']
        features['vader_negative'] = vader_scores['neg']
        features['vader_neutral'] = vader_scores['neu']
        features['vader_compound'] = vader_scores['compound']
        
        # Emotional intensity indicators
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Stress and mental health indicators
        features['stress_score'] = sum(1 for word in self.stress_indicators if word in text.lower())
        features['mental_health_score'] = sum(1 for word in self.mental_health_indicators if word in text.lower())
        
        return features

    def predict_emotion_category(self, text):
        """Predict emotion category using rule-based approach"""
        text_lower = text.lower()
        
        # Count matches for each emotion category
        emotion_scores = {}
        for emotion, keywords in self.emotion_categories.items():
            emotion_scores[emotion] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Return the emotion with highest score, default to neutral
        if max(emotion_scores.values()) == 0:
            return 'neutral'
        return max(emotion_scores, key=emotion_scores.get)

    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        # Preprocess training data
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        # Vectorize text
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_train_processed)
        X_count = self.count_vectorizer.fit_transform(X_train_processed)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train each model
        for name, model in self.models.items():
            if name in ['logistic_regression', 'svm']:
                model.fit(X_tfidf, y_encoded)
            else:  # random_forest
                model.fit(X_count, y_encoded)
            self.trained_models[name] = model
        
        self.is_trained = True

    def predict_sentiment_ensemble(self, text):
        """Predict sentiment using ensemble of models"""
        if not self.is_trained:
            return self.predict_emotion_category(text)
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        X_tfidf = self.tfidf_vectorizer.transform([processed_text])
        X_count = self.count_vectorizer.transform([processed_text])
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for name, model in self.trained_models.items():
            if name in ['logistic_regression', 'svm']:
                pred = model.predict(X_tfidf)[0]
                prob = model.predict_proba(X_tfidf)[0]
            else:  # random_forest
                pred = model.predict(X_count)[0]
                prob = model.predict_proba(X_count)[0]
            
            predictions.append(pred)
            probabilities.append(prob)
        
        # Ensemble prediction (majority vote)
        ensemble_pred = max(set(predictions), key=predictions.count)
        
        # Average probabilities
        avg_prob = np.mean(probabilities, axis=0)
        confidence = max(avg_prob)
        
        # Decode prediction
        predicted_label = self.label_encoder.inverse_transform([ensemble_pred])[0]
        
        return predicted_label, confidence

    def comprehensive_analysis(self, text):
        """Perform comprehensive sentiment and emotional analysis"""
        if not text or not isinstance(text, str):
            return {
                'emotion': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'stress_level': 'low',
                'mental_health_concern': False,
                'recommendations': ['Consider journaling your thoughts'],
                'analysis_details': {}
            }
        
        # Get ensemble prediction
        if self.is_trained:
            emotion, confidence = self.predict_sentiment_ensemble(text)
        else:
            emotion = self.predict_emotion_category(text)
            confidence = 0.7  # Default confidence for rule-based
        
        # Extract features
        features = self.extract_features(text)
        
        # Calculate overall sentiment score
        sentiment_score = (features['textblob_polarity'] + features['vader_compound']) / 2
        
        # Determine stress level
        stress_level = 'low'
        if features['stress_score'] >= 3:
            stress_level = 'high'
        elif features['stress_score'] >= 1:
            stress_level = 'medium'
        
        # Check for mental health concerns
        mental_health_concern = features['mental_health_score'] > 0
        
        # Generate recommendations
        recommendations = self.generate_recommendations(emotion, stress_level, mental_health_concern, features)
        
        return {
            'emotion': emotion,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'stress_level': stress_level,
            'mental_health_concern': mental_health_concern,
            'recommendations': recommendations,
            'analysis_details': {
                'textblob_polarity': features['textblob_polarity'],
                'textblob_subjectivity': features['textblob_subjectivity'],
                'vader_compound': features['vader_compound'],
                'word_count': features['word_count'],
                'stress_score': features['stress_score'],
                'mental_health_score': features['mental_health_score']
            }
        }

    def generate_recommendations(self, emotion, stress_level, mental_health_concern, features):
        """Generate personalized recommendations based on analysis"""
        recommendations = []
        
        # Emotion-based recommendations
        if emotion == 'sadness':
            recommendations.extend([
                "Consider talking to a trusted friend or family member",
                "Try engaging in activities you enjoy",
                "Practice self-compassion and remind yourself that feelings are temporary"
            ])
        elif emotion == 'anger':
            recommendations.extend([
                "Take deep breaths and count to 10",
                "Try physical exercise to release tension",
                "Consider what's really bothering you beneath the surface"
            ])
        elif emotion == 'fear' or emotion == 'anxiety':
            recommendations.extend([
                "Practice grounding techniques (5-4-3-2-1 method)",
                "Try progressive muscle relaxation",
                "Focus on what you can control right now"
            ])
        elif emotion == 'joy':
            recommendations.extend([
                "Celebrate this positive moment",
                "Share your joy with others",
                "Consider what contributed to this feeling"
            ])
        
        # Stress level recommendations
        if stress_level == 'high':
            recommendations.extend([
                "Consider taking a break or reducing your workload",
                "Practice mindfulness or meditation",
                "Ensure you're getting adequate sleep and nutrition"
            ])
        elif stress_level == 'medium':
            recommendations.extend([
                "Try time management techniques",
                "Practice regular relaxation exercises",
                "Consider delegating some tasks"
            ])
        
        # Mental health concern recommendations
        if mental_health_concern:
            recommendations.extend([
                "Consider speaking with a mental health professional",
                "Reach out to a crisis helpline if you're in immediate distress",
                "Remember that seeking help is a sign of strength"
            ])
        
        # General recommendations
        if features['word_count'] < 5:
            recommendations.append("Consider expressing more of your thoughts and feelings")
        
        if features['exclamation_count'] > 3:
            recommendations.append("High emotional intensity detected - consider calming techniques")
        
        return recommendations[:5]  # Limit to 5 recommendations

    def save_models(self, filepath='models/sentiment_models.pkl'):
        """Save trained models"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'trained_models': self.trained_models,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_models(self, filepath='models/sentiment_models.pkl'):
        """Load trained models"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                self.trained_models = model_data['trained_models']
                self.tfidf_vectorizer = model_data['tfidf_vectorizer']
                self.count_vectorizer = model_data['count_vectorizer']
                self.label_encoder = model_data['label_encoder']
                self.is_trained = model_data['is_trained']
