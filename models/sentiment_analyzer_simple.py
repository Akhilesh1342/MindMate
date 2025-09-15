"""
Simple Sentiment Analyzer for MindMate (No NLTK Required)
Provides basic sentiment analysis without external dependencies
"""

import re
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

class SimpleSentimentAnalyzer:
    """
    Simple sentiment analyzer that works without NLTK
    """
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Emotional categories with keywords
        self.emotion_categories = {
            'joy': ['happy', 'excited', 'elated', 'cheerful', 'content', 'pleased', 'thrilled', 'grateful', 'optimistic', 'proud', 'energized'],
            'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'down', 'blue', 'miserable', 'lonely', 'hopeless', 'worthless'],
            'anger': ['angry', 'furious', 'irritated', 'annoyed', 'mad', 'rage', 'frustrated', 'enraged'],
            'fear': ['anxious', 'worried', 'scared', 'terrified', 'nervous', 'afraid', 'panicked', 'fearful'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'startled', 'bewildered'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled'],
            'neutral': ['neutral', 'calm', 'peaceful', 'serene', 'balanced', 'content']
        }
        
        # Stress indicators
        self.stress_indicators = [
            'stressed', 'overwhelmed', 'pressure', 'deadline', 'exhausted', 'burnout',
            'anxiety', 'panic', 'worry', 'concern', 'tension', 'strain', 'burden'
        ]
        
        # Mental health indicators
        self.mental_health_indicators = [
            'depression', 'depressed', 'hopeless', 'worthless', 'suicidal', 'self-harm',
            'kill myself', 'end my life', 'not worth living', 'want to die', 'better off dead',
            'harm myself', 'hurt myself', 'cut myself', 'overdose', 'take pills',
            'no friends', 'no one cares', 'completely alone', 'nobody loves me', 'no one understands',
            'give up', 'can\'t go on', 'no point'
        ]

    def simple_preprocess_text(self, text):
        """Simple text preprocessing without NLTK"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text

    def predict_emotion_category(self, text):
        """Predict emotion category using keyword matching with priority for critical concerns"""
        text_lower = text.lower()
        
        # Check for critical mental health concerns first (highest priority)
        critical_keywords = [
            'kill myself', 'suicide', 'end my life', 'not worth living',
            'want to die', 'better off dead', 'harm myself', 'hurt myself',
            'self harm', 'cut myself', 'overdose', 'take pills'
        ]
        
        if any(keyword in text_lower for keyword in critical_keywords):
            return 'sadness'  # Critical concerns are typically sadness/depression
        
        # Check for severe loneliness and isolation
        loneliness_keywords = ['no friends', 'no one cares', 'completely alone', 'nobody loves me', 'no one understands']
        if any(keyword in text_lower for keyword in loneliness_keywords):
            return 'sadness'
        
        # Check for severe depression indicators
        depression_keywords = ['hopeless', 'worthless', 'no point', 'give up', 'can\'t go on']
        if any(keyword in text_lower for keyword in depression_keywords):
            return 'sadness'
        
        # Count matches for each emotion category
        emotion_scores = {}
        for emotion, keywords in self.emotion_categories.items():
            emotion_scores[emotion] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Also check for emotional intensity indicators
        intensity_words = ['really', 'so', 'very', 'extremely', 'completely', 'totally']
        intensity_boost = sum(1 for word in intensity_words if word in text_lower)
        
        # Boost emotion scores based on intensity
        for emotion in emotion_scores:
            emotion_scores[emotion] += intensity_boost
        
        # Return the emotion with highest score, default to neutral
        if max(emotion_scores.values()) == 0:
            return 'neutral'
        return max(emotion_scores, key=emotion_scores.get)

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
        
        # Get emotion prediction
        emotion = self.predict_emotion_category(text)
        
        # Get sentiment scores
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # VADER sentiment scores
        vader_scores = self.vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # Calculate overall sentiment score with emotional intensity adjustment
        base_sentiment = (textblob_polarity + vader_compound) / 2
        
        # Adjust sentiment based on detected emotion
        if emotion == 'sadness':
            sentiment_score = min(-0.5, base_sentiment - 0.3)  # Ensure sadness shows negative sentiment
        elif emotion == 'anger':
            sentiment_score = min(-0.3, base_sentiment - 0.2)  # Ensure anger shows negative sentiment
        elif emotion == 'fear':
            sentiment_score = min(-0.2, base_sentiment - 0.1)  # Ensure fear shows negative sentiment
        elif emotion == 'joy':
            sentiment_score = max(0.5, base_sentiment + 0.3)   # Ensure joy shows positive sentiment
        else:
            sentiment_score = base_sentiment
        
        # Determine stress level
        stress_level = 'low'
        stress_score = sum(1 for word in self.stress_indicators if word in text.lower())
        if stress_score >= 3:
            stress_level = 'high'
        elif stress_score >= 1:
            stress_level = 'medium'
        
        # Check for mental health concerns
        mental_health_concern = any(word in text.lower() for word in self.mental_health_indicators)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(emotion, stress_level, mental_health_concern, text)
        
        # Calculate confidence based on emotional intensity and keyword matches
        confidence = 0.7  # Default confidence for rule-based approach
        
        # Increase confidence for critical concerns
        if mental_health_concern:
            confidence = 0.95  # Very high confidence for critical concerns
        
        # Increase confidence for strong emotional indicators
        intensity_words = ['really', 'so', 'very', 'extremely', 'completely', 'totally']
        if any(word in text.lower() for word in intensity_words):
            confidence = min(0.9, confidence + 0.1)
        
        # Increase confidence for multiple emotion keywords
        emotion_keyword_count = sum(1 for emotion_keywords in self.emotion_categories.values() 
                                   for keyword in emotion_keywords if keyword in text.lower())
        if emotion_keyword_count > 1:
            confidence = min(0.9, confidence + 0.1)
        
        return {
            'emotion': emotion,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'stress_level': stress_level,
            'mental_health_concern': mental_health_concern,
            'recommendations': recommendations,
            'analysis_details': {
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'vader_compound': vader_compound,
                'word_count': len(text.split()),
                'stress_score': stress_score,
                'mental_health_score': sum(1 for word in self.mental_health_indicators if word in text.lower())
            }
        }

    def generate_recommendations(self, emotion, stress_level, mental_health_concern, text):
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
        
        return recommendations[:5]  # Limit to 5 recommendations
