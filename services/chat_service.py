"""
Advanced Chat Service for MindMate
Provides intelligent responses based on sentiment analysis and emotional understanding
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
try:
    from models.sentiment_analyzer import AdvancedSentimentAnalyzer
    SENTIMENT_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from models.sentiment_analyzer_simple import SimpleSentimentAnalyzer as AdvancedSentimentAnalyzer
        SENTIMENT_ANALYZER_AVAILABLE = True
        print("⚠️ Using simple sentiment analyzer (NLTK not available)")
    except ImportError:
        AdvancedSentimentAnalyzer = None
        SENTIMENT_ANALYZER_AVAILABLE = False
        print("❌ No sentiment analyzer available")
from models.database import db, ChatSession, ChatMessage, Recommendation
from datetime import datetime
import random

class IntelligentChatService:
    """
    Advanced chat service that provides contextual and emotionally aware responses
    """
    
    def __init__(self):
        if SENTIMENT_ANALYZER_AVAILABLE:
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        else:
            self.sentiment_analyzer = None
        self.qa_data = None
        self.vectorizer = None
        self.response_templates = self._load_response_templates()
        self.load_qa_data()
    
    def _load_response_templates(self):
        """Load response templates for different emotional states"""
        return {
            'joy': [
                "I'm so glad to hear you're feeling positive! What's contributing to this good mood?",
                "That's wonderful! Celebrating the good moments is important. What made you feel this way?",
                "It's great to see you in such a positive state! Would you like to share what's bringing you joy?"
            ],
            'sadness': [
                "I hear you, and I want you to know that your feelings are completely valid. Loneliness can be really difficult to bear. What's been weighing on your heart lately?",
                "It's okay to feel sad and lonely. These feelings are real and important. Would you like to talk about what's making you feel this way?",
                "I'm here with you through this difficult time. Sometimes sharing our sadness can help lighten the burden. What's been on your mind?"
            ],
            'anger': [
                "I can sense your frustration. Anger is a natural emotion, and it's okay to feel this way. What's making you feel so frustrated?",
                "It sounds like you're dealing with something really frustrating. Would you like to talk about what's bothering you?",
                "I understand you're feeling angry. Sometimes it helps to identify what's really bothering us beneath the surface. What's going on?"
            ],
            'fear': [
                "I can hear the worry in your words. Anxiety can feel overwhelming, but you're not alone in this. What's making you feel anxious?",
                "It's completely normal to feel anxious sometimes. Let's work through this together. What specific concerns are you facing?",
                "I'm here to help you through this anxious moment. What's on your mind that's causing you worry?"
            ],
            'surprise': [
                "Wow, that sounds unexpected! How are you feeling about this surprise?",
                "That's quite a turn of events! What's your reaction to this unexpected situation?",
                "Surprises can be exciting or challenging. How are you processing this new development?"
            ],
            'disgust': [
                "I can sense your discomfort with this situation. What's making you feel this way?",
                "It sounds like something is really bothering you. Would you like to talk about what's causing this feeling?",
                "I understand you're feeling repulsed by something. Sometimes it helps to identify what's triggering this reaction."
            ],
            'neutral': [
                "I'm here to listen. How are you feeling today?",
                "Tell me what's on your mind. I'm here to support you.",
                "What would you like to talk about today?"
            ]
        }
    
    def load_qa_data(self):
        """Load Q&A data and prepare for similarity matching"""
        try:
            self.qa_data = pd.read_csv("data.csv")
            self.vectorizer = TfidfVectorizer()
            X = self.vectorizer.fit_transform(self.qa_data["question"])
            self.qa_vectors = X
        except Exception as e:
            print(f"Error loading Q&A data: {e}")
            self.qa_data = pd.DataFrame(columns=['question', 'answer'])
            self.vectorizer = TfidfVectorizer()
            self.qa_vectors = None
    
    def get_contextual_response(self, user_message, user_id=None, session_id=None):
        """Generate contextual response based on sentiment analysis and conversation history"""
        
        # Perform comprehensive sentiment analysis
        if self.sentiment_analyzer:
            analysis = self.sentiment_analyzer.comprehensive_analysis(user_message)
        else:
            # Fallback analysis without sentiment analyzer
            analysis = {
                'emotion': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'stress_level': 'low',
                'mental_health_concern': False,
                'recommendations': ['Consider talking to someone about your feelings']
            }
        
        # Get conversation context
        context = self._get_conversation_context(session_id) if session_id else []
        
        # Generate response based on analysis
        response = self._generate_emotionally_aware_response(
            user_message, analysis, context
        )
        
        # Store the conversation
        if session_id:
            self._store_message(session_id, 'user', user_message, analysis)
            self._store_message(session_id, 'bot', response)
        
        # Generate and store recommendations if needed
        if analysis['recommendations'] and user_id:
            self._store_recommendations(user_id, analysis['recommendations'], analysis)
        
        return {
            'response': response,
            'sentiment_analysis': analysis,
            'recommendations': analysis['recommendations']
        }
    
    def _get_conversation_context(self, session_id):
        """Get recent conversation context"""
        if not session_id:
            return []
        
        recent_messages = ChatMessage.query.filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.desc()).limit(5).all()
        
        return [msg.content for msg in reversed(recent_messages)]
    
    def _check_critical_concerns(self, user_message, analysis):
        """Check for critical mental health concerns that need immediate attention"""
        message_lower = user_message.lower()
        
        # Critical keywords that require immediate professional response
        critical_keywords = [
            'kill myself', 'suicide', 'end my life', 'not worth living',
            'want to die', 'better off dead', 'harm myself', 'hurt myself',
            'self harm', 'cut myself', 'overdose', 'take pills'
        ]
        
        # Check for critical concerns
        for keyword in critical_keywords:
            if keyword in message_lower:
                return (
                    "I'm deeply concerned about what you're saying. Your life has value and meaning, even when it doesn't feel that way. "
                    "Please reach out for immediate help:\n\n"
                    "🚨 **Crisis Resources:**\n"
                    "• National Suicide Prevention Lifeline: 988 (US)\n"
                    "• Crisis Text Line: Text HOME to 741741\n"
                    "• Emergency Services: 911\n\n"
                    "You don't have to face this alone. There are people who want to help you through this difficult time. "
                    "Please consider reaching out to a trusted friend, family member, or mental health professional right now."
                )
        
        # Check for severe loneliness and isolation
        loneliness_keywords = ['no friends', 'no one cares', 'completely alone', 'nobody loves me', 'no one understands']
        if any(keyword in message_lower for keyword in loneliness_keywords):
            return (
                "I hear how deeply lonely you're feeling, and I want you to know that your feelings are completely valid. "
                "Loneliness can be one of the most painful emotions to experience. While it might feel like no one cares, "
                "there are people who do care about you, even if you haven't met them yet.\n\n"
                "💙 **You're not alone in feeling alone:**\n"
                "• Many people experience deep loneliness\n"
                "• It's okay to feel this way\n"
                "• These feelings can change\n\n"
                "Would you like to talk more about what's making you feel so isolated? Sometimes sharing these feelings "
                "can help lighten the burden, even just a little."
            )
        
        # Check for severe depression indicators
        depression_keywords = ['hopeless', 'worthless', 'no point', 'give up', 'can\'t go on']
        if any(keyword in message_lower for keyword in depression_keywords):
            return (
                "I can hear the pain and hopelessness in your words, and I want you to know that these feelings, "
                "while overwhelming, are not permanent. Depression can make everything feel hopeless, but that's "
                "the depression talking, not the truth about your worth or your future.\n\n"
                "🌟 **Important reminders:**\n"
                "• Your feelings are valid and temporary\n"
                "• You have inherent worth, regardless of how you feel\n"
                "• Help is available and effective\n"
                "• You don't have to face this alone\n\n"
                "Would you like to talk about what's been weighing on you? Sometimes sharing these feelings "
                "can help you feel less alone in them."
            )
        
        return None
    
    def _generate_emotionally_aware_response(self, user_message, analysis, context):
        """Generate response based on emotional analysis"""
        emotion = analysis['emotion']
        sentiment_score = analysis['sentiment_score']
        stress_level = analysis['stress_level']
        mental_health_concern = analysis['mental_health_concern']
        
        # Check for critical mental health concerns first
        critical_response = self._check_critical_concerns(user_message, analysis)
        if critical_response:
            return critical_response
        
        # Check for exact Q&A matches
        qa_response = self._get_qa_response(user_message)
        if qa_response:
            return qa_response
        
        # Generate emotion-specific response
        if emotion in self.response_templates:
            base_response = random.choice(self.response_templates[emotion])
        else:
            base_response = random.choice(self.response_templates['neutral'])
        
        # Enhance response based on additional factors
        enhanced_response = self._enhance_response(
            base_response, analysis, context
        )
        
        return enhanced_response
    
    def _get_qa_response(self, user_message):
        """Get response from Q&A database using similarity matching"""
        if self.qa_vectors is None or len(self.qa_data) == 0:
            return None
        
        try:
            # Vectorize user message
            user_vector = self.vectorizer.transform([user_message])
            
            # Calculate similarity with all questions
            similarities = cosine_similarity(user_vector, self.qa_vectors).flatten()
            
            # Get best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            # Return response if similarity is high enough and contextually appropriate
            if best_similarity > 0.4:  # Higher threshold for better matching
                matched_question = self.qa_data.iloc[best_match_idx]['question']
                matched_answer = self.qa_data.iloc[best_match_idx]['answer']
                
                # Check if the match is contextually appropriate
                if self._is_contextually_appropriate(user_message, matched_question):
                    return matched_answer
            
        except Exception as e:
            print(f"Error in Q&A matching: {e}")
        
        return None
    
    def _is_contextually_appropriate(self, user_message, matched_question):
        """Check if the matched question is contextually appropriate for the user's message"""
        user_lower = user_message.lower()
        question_lower = matched_question.lower()
        
        # Check for critical mismatches
        critical_user_words = ['kill', 'suicide', 'die', 'lonely', 'no friends', 'hopeless']
        inappropriate_responses = ['sleep', 'teacher', 'study', 'exam']
        
        # If user mentions critical concerns, don't give generic responses
        if any(word in user_lower for word in critical_user_words):
            if any(word in question_lower for word in inappropriate_responses):
                return False
        
        return True
    
    def _enhance_response(self, base_response, analysis, context):
        """Enhance response based on analysis and context"""
        enhanced = base_response
        
        # Add stress acknowledgment
        if analysis['stress_level'] == 'high':
            enhanced += " I can sense you're under a lot of stress right now. Remember to take care of yourself."
        elif analysis['stress_level'] == 'medium':
            enhanced += " It sounds like you might be feeling some pressure. How can I help you manage this?"
        
        # Add mental health concern acknowledgment
        if analysis['mental_health_concern']:
            enhanced += " If you're struggling with your mental health, please consider reaching out to a professional. You don't have to face this alone."
        
        # Add follow-up questions based on emotion
        emotion = analysis['emotion']
        if emotion == 'sadness':
            enhanced += " Would you like to talk about what's making you feel this way?"
        elif emotion == 'anger':
            enhanced += " What's really bothering you beneath the surface?"
        elif emotion == 'fear':
            enhanced += " What specific concerns are you facing right now?"
        elif emotion == 'joy':
            enhanced += " What's contributing to this positive feeling?"
        
        return enhanced
    
    def _store_message(self, session_id, message_type, content, analysis=None):
        """Store message in database"""
        try:
            message = ChatMessage(
                session_id=session_id,
                message_type=message_type,
                content=content
            )
            
            # Add sentiment analysis for user messages
            if message_type == 'user' and analysis:
                message.sentiment_score = analysis['sentiment_score']
                message.emotion_category = analysis['emotion']
                message.confidence_score = analysis['confidence']
            
            db.session.add(message)
            db.session.commit()
            
        except Exception as e:
            print(f"Error storing message: {e}")
            db.session.rollback()
    
    def _store_recommendations(self, user_id, recommendations, analysis):
        """Store recommendations for the user"""
        try:
            for rec_text in recommendations:
                # Determine category based on analysis
                category = 'general'
                if analysis['mental_health_concern']:
                    category = 'mental_health'
                elif analysis['stress_level'] in ['high', 'medium']:
                    category = 'stress'
                elif analysis['emotion'] != 'neutral':
                    category = 'emotion'
                
                recommendation = Recommendation(
                    user_id=user_id,
                    recommendation_text=rec_text,
                    category=category
                )
                db.session.add(recommendation)
            
            db.session.commit()
            
        except Exception as e:
            print(f"Error storing recommendations: {e}")
            db.session.rollback()
    
    def create_chat_session(self, user_id, session_name=None):
        """Create a new chat session"""
        try:
            if not session_name:
                session_name = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            session = ChatSession(
                user_id=user_id,
                session_name=session_name
            )
            
            db.session.add(session)
            db.session.commit()
            
            return session.id
            
        except Exception as e:
            print(f"Error creating chat session: {e}")
            db.session.rollback()
            return None
    
    def get_chat_history(self, session_id, limit=50):
        """Get chat history for a session"""
        try:
            messages = ChatMessage.query.filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.timestamp.asc()).limit(limit).all()
            
            return [msg.to_dict() for msg in messages]
            
        except Exception as e:
            print(f"Error getting chat history: {e}")
            return []
    
    def get_user_sessions(self, user_id, limit=20):
        """Get user's chat sessions"""
        try:
            sessions = ChatSession.query.filter(
                ChatSession.user_id == user_id
            ).order_by(ChatSession.last_activity.desc()).limit(limit).all()
            
            return [session.to_dict() for session in sessions]
            
        except Exception as e:
            print(f"Error getting user sessions: {e}")
            return []
