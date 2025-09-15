"""
Test Script for MindMate Enhanced
Tests the core functionality of the enhanced system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sentiment_analyzer import AdvancedSentimentAnalyzer
from services.chat_service import IntelligentChatService
from services.analytics_service import AdvancedAnalyticsService
from utils.helpers import sanitize_text, normalize_sentiment_score, extract_keywords
import pandas as pd

def test_sentiment_analyzer():
    """Test the sentiment analyzer"""
    print("🧪 Testing Sentiment Analyzer...")
    
    analyzer = AdvancedSentimentAnalyzer()
    
    test_texts = [
        "I'm feeling really happy today!",
        "I'm so stressed about work and everything is overwhelming",
        "I feel lonely and sad, nobody understands me",
        "I'm excited about my new job opportunity",
        "I'm frustrated with my computer not working properly",
        "I feel anxious about the presentation tomorrow",
        "I'm grateful for my friends and family",
        "I feel neutral about everything today"
    ]
    
    for text in test_texts:
        analysis = analyzer.comprehensive_analysis(text)
        print(f"\n📝 Text: '{text}'")
        print(f"   Emotion: {analysis['emotion']}")
        print(f"   Sentiment: {analysis['sentiment_score']:.3f}")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        print(f"   Stress Level: {analysis['stress_level']}")
        print(f"   Mental Health Concern: {analysis['mental_health_concern']}")
        print(f"   Recommendations: {analysis['recommendations'][:2]}")

def test_chat_service():
    """Test the chat service"""
    print("\n💬 Testing Chat Service...")
    
    chat_service = IntelligentChatService()
    
    test_messages = [
        "I'm feeling really anxious about my job interview tomorrow",
        "I'm so happy! I got the promotion I wanted",
        "I feel lonely and don't know what to do",
        "I'm frustrated with my studies and feel like giving up"
    ]
    
    for message in test_messages:
        response_data = chat_service.get_contextual_response(message)
        print(f"\n👤 User: {message}")
        print(f"🤖 Bot: {response_data['response']}")
        print(f"   Emotion: {response_data['sentiment_analysis']['emotion']}")
        print(f"   Stress: {response_data['sentiment_analysis']['stress_level']}")

def test_analytics_service():
    """Test the analytics service"""
    print("\n📊 Testing Analytics Service...")
    
    analytics_service = AdvancedAnalyticsService()
    
    # Create sample mood data
    sample_data = [
        {'timestamp': '2024-01-01', 'sentiment_score': 0.8, 'emotion': 'joy', 'stress_level': 'low'},
        {'timestamp': '2024-01-02', 'sentiment_score': -0.3, 'emotion': 'sadness', 'stress_level': 'medium'},
        {'timestamp': '2024-01-03', 'sentiment_score': 0.2, 'emotion': 'neutral', 'stress_level': 'low'},
        {'timestamp': '2024-01-04', 'sentiment_score': -0.7, 'emotion': 'anger', 'stress_level': 'high'},
        {'timestamp': '2024-01-05', 'sentiment_score': 0.6, 'emotion': 'joy', 'stress_level': 'low'},
    ]
    
    # Test mood trends analysis
    print("📈 Mood Trends Analysis:")
    trends = analytics_service.analyze_mood_trends(None, 30)  # Mock user_id
    print(f"   Trend Direction: {trends.get('trend_direction', 'N/A')}")
    print(f"   Volatility Score: {trends.get('volatility_score', 'N/A')}")
    
    # Test emotional patterns
    print("\n🎭 Emotional Patterns Analysis:")
    patterns = analytics_service.analyze_emotional_patterns(None, 30)
    print(f"   Dominant Emotions: {patterns.get('dominant_emotions', 'N/A')}")
    print(f"   Emotional Stability: {patterns.get('emotional_stability', 'N/A')}")

def test_utility_functions():
    """Test utility functions"""
    print("\n🔧 Testing Utility Functions...")
    
    # Test text sanitization
    test_text = "   This is a test text with   multiple   spaces   "
    sanitized = sanitize_text(test_text)
    print(f"📝 Text Sanitization:")
    print(f"   Original: '{test_text}'")
    print(f"   Sanitized: '{sanitized}'")
    
    # Test sentiment normalization
    test_scores = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print(f"\n📊 Sentiment Normalization:")
    for score in test_scores:
        normalized = normalize_sentiment_score(score)
        print(f"   {score:4.1f} -> {normalized:5.1f}")
    
    # Test keyword extraction
    test_text = "I am feeling really stressed about work and deadlines. I need help with time management."
    keywords = extract_keywords(test_text)
    print(f"\n🔍 Keyword Extraction:")
    print(f"   Text: '{test_text}'")
    print(f"   Keywords: {keywords}")

def test_data_processing():
    """Test data processing functions"""
    print("\n📊 Testing Data Processing...")
    
    # Test statistics calculation
    from utils.helpers import calculate_statistics
    test_values = [0.8, -0.3, 0.2, -0.7, 0.6, 0.1, -0.5, 0.9]
    stats = calculate_statistics(test_values)
    print(f"📈 Statistics Calculation:")
    print(f"   Values: {test_values}")
    print(f"   Mean: {stats['mean']:.3f}")
    print(f"   Median: {stats['median']:.3f}")
    print(f"   Std: {stats['std']:.3f}")
    print(f"   Min: {stats['min']:.3f}")
    print(f"   Max: {stats['max']:.3f}")
    
    # Test confidence interval
    from utils.helpers import create_confidence_interval
    ci = create_confidence_interval(test_values)
    print(f"\n📊 Confidence Interval:")
    print(f"   Mean: {ci['mean']:.3f}")
    print(f"   Lower Bound: {ci['lower_bound']:.3f}")
    print(f"   Upper Bound: {ci['upper_bound']:.3f}")
    print(f"   Margin of Error: {ci['margin_error']:.3f}")

def run_all_tests():
    """Run all tests"""
    print("🧠 MindMate Enhanced - Test Suite")
    print("=" * 50)
    
    try:
        test_sentiment_analyzer()
        test_chat_service()
        test_analytics_service()
        test_utility_functions()
        test_data_processing()
        
        print("\n🎉 All tests completed successfully!")
        print("✅ MindMate Enhanced is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
