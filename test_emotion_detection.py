#!/usr/bin/env python3
"""
Test Emotion Detection
Tests the improved emotion detection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sentiment_analyzer_simple import SimpleSentimentAnalyzer

def test_emotion_detection():
    """Test emotion detection with various inputs"""
    print("🧪 Testing Improved Emotion Detection")
    print("=" * 50)
    
    analyzer = SimpleSentimentAnalyzer()
    
    test_cases = [
        "I want to kill myself",
        "I am lonely",
        "I dont have any friends", 
        "I feel hopeless",
        "I'm so happy today!",
        "I'm really angry about this",
        "I feel anxious about the future",
        "I'm excited about my new job",
        "I feel neutral about everything",
        "I'm extremely frustrated",
        "I feel worthless",
        "I'm grateful for my friends"
    ]
    
    for text in test_cases:
        analysis = analyzer.comprehensive_analysis(text)
        print(f"\n📝 Text: '{text}'")
        print(f"   Emotion: {analysis['emotion']}")
        print(f"   Sentiment Score: {analysis['sentiment_score']:.3f}")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        print(f"   Stress Level: {analysis['stress_level']}")
        print(f"   Mental Health Concern: {analysis['mental_health_concern']}")
        print(f"   Recommendations: {analysis['recommendations'][:2]}")

def main():
    """Main test function"""
    test_emotion_detection()
    print("\n🎉 Emotion detection test completed!")

if __name__ == "__main__":
    main()
