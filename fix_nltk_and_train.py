#!/usr/bin/env python3
"""
Fix NLTK Data and Train Models
Simple script to download NLTK data and train models
"""

import sys
import os
import subprocess

def fix_nltk_data():
    """Download NLTK data properly"""
    print("🔧 Fixing NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        print("📚 Downloading NLTK data...")
        nltk.download('punkt', quiet=False)
        nltk.download('stopwords', quiet=False)
        nltk.download('wordnet', quiet=False)
        nltk.download('vader_lexicon', quiet=False)
        
        print("✅ NLTK data downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ NLTK download failed: {e}")
        return False

def train_models_simple():
    """Train models with simple approach"""
    print("🤖 Training models...")
    
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from models.sentiment_analyzer import AdvancedSentimentAnalyzer
        import pandas as pd
        import numpy as np
        
        # Load data
        qa_data = pd.read_csv("data.csv")
        print(f"📊 Loaded {len(qa_data)} Q&A pairs")
        
        # Create emotion labels
        emotion_mapping = {
            'stressed': 'stress',
            'anxious': 'anxiety', 
            'lonely': 'sadness',
            'sad': 'sadness',
            'depressed': 'sadness',
            'worried': 'anxiety',
            'overwhelmed': 'stress',
            'frustrated': 'anger',
            'nervous': 'anxiety',
            'guilty': 'sadness',
            'hopeless': 'sadness',
            'happy': 'joy',
            'excited': 'joy',
            'grateful': 'joy',
            'confident': 'joy',
            'optimistic': 'joy',
            'peaceful': 'neutral',
            'calm': 'neutral',
            'content': 'joy',
            'motivated': 'joy',
            'proud': 'joy',
            'energized': 'joy'
        }
        
        labels = []
        for question in qa_data['question']:
            question_lower = question.lower()
            emotion = 'neutral'  # default
            
            for keyword, emotion_label in emotion_mapping.items():
                if keyword in question_lower:
                    emotion = emotion_label
                    break
            
            labels.append(emotion)
        
        # Prepare training data
        X_train = qa_data['question'].tolist()
        y_train = labels
        
        print(f"🎯 Training data: {len(X_train)} samples")
        print(f"📈 Emotion distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        # Initialize and train analyzer
        analyzer = AdvancedSentimentAnalyzer()
        analyzer.train_models(X_train, y_train)
        
        # Save models
        analyzer.save_models()
        
        print("✅ Models trained and saved successfully!")
        
        # Test the models
        test_examples = [
            "I'm feeling really anxious about my presentation tomorrow",
            "I'm so happy about my promotion!",
            "I feel so lonely and isolated",
            "I'm frustrated with my computer not working",
            "I feel peaceful after my meditation"
        ]
        
        print("\n🧪 Testing trained models...")
        for example in test_examples:
            analysis = analyzer.comprehensive_analysis(example)
            print(f"\n📝 Text: '{example}'")
            print(f"   Emotion: {analysis['emotion']}")
            print(f"   Sentiment Score: {analysis['sentiment_score']:.3f}")
            print(f"   Confidence: {analysis['confidence']:.3f}")
            print(f"   Stress Level: {analysis['stress_level']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🧠 MindMate - Fix NLTK and Train Models")
    print("=" * 50)
    
    # Fix NLTK data
    if not fix_nltk_data():
        print("⚠️ Continuing without NLTK data...")
    
    # Train models
    if train_models_simple():
        print("\n🎉 All done! Models are ready to use.")
        print("🚀 You can now run: python start_mindmate.py")
    else:
        print("\n❌ Training failed, but the app will still work with basic functionality.")

if __name__ == "__main__":
    main()
