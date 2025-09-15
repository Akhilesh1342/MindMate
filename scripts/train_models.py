"""
Model Training Script for MindMate
Trains advanced sentiment analysis models using the existing data
"""

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.sentiment_analyzer import AdvancedSentimentAnalyzer
from models.database import db, MoodEntry
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Load and prepare training data"""
    print("Loading training data...")
    
    # Load Q&A data
    try:
        qa_data = pd.read_csv("data.csv")
        print(f"Loaded {len(qa_data)} Q&A pairs")
    except FileNotFoundError:
        print("Q&A data not found, creating sample data...")
        qa_data = create_sample_qa_data()
    
    # Load user mood data if available
    try:
        user_data = pd.read_csv("user_data.csv")
        print(f"Loaded {len(user_data)} user mood entries")
    except FileNotFoundError:
        print("User mood data not found")
        user_data = pd.DataFrame()
    
    return qa_data, user_data

def create_sample_qa_data():
    """Create sample Q&A data for training"""
    sample_data = {
        'question': [
            "I feel stressed about work",
            "I'm anxious about exams",
            "I feel lonely and sad",
            "I'm having trouble sleeping",
            "I lack motivation today",
            "I feel depressed",
            "I'm worried about my future",
            "I feel overwhelmed with everything",
            "I'm frustrated with my situation",
            "I need help with time management",
            "I feel anxious in social situations",
            "I'm feeling unmotivated",
            "I'm sad about recent events",
            "I feel stressed about deadlines",
            "I'm nervous about speaking in class",
            "I feel guilty about my mistakes",
            "I'm anxious about the future",
            "I feel lonely at night",
            "I'm frustrated with my studies",
            "I feel hopeless about everything",
            "I'm happy about my progress",
            "I feel excited about new opportunities",
            "I'm grateful for my friends",
            "I feel confident about my abilities",
            "I'm optimistic about tomorrow",
            "I feel peaceful and calm",
            "I'm content with my life",
            "I feel motivated to achieve my goals",
            "I'm proud of my accomplishments",
            "I feel energized and ready to go"
        ],
        'answer': [
            "I hear you. Stress can feel heavy sometimes. What's on your mind?",
            "It's normal to feel anxious. Take a few deep breaths, and maybe review one topic at a time.",
            "You're not alone. Sometimes talking to a friend or family member helps a lot.",
            "Sleep is important. Try going to bed at the same time and avoiding screens before sleeping.",
            "Small steps can make a big difference. Celebrate every little win!",
            "It's okay to feel down. Sharing your feelings with someone you trust can help.",
            "Focus on what you can control and take breaks to relax your mind.",
            "Break your tasks into smaller chunks and tackle them one at a time.",
            "It's normal to feel frustrated. A short walk or some breathing exercises can help clear your mind.",
            "Prioritize your tasks and create a schedule. Start with the most important tasks first.",
            "Many people feel that way. Try taking deep breaths and starting with small conversations.",
            "It's okay to have off days. Do something small that makes you feel accomplished.",
            "I'm here for you. Sometimes sharing what's making you sad helps lighten the burden.",
            "Try making a simple to-do list. Tackling tasks one by one makes them less scary.",
            "It's normal to feel nervous. Remember, everyone is focused on their own learning too.",
            "Mistakes are part of learning. Be kind to yourself and focus on what you can improve.",
            "It's okay to feel uncertain. Take things step by step and focus on the present.",
            "Nighttime can be tough. Maybe read a book, listen to calming music, or message a friend.",
            "Take a short break, stretch, or do something you enjoy before getting back to it.",
            "I hear you. Sometimes talking to someone you trust or a counselor can help you feel better.",
            "That's wonderful! Celebrating the good moments is important. What made you feel this way?",
            "It's great to see you excited! What opportunities are you looking forward to?",
            "Gratitude is powerful. What specifically are you grateful for today?",
            "Confidence is beautiful. What's helping you feel more confident lately?",
            "Optimism is a great mindset. What are you looking forward to tomorrow?",
            "Peace and calm are precious. What's helping you feel centered right now?",
            "Contentment is a wonderful feeling. What aspects of your life are bringing you satisfaction?",
            "Motivation is energizing! What goals are you most excited to work toward?",
            "You should be proud! What accomplishments are you celebrating today?",
            "Energy and readiness are fantastic! What's got you feeling so energized?"
        ]
    }
    
    return pd.DataFrame(sample_data)

def create_emotion_labels(qa_data):
    """Create emotion labels for Q&A data"""
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
    
    return labels

def train_sentiment_models():
    """Train the sentiment analysis models"""
    print("Starting model training...")
    
    # Load data
    qa_data, user_data = load_training_data()
    
    # Create emotion labels
    emotion_labels = create_emotion_labels(qa_data)
    
    # Prepare training data
    X_train = qa_data['question'].tolist()
    y_train = emotion_labels
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Emotion distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Initialize analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    # Train models
    print("Training models...")
    analyzer.train_models(X_train, y_train)
    
    # Evaluate models
    print("Evaluating models...")
    evaluate_models(analyzer, X_train, y_train)
    
    # Save models
    print("Saving models...")
    analyzer.save_models()
    
    print("Model training completed!")
    return analyzer

def evaluate_models(analyzer, X_test, y_test):
    """Evaluate the trained models"""
    print("\nModel Evaluation:")
    print("=" * 50)
    
    # Test on a subset of data
    test_size = min(100, len(X_test))
    X_test_sample = X_test[:test_size]
    y_test_sample = y_test[:test_size]
    
    predictions = []
    confidences = []
    
    for text in X_test_sample:
        pred, conf = analyzer.predict_sentiment_ensemble(text)
        predictions.append(pred)
        confidences.append(conf)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_sample, predictions)
    print(f"Overall Accuracy: {accuracy:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_sample, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_sample, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y_test_sample)), 
                yticklabels=sorted(set(y_test_sample)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('static/plots', exist_ok=True)
    plt.savefig('static/plots/confusion_matrix.png')
    plt.close()
    
    # Confidence analysis
    avg_confidence = np.mean(confidences)
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Per-emotion accuracy
    emotion_accuracies = {}
    for emotion in set(y_test_sample):
        emotion_indices = [i for i, e in enumerate(y_test_sample) if e == emotion]
        if emotion_indices:
            emotion_preds = [predictions[i] for i in emotion_indices]
            emotion_acc = sum(1 for p in emotion_preds if p == emotion) / len(emotion_preds)
            emotion_accuracies[emotion] = emotion_acc
            print(f"{emotion.capitalize()} Accuracy: {emotion_acc:.3f}")

def create_sample_training_data():
    """Create additional sample training data"""
    print("Creating additional sample training data...")
    
    # Create more diverse training examples
    additional_data = {
        'question': [
            "I'm feeling really down today",
            "Work is stressing me out completely",
            "I can't stop worrying about everything",
            "I feel so angry about what happened",
            "I'm scared about the future",
            "I feel disgusted by this situation",
            "I'm surprised by this turn of events",
            "I feel so happy and grateful",
            "I'm excited about my new job",
            "I feel confident and strong",
            "I'm optimistic about everything",
            "I feel peaceful and serene",
            "I'm content with my life right now",
            "I feel motivated and energized",
            "I'm proud of what I've accomplished",
            "I feel neutral about everything",
            "I'm not sure how I feel",
            "I feel mixed emotions",
            "I'm confused about my feelings",
            "I feel numb and empty"
        ],
        'answer': [
            "I'm here with you through this difficult time. What's making you feel down?",
            "Work stress can be overwhelming. Let's talk about what's bothering you most.",
            "Constant worry is exhausting. Let's work on some calming techniques together.",
            "Anger is a natural emotion. What's really making you feel this way?",
            "Fear about the future is understandable. Let's focus on what you can control.",
            "I understand you're feeling repulsed. What's causing this reaction?",
            "Surprises can be challenging. How are you processing this unexpected change?",
            "It's wonderful to hear you're feeling positive! What's bringing you joy?",
            "New opportunities are exciting! Tell me more about what you're looking forward to.",
            "Confidence is beautiful to see. What's helping you feel so strong?",
            "Optimism is a great mindset. What are you feeling hopeful about?",
            "Peace and serenity are precious. What's helping you feel so centered?",
            "Contentment is a wonderful feeling. What aspects of your life are satisfying?",
            "Motivation and energy are fantastic! What's driving you forward?",
            "You should be proud! What accomplishments are you celebrating?",
            "Sometimes neutral feelings are okay. How are you doing overall?",
            "It's okay to be unsure. Sometimes our feelings are complex.",
            "Mixed emotions are normal. Would you like to talk through them?",
            "Confusion about feelings is common. Let's explore what you're experiencing.",
            "Feeling numb can be concerning. Let's talk about what might be causing this."
        ]
    }
    
    # Save additional data
    additional_df = pd.DataFrame(additional_data)
    additional_df.to_csv('additional_training_data.csv', index=False)
    print(f"Created {len(additional_data['question'])} additional training examples")

def main():
    """Main training function"""
    print("MindMate Model Training Script")
    print("=" * 40)
    
    # Create sample data if needed
    create_sample_training_data()
    
    # Train models
    analyzer = train_sentiment_models()
    
    # Test the trained models
    print("\nTesting trained models...")
    test_examples = [
        "I'm feeling really anxious about my presentation tomorrow",
        "I'm so happy about my promotion!",
        "I feel so lonely and isolated",
        "I'm frustrated with my computer not working",
        "I feel peaceful after my meditation"
    ]
    
    for example in test_examples:
        analysis = analyzer.comprehensive_analysis(example)
        print(f"\nText: '{example}'")
        print(f"Emotion: {analysis['emotion']}")
        print(f"Sentiment Score: {analysis['sentiment_score']:.3f}")
        print(f"Confidence: {analysis['confidence']:.3f}")
        print(f"Stress Level: {analysis['stress_level']}")
        print(f"Recommendations: {analysis['recommendations'][:2]}")
    
    print("\nTraining completed successfully!")
    print("Models saved to: models/sentiment_models.pkl")

if __name__ == "__main__":
    main()
