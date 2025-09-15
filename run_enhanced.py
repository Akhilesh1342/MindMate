#!/usr/bin/env python3
"""
Enhanced MindMate Startup Script
Initializes the application with proper setup and model training
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = [
        "models",
        "static/charts",
        "static/reports",
        "static/plots",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def initialize_database():
    """Initialize the database"""
    print("\n🗄️ Initializing database...")
    try:
        from app_enhanced import app, db
        with app.app_context():
            db.create_all()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

def train_models():
    """Train the ML models"""
    print("\n🤖 Training ML models...")
    try:
        subprocess.check_call([sys.executable, "scripts/train_models.py"])
        print("✅ Models trained successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train models: {e}")
        print("⚠️ Continuing without trained models...")

def check_models():
    """Check if models exist"""
    model_path = "models/sentiment_models.pkl"
    if os.path.exists(model_path):
        print("✅ Pre-trained models found")
        return True
    else:
        print("⚠️ No pre-trained models found")
        return False

def create_sample_data():
    """Create sample data if needed"""
    print("\n📊 Creating sample data...")
    
    # Check if data.csv exists
    if not os.path.exists("data.csv"):
        print("Creating sample Q&A data...")
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
        
        import pandas as pd
        df = pd.DataFrame(sample_data)
        df.to_csv("data.csv", index=False)
        print("✅ Sample Q&A data created")
    else:
        print("✅ Q&A data already exists")

def run_application():
    """Run the enhanced application"""
    print("\n🚀 Starting MindMate Enhanced...")
    print("=" * 50)
    print("🌐 Application will be available at: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000/dashboard")
    print("💬 Chat: http://localhost:5000/home")
    print("📈 Mood Tracker: http://localhost:5000/mood-tracker-page")
    print("=" * 50)
    
    try:
        from app_enhanced import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Shutting down MindMate Enhanced...")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🧠 MindMate Enhanced - Advanced Mental Health Companion")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Create sample data
    create_sample_data()
    
    # Initialize database
    initialize_database()
    
    # Check for existing models
    if not check_models():
        # Train models if they don't exist
        train_models()
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()
