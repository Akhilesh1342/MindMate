#!/usr/bin/env python3
"""
Simple MindMate Enhanced Startup Script
Handles all setup and starts the application
"""

import sys
import os
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the Python environment"""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print(f"✅ Python path configured: {current_dir}")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

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
        return True
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False

def train_models():
    """Train the ML models"""
    print("\n🤖 Training ML models...")
    try:
        # Change to the scripts directory and run training
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "train_models.py")
        subprocess.check_call([sys.executable, script_path])
        print("✅ Models trained successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train models: {e}")
        print("⚠️ Continuing without trained models...")
        return False

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
        return False
    
    return True

def main():
    """Main startup function"""
    print("🧠 MindMate Enhanced - Advanced Mental Health Companion")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Cannot continue without dependencies")
        return
    
    # Create directories
    create_directories()
    
    # Initialize database
    if not initialize_database():
        print("❌ Cannot continue without database")
        return
    
    # Train models (optional)
    train_models()
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()
