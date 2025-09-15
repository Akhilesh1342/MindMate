"""
Data Migration Script for MindMate Enhanced
Migrates data from the old CSV-based system to the new database system
"""

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from app_enhanced import app, db
from models.database import User, MoodEntry, Recommendation
from models.sentiment_analyzer import AdvancedSentimentAnalyzer

def migrate_user_data():
    """Migrate user data from CSV to database"""
    print("🔄 Migrating user data...")
    
    csv_file = "user_data.csv"
    if not os.path.exists(csv_file):
        print("⚠️ No user_data.csv found, skipping migration")
        return
    
    try:
        # Read CSV data
        df = pd.read_csv(csv_file)
        print(f"📊 Found {len(df)} entries in CSV")
        
        # Create a default user for migration
        with app.app_context():
            # Check if default user exists
            default_user = User.query.filter_by(username="migrated_user").first()
            if not default_user:
                default_user = User(
                    username="migrated_user",
                    email="migrated@mindmate.com"
                )
                db.session.add(default_user)
                db.session.commit()
                print("✅ Created default user for migration")
            
            # Initialize sentiment analyzer
            analyzer = AdvancedSentimentAnalyzer()
            
            # Migrate mood entries
            migrated_count = 0
            for _, row in df.iterrows():
                try:
                    # Parse timestamp
                    timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    
                    # Perform sentiment analysis on the message
                    analysis = analyzer.comprehensive_analysis(row['message'])
                    
                    # Create mood entry
                    mood_entry = MoodEntry(
                        user_id=default_user.id,
                        timestamp=timestamp,
                        mood_label=row['mood'],
                        journal_text=row['message'],
                        sentiment_score=analysis['sentiment_score'],
                        emotion_category=analysis['emotion'],
                        confidence_score=analysis['confidence'],
                        stress_level=analysis['stress_level'],
                        mental_health_concern=analysis['mental_health_concern']
                    )
                    
                    # Store analysis details
                    mood_entry.set_analysis_details(analysis['analysis_details'])
                    
                    db.session.add(mood_entry)
                    
                    # Add recommendations if available
                    for rec_text in analysis['recommendations']:
                        category = 'general'
                        if analysis['mental_health_concern']:
                            category = 'mental_health'
                        elif analysis['stress_level'] in ['high', 'medium']:
                            category = 'stress'
                        elif analysis['emotion'] != 'neutral':
                            category = 'emotion'
                        
                        recommendation = Recommendation(
                            user_id=default_user.id,
                            mood_entry_id=mood_entry.id,
                            recommendation_text=rec_text,
                            category=category
                        )
                        db.session.add(recommendation)
                    
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error migrating entry {row.get('timestamp', 'unknown')}: {e}")
                    continue
            
            db.session.commit()
            print(f"✅ Successfully migrated {migrated_count} mood entries")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.session.rollback()

def backup_original_data():
    """Create backup of original data"""
    print("💾 Creating backup of original data...")
    
    backup_files = []
    
    # Backup CSV files
    csv_files = ["user_data.csv", "data.csv"]
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            backup_name = f"{csv_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(csv_file, backup_name)
            backup_files.append(backup_name)
            print(f"✅ Backed up {csv_file} as {backup_name}")
    
    return backup_files

def verify_migration():
    """Verify that migration was successful"""
    print("🔍 Verifying migration...")
    
    with app.app_context():
        # Check users
        user_count = User.query.count()
        print(f"👥 Users in database: {user_count}")
        
        # Check mood entries
        mood_count = MoodEntry.query.count()
        print(f"📊 Mood entries in database: {mood_count}")
        
        # Check recommendations
        rec_count = Recommendation.query.count()
        print(f"💡 Recommendations in database: {rec_count}")
        
        if mood_count > 0:
            print("✅ Migration verification successful")
            return True
        else:
            print("❌ Migration verification failed")
            return False

def create_sample_user():
    """Create a sample user for testing"""
    print("👤 Creating sample user...")
    
    with app.app_context():
        # Check if sample user exists
        sample_user = User.query.filter_by(username="demo_user").first()
        if not sample_user:
            sample_user = User(
                username="demo_user",
                email="demo@mindmate.com"
            )
            db.session.add(sample_user)
            db.session.commit()
            print("✅ Created demo user")
        else:
            print("✅ Demo user already exists")

def main():
    """Main migration function"""
    print("🔄 MindMate Data Migration")
    print("=" * 40)
    
    # Initialize app context
    with app.app_context():
        # Create tables
        db.create_all()
        print("✅ Database tables created")
    
    # Create sample user
    create_sample_user()
    
    # Migrate user data
    migrate_user_data()
    
    # Verify migration
    if verify_migration():
        print("\n🎉 Migration completed successfully!")
        print("📝 You can now use the enhanced MindMate application")
        print("🌐 Run 'python run_enhanced.py' to start the application")
    else:
        print("\n❌ Migration failed. Please check the errors above.")

if __name__ == "__main__":
    main()
