#!/usr/bin/env python3
"""
Simple NLTK Fix Script
Downloads the specific NLTK data needed
"""

import nltk
import sys

def download_nltk_data():
    """Download all required NLTK data"""
    print("🔧 Downloading NLTK data...")
    
    try:
        # Download the specific data that's missing
        print("📚 Downloading punkt_tab...")
        nltk.download('punkt_tab', quiet=False)
        
        print("📚 Downloading punkt...")
        nltk.download('punkt', quiet=False)
        
        print("📚 Downloading stopwords...")
        nltk.download('stopwords', quiet=False)
        
        print("📚 Downloading wordnet...")
        nltk.download('wordnet', quiet=False)
        
        print("📚 Downloading vader_lexicon...")
        nltk.download('vader_lexicon', quiet=False)
        
        print("✅ All NLTK data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def test_nltk():
    """Test if NLTK is working"""
    print("🧪 Testing NLTK...")
    
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        # Test tokenization
        text = "Hello world, this is a test!"
        tokens = word_tokenize(text)
        print(f"✅ Tokenization works: {tokens}")
        
        # Test stopwords
        stop_words = set(stopwords.words('english'))
        print(f"✅ Stopwords loaded: {len(stop_words)} words")
        
        # Test lemmatizer
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize("running")
        print(f"✅ Lemmatization works: running -> {lemma}")
        
        # Test VADER
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("I love this!")
        print(f"✅ VADER sentiment works: {scores}")
        
        print("🎉 All NLTK components working!")
        return True
        
    except Exception as e:
        print(f"❌ NLTK test failed: {e}")
        return False

def main():
    """Main function"""
    print("🧠 MindMate - NLTK Data Fix")
    print("=" * 40)
    
    # Download NLTK data
    if download_nltk_data():
        # Test NLTK
        if test_nltk():
            print("\n🎉 NLTK is ready! You can now train models.")
            print("🚀 Run: python fix_nltk_and_train.py")
        else:
            print("\n⚠️ NLTK downloaded but not working properly.")
    else:
        print("\n❌ Failed to download NLTK data.")
        print("💡 You can still use the app without trained models.")

if __name__ == "__main__":
    main()
