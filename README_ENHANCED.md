# MindMate 2.0 - Advanced Mental Health Companion

A sophisticated mental health tracking and sentiment analysis application with advanced ML models, comprehensive analytics, and intelligent chat capabilities.

## 🚀 Features

### Advanced Sentiment Analysis
- **Multi-Model Ensemble**: Combines Logistic Regression, Random Forest, and SVM for robust predictions
- **Comprehensive Emotional Analysis**: Detects 7 core emotions (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Stress Level Detection**: Automatically identifies low, medium, and high stress levels
- **Mental Health Monitoring**: Flags potential mental health concerns
- **Confidence Scoring**: Provides confidence levels for all predictions

### Intelligent Chat System
- **Context-Aware Responses**: Understands conversation history and emotional context
- **Emotionally Intelligent**: Responds appropriately to different emotional states
- **Personalized Recommendations**: Generates tailored suggestions based on analysis
- **Session Management**: Tracks conversations across multiple sessions

### Comprehensive Analytics
- **Mood Trends**: Daily, weekly, and monthly mood pattern analysis
- **Emotional Patterns**: Identifies dominant emotions and transitions
- **Stress Analysis**: Tracks stress patterns and identifies triggers
- **Wellbeing Metrics**: Calculates overall mental health scores
- **Predictive Insights**: Forecasts mood trends and identifies risk factors
- **Behavioral Insights**: Analyzes journaling patterns and coping effectiveness

### Advanced Reporting
- **Interactive Charts**: Beautiful visualizations using Plotly and Matplotlib
- **PDF Reports**: Comprehensive mood reports with insights and recommendations
- **Data Export**: Export mood data for external analysis
- **Trend Analysis**: Historical pattern recognition

## 🏗️ Architecture

### Backend Structure
```
mindmate/
├── app_enhanced.py              # Main Flask application
├── models/
│   ├── sentiment_analyzer.py   # Advanced ML sentiment analysis
│   └── database.py             # Database models and utilities
├── services/
│   ├── chat_service.py         # Intelligent chat system
│   └── analytics_service.py    # Comprehensive analytics
├── utils/
│   └── helpers.py              # Utility functions
├── scripts/
│   └── train_models.py         # Model training script
├── config.py                   # Configuration settings
└── requirements.txt            # Dependencies
```

### Database Schema
- **Users**: User accounts and authentication
- **MoodEntries**: Mood tracking with advanced sentiment analysis
- **ChatSessions**: Conversation management
- **ChatMessages**: Individual chat messages with sentiment
- **Recommendations**: Personalized suggestions
- **MoodInsights**: Aggregated daily insights

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda
- SQLite (default) or PostgreSQL (production)

### Installation Steps

1. **Clone and navigate to the project**
```bash
cd mindmate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize the database**
```bash
python -c "from app_enhanced import app, db; app.app_context().push(); db.create_all()"
```

4. **Train the ML models**
```bash
python scripts/train_models.py
```

5. **Run the application**
```bash
python app_enhanced.py
```

The application will be available at `http://localhost:5000`

## 📊 ML Models & Sentiment Analysis

### Model Architecture
- **Text Preprocessing**: Advanced tokenization, lemmatization, and stopword removal
- **Feature Extraction**: TF-IDF and Count vectorization with n-grams
- **Ensemble Learning**: Multiple models with majority voting
- **Confidence Scoring**: Probabilistic confidence measures

### Supported Emotions
- **Joy**: Happy, excited, grateful, optimistic
- **Sadness**: Sad, depressed, lonely, hopeless
- **Anger**: Angry, frustrated, irritated, enraged
- **Fear**: Anxious, worried, scared, nervous
- **Surprise**: Surprised, shocked, amazed, startled
- **Disgust**: Disgusted, revolted, repulsed
- **Neutral**: Calm, peaceful, balanced, content

### Advanced Features
- **Stress Detection**: Identifies stress levels and triggers
- **Mental Health Monitoring**: Flags concerning patterns
- **Sentiment Scoring**: Continuous sentiment scores (-1 to 1)
- **Emotional Intensity**: Measures emotional strength
- **Pattern Recognition**: Identifies mood cycles and trends

## 🔌 API Endpoints

### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - User login
- `POST /api/logout` - User logout

### Mood Tracking
- `POST /api/log-mood` - Log mood with advanced analysis
- `GET /api/mood-data` - Get comprehensive mood data
- `GET /api/insights` - Get detailed analytics and insights

### Chat System
- `POST /api/chat` - Intelligent chat with sentiment analysis
- `GET /api/chat-sessions` - Get user's chat sessions
- `GET /api/chat-history/<session_id>` - Get chat history

### Analytics & Reports
- `GET /api/generate-report` - Generate comprehensive PDF report
- `GET /api/analytics` - Get advanced analytics data

## 📈 Analytics Features

### Mood Analytics
- **Trend Analysis**: Daily, weekly, monthly mood trends
- **Volatility Scoring**: Measures mood stability
- **Pattern Recognition**: Identifies recurring patterns
- **Seasonal Analysis**: Detects seasonal mood variations

### Stress Analytics
- **Stress Level Tracking**: Monitors stress over time
- **Trigger Identification**: Identifies stress triggers
- **Recovery Analysis**: Measures stress recovery patterns
- **Management Effectiveness**: Evaluates coping strategies

### Wellbeing Metrics
- **Overall Wellbeing Score**: Comprehensive mental health score
- **Emotional Wellbeing**: Emotional health component
- **Stress Wellbeing**: Stress management component
- **Mental Health Wellbeing**: Mental health concern tracking

### Predictive Insights
- **Mood Forecasting**: Predicts future mood trends
- **Risk Factor Identification**: Identifies potential risk factors
- **Intervention Timing**: Suggests optimal intervention times
- **Pattern Prediction**: Forecasts mood cycles

## 🎯 Key Improvements Over Original

### Enhanced ML Models
- **Multi-Model Ensemble**: More accurate predictions
- **Advanced Preprocessing**: Better text understanding
- **Confidence Scoring**: Reliability measures
- **Emotion Categories**: 7 distinct emotional states

### Sophisticated Analytics
- **Comprehensive Insights**: Deep pattern analysis
- **Predictive Capabilities**: Future trend prediction
- **Behavioral Analysis**: User behavior insights
- **Risk Assessment**: Mental health risk identification

### Robust Backend
- **Database Integration**: Proper data persistence
- **Session Management**: User authentication
- **API Structure**: RESTful API design
- **Error Handling**: Comprehensive error management

### Advanced Features
- **Intelligent Chat**: Context-aware conversations
- **Personalized Recommendations**: Tailored suggestions
- **Comprehensive Reporting**: Detailed PDF reports
- **Data Visualization**: Interactive charts

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=sqlite:///mindmate.db

# Security
SECRET_KEY=your-secret-key

# Email (optional)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email
MAIL_PASSWORD=your-password

# External APIs (optional)
OPENAI_API_KEY=your-openai-key
```

### Feature Flags
- `ENABLE_ADVANCED_ANALYTICS`: Enable comprehensive analytics
- `ENABLE_PREDICTIVE_INSIGHTS`: Enable predictive features
- `ENABLE_EMAIL_NOTIFICATIONS`: Enable email notifications
- `ENABLE_EXTERNAL_APIS`: Enable external API integrations

## 📱 Usage Examples

### Logging Mood with Analysis
```python
# POST /api/log-mood
{
    "mood": "😊",
    "journal": "I'm feeling great today! Got a promotion at work."
}

# Response includes comprehensive analysis
{
    "emotion": "joy",
    "sentiment_score": 0.8,
    "confidence": 0.92,
    "stress_level": "low",
    "recommendations": [
        "Celebrate this positive moment",
        "Share your joy with others"
    ]
}
```

### Intelligent Chat
```python
# POST /api/chat
{
    "message": "I'm feeling really anxious about my presentation tomorrow"
}

# Response with emotional understanding
{
    "reply": "I can hear the worry in your words. Anxiety can feel overwhelming. What's making you feel anxious?",
    "sentiment_analysis": {
        "emotion": "fear",
        "stress_level": "medium",
        "recommendations": ["Practice grounding techniques", "Focus on what you can control"]
    }
}
```

## 🚀 Deployment

### Production Setup
1. **Use PostgreSQL**: Replace SQLite with PostgreSQL
2. **Environment Variables**: Set production environment variables
3. **Security**: Use strong secret keys and HTTPS
4. **Monitoring**: Implement logging and monitoring
5. **Backup**: Set up regular database backups

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app_enhanced.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API endpoints

## 🔮 Future Enhancements

- **Mobile App**: React Native mobile application
- **AI Therapist**: Advanced conversational AI
- **Group Support**: Community features
- **Integration**: Calendar and health app integration
- **Advanced Analytics**: Machine learning insights
- **Multilingual Support**: Multiple language support

---

**MindMate 2.0** - Your intelligent mental health companion with advanced sentiment analysis and comprehensive emotional understanding.
