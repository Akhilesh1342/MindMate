"""
Advanced Analytics Service for MindMate
Provides comprehensive mood analytics, pattern recognition, and insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_
from models.database import db, MoodEntry, User, MoodInsight
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.utils
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyticsService:
    """
    Advanced analytics service for mood tracking and mental health insights
    """
    
    def __init__(self):
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def get_comprehensive_analytics(self, user_id, days=30):
        """Get comprehensive analytics for a user"""
        try:
            analytics = {
                'mood_trends': self.analyze_mood_trends(user_id, days),
                'emotional_patterns': self.analyze_emotional_patterns(user_id, days),
                'stress_analysis': self.analyze_stress_patterns(user_id, days),
                'wellbeing_metrics': self.calculate_wellbeing_metrics(user_id, days),
                'behavioral_insights': self.generate_behavioral_insights(user_id, days),
                'predictive_insights': self.generate_predictive_insights(user_id, days),
                'recommendations': self.generate_analytics_recommendations(user_id, days)
            }
            
            return analytics
            
        except Exception as e:
            print(f"Error in comprehensive analytics: {e}")
            return {}
    
    def analyze_mood_trends(self, user_id, days):
        """Analyze mood trends over time"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.asc()).all()
        
        if len(entries) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Daily sentiment trends
        daily_data = defaultdict(list)
        for entry in entries:
            date = entry.timestamp.date()
            if entry.sentiment_score is not None:
                daily_data[date].append(entry.sentiment_score)
        
        # Calculate daily averages
        daily_trends = []
        for date, scores in sorted(daily_data.items()):
            daily_trends.append({
                'date': date.isoformat(),
                'avg_sentiment': np.mean(scores),
                'entry_count': len(scores),
                'volatility': np.std(scores) if len(scores) > 1 else 0
            })
        
        # Calculate overall trend
        if len(daily_trends) >= 3:
            sentiments = [day['avg_sentiment'] for day in daily_trends]
            trend_slope = np.polyfit(range(len(sentiments)), sentiments, 1)[0]
            
            if trend_slope > 0.01:
                trend_direction = 'improving'
            elif trend_slope < -0.01:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        # Weekly patterns
        weekly_patterns = self._analyze_weekly_patterns(entries)
        
        # Time-of-day patterns
        time_patterns = self._analyze_time_patterns(entries)
        
        return {
            'daily_trends': daily_trends,
            'trend_direction': trend_direction,
            'weekly_patterns': weekly_patterns,
            'time_patterns': time_patterns,
            'volatility_score': self._calculate_volatility_score(daily_trends)
        }
    
    def analyze_emotional_patterns(self, user_id, days):
        """Analyze emotional patterns and transitions"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.asc()).all()
        
        if len(entries) < 2:
            return {'message': 'Insufficient data for emotional pattern analysis'}
        
        # Emotion frequency
        emotion_counts = defaultdict(int)
        emotion_sentiments = defaultdict(list)
        
        for entry in entries:
            emotion = entry.emotion_category or 'unknown'
            emotion_counts[emotion] += 1
            if entry.sentiment_score is not None:
                emotion_sentiments[emotion].append(entry.sentiment_score)
        
        # Calculate emotion statistics
        emotion_stats = {}
        for emotion, sentiments in emotion_sentiments.items():
            if sentiments:
                emotion_stats[emotion] = {
                    'frequency': emotion_counts[emotion],
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments),
                    'percentage': (emotion_counts[emotion] / len(entries)) * 100
                }
        
        # Emotion transitions
        transitions = self._analyze_emotion_transitions(entries)
        
        # Emotional stability
        stability_score = self._calculate_emotional_stability(entries)
        
        return {
            'emotion_distribution': dict(emotion_counts),
            'emotion_statistics': emotion_stats,
            'emotion_transitions': transitions,
            'emotional_stability': stability_score,
            'dominant_emotions': self._get_dominant_emotions(emotion_counts)
        }
    
    def analyze_stress_patterns(self, user_id, days):
        """Analyze stress patterns and triggers"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.asc()).all()
        
        if not entries:
            return {'message': 'No data available for stress analysis'}
        
        # Stress level distribution
        stress_levels = defaultdict(int)
        stress_sentiments = defaultdict(list)
        mental_health_concerns = 0
        
        for entry in entries:
            stress_level = entry.stress_level or 'unknown'
            stress_levels[stress_level] += 1
            
            if entry.sentiment_score is not None:
                stress_sentiments[stress_level].append(entry.sentiment_score)
            
            if entry.mental_health_concern:
                mental_health_concerns += 1
        
        # Stress trends
        stress_trends = self._analyze_stress_trends(entries)
        
        # Stress triggers (based on journal text analysis)
        stress_triggers = self._identify_stress_triggers(entries)
        
        # Stress management effectiveness
        stress_management = self._analyze_stress_management(entries)
        
        return {
            'stress_distribution': dict(stress_levels),
            'stress_statistics': {
                level: {
                    'count': count,
                    'avg_sentiment': np.mean(sentiments) if sentiments else 0,
                    'percentage': (count / len(entries)) * 100
                }
                for level, count in stress_levels.items()
                for sentiments in [stress_sentiments[level]]
            },
            'mental_health_concerns': mental_health_concerns,
            'stress_trends': stress_trends,
            'stress_triggers': stress_triggers,
            'stress_management': stress_management,
            'stress_score': self._calculate_stress_score(stress_levels, mental_health_concerns)
        }
    
    def calculate_wellbeing_metrics(self, user_id, days):
        """Calculate comprehensive wellbeing metrics"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.asc()).all()
        
        if not entries:
            return {'message': 'No data available for wellbeing calculation'}
        
        # Basic metrics
        total_entries = len(entries)
        sentiment_scores = [e.sentiment_score for e in entries if e.sentiment_score is not None]
        
        if not sentiment_scores:
            return {'message': 'No sentiment data available'}
        
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_volatility = np.std(sentiment_scores)
        
        # Wellbeing components
        emotional_wellbeing = self._calculate_emotional_wellbeing(entries)
        stress_wellbeing = self._calculate_stress_wellbeing(entries)
        mental_health_wellbeing = self._calculate_mental_health_wellbeing(entries)
        
        # Overall wellbeing score
        overall_wellbeing = (
            emotional_wellbeing * 0.4 +
            stress_wellbeing * 0.3 +
            mental_health_wellbeing * 0.3
        )
        
        # Wellbeing trends
        wellbeing_trend = self._calculate_wellbeing_trend(entries)
        
        return {
            'overall_score': overall_wellbeing,
            'emotional_wellbeing': emotional_wellbeing,
            'stress_wellbeing': stress_wellbeing,
            'mental_health_wellbeing': mental_health_wellbeing,
            'wellbeing_trend': wellbeing_trend,
            'consistency_score': 100 - (sentiment_volatility * 50),  # Lower volatility = higher consistency
            'engagement_score': min(100, (total_entries / days) * 10),  # Entries per day * 10
            'metrics': {
                'avg_sentiment': avg_sentiment,
                'sentiment_volatility': sentiment_volatility,
                'total_entries': total_entries,
                'entries_per_day': total_entries / days
            }
        }
    
    def generate_behavioral_insights(self, user_id, days):
        """Generate behavioral insights and patterns"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.asc()).all()
        
        if len(entries) < 5:
            return {'message': 'Insufficient data for behavioral insights'}
        
        insights = {
            'journaling_patterns': self._analyze_journaling_patterns(entries),
            'mood_cycles': self._detect_mood_cycles(entries),
            'recovery_patterns': self._analyze_recovery_patterns(entries),
            'trigger_patterns': self._identify_trigger_patterns(entries),
            'coping_effectiveness': self._analyze_coping_effectiveness(entries)
        }
        
        return insights
    
    def generate_predictive_insights(self, user_id, days):
        """Generate predictive insights based on patterns"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        entries = MoodEntry.query.filter(
            MoodEntry.user_id == user_id,
            MoodEntry.timestamp >= start_date
        ).order_by(MoodEntry.timestamp.asc()).all()
        
        if len(entries) < 10:
            return {'message': 'Insufficient data for predictive analysis'}
        
        predictions = {
            'mood_forecast': self._predict_mood_trend(entries),
            'risk_factors': self._identify_risk_factors(entries),
            'optimal_intervention_times': self._identify_intervention_times(entries),
            'seasonal_patterns': self._analyze_seasonal_patterns(entries)
        }
        
        return predictions
    
    def generate_analytics_recommendations(self, user_id, days):
        """Generate personalized recommendations based on analytics"""
        analytics = self.get_comprehensive_analytics(user_id, days)
        
        recommendations = []
        
        # Mood trend recommendations
        if 'mood_trends' in analytics:
            trend = analytics['mood_trends'].get('trend_direction', 'stable')
            if trend == 'declining':
                recommendations.append({
                    'category': 'mood_trend',
                    'priority': 'high',
                    'recommendation': 'Your mood trend is declining. Consider increasing self-care activities and reaching out for support.',
                    'action_items': ['Schedule regular check-ins', 'Increase physical activity', 'Consider professional support']
                })
        
        # Stress recommendations
        if 'stress_analysis' in analytics:
            stress_score = analytics['stress_analysis'].get('stress_score', 0)
            if stress_score > 70:
                recommendations.append({
                    'category': 'stress_management',
                    'priority': 'high',
                    'recommendation': 'High stress levels detected. Focus on stress management techniques.',
                    'action_items': ['Practice daily meditation', 'Implement relaxation techniques', 'Consider stress management counseling']
                })
        
        # Wellbeing recommendations
        if 'wellbeing_metrics' in analytics:
            wellbeing = analytics['wellbeing_metrics'].get('overall_score', 50)
            if wellbeing < 40:
                recommendations.append({
                    'category': 'wellbeing',
                    'priority': 'high',
                    'recommendation': 'Overall wellbeing is low. Consider comprehensive mental health support.',
                    'action_items': ['Seek professional mental health support', 'Develop daily wellness routine', 'Connect with support network']
                })
        
        return recommendations
    
    # Helper methods
    def _analyze_weekly_patterns(self, entries):
        """Analyze mood patterns by day of week"""
        weekly_data = defaultdict(list)
        for entry in entries:
            day = entry.timestamp.strftime('%A')
            if entry.sentiment_score is not None:
                weekly_data[day].append(entry.sentiment_score)
        
        weekly_avg = {}
        for day, scores in weekly_data.items():
            weekly_avg[day] = np.mean(scores)
        
        return weekly_avg
    
    def _analyze_time_patterns(self, entries):
        """Analyze mood patterns by time of day"""
        time_data = defaultdict(list)
        for entry in entries:
            hour = entry.timestamp.hour
            time_period = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening'
            
            if entry.sentiment_score is not None:
                time_data[time_period].append(entry.sentiment_score)
        
        time_avg = {}
        for period, scores in time_data.items():
            time_avg[period] = np.mean(scores)
        
        return time_avg
    
    def _analyze_emotion_transitions(self, entries):
        """Analyze transitions between emotions"""
        transitions = defaultdict(int)
        for i in range(len(entries) - 1):
            current_emotion = entries[i].emotion_category or 'unknown'
            next_emotion = entries[i + 1].emotion_category or 'unknown'
            transition = f"{current_emotion} -> {next_emotion}"
            transitions[transition] += 1
        
        return dict(transitions)
    
    def _calculate_volatility_score(self, daily_trends):
        """Calculate mood volatility score"""
        if len(daily_trends) < 2:
            return 0
        
        volatilities = [day['volatility'] for day in daily_trends]
        return np.mean(volatilities) * 100  # Scale to 0-100
    
    def _calculate_emotional_stability(self, entries):
        """Calculate emotional stability score"""
        if len(entries) < 3:
            return 50  # Neutral score
        
        emotions = [entry.emotion_category for entry in entries if entry.emotion_category]
        if not emotions:
            return 50
        
        # Count unique emotions vs total entries
        unique_emotions = len(set(emotions))
        total_entries = len(emotions)
        
        # Higher stability = fewer emotion changes
        stability = (1 - (unique_emotions / total_entries)) * 100
        return max(0, min(100, stability))
    
    def _get_dominant_emotions(self, emotion_counts):
        """Get dominant emotions"""
        if not emotion_counts:
            return []
        
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        return [emotion for emotion, count in sorted_emotions[:3]]
    
    def _analyze_stress_trends(self, entries):
        """Analyze stress level trends over time"""
        stress_timeline = []
        for entry in entries:
            stress_timeline.append({
                'date': entry.timestamp.date().isoformat(),
                'stress_level': entry.stress_level or 'unknown',
                'sentiment_score': entry.sentiment_score
            })
        
        return stress_timeline
    
    def _identify_stress_triggers(self, entries):
        """Identify potential stress triggers from journal text"""
        # This is a simplified version - in production, you'd use NLP to identify triggers
        stress_keywords = ['work', 'deadline', 'exam', 'relationship', 'money', 'health', 'family']
        triggers = defaultdict(int)
        
        for entry in entries:
            if entry.journal_text:
                text_lower = entry.journal_text.lower()
                for keyword in stress_keywords:
                    if keyword in text_lower and entry.stress_level in ['high', 'medium']:
                        triggers[keyword] += 1
        
        return dict(triggers)
    
    def _analyze_stress_management(self, entries):
        """Analyze effectiveness of stress management"""
        # Look for patterns in stress recovery
        stress_entries = [e for e in entries if e.stress_level in ['high', 'medium']]
        
        if len(stress_entries) < 2:
            return {'message': 'Insufficient stress data for analysis'}
        
        # Calculate average recovery time
        recovery_times = []
        for i in range(len(stress_entries) - 1):
            if stress_entries[i].stress_level == 'high' and stress_entries[i + 1].stress_level in ['low', 'medium']:
                time_diff = (stress_entries[i + 1].timestamp - stress_entries[i].timestamp).total_seconds() / 3600  # hours
                recovery_times.append(time_diff)
        
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        return {
            'avg_recovery_time_hours': avg_recovery_time,
            'stress_episodes': len(stress_entries),
            'recovery_rate': len(recovery_times) / len(stress_entries) if stress_entries else 0
        }
    
    def _calculate_stress_score(self, stress_levels, mental_health_concerns):
        """Calculate overall stress score"""
        if not stress_levels:
            return 0
        
        total_entries = sum(stress_levels.values())
        high_stress = stress_levels.get('high', 0)
        medium_stress = stress_levels.get('medium', 0)
        
        stress_score = ((high_stress * 2 + medium_stress) / total_entries) * 100
        stress_score += mental_health_concerns * 10  # Add penalty for mental health concerns
        
        return min(100, stress_score)
    
    def _calculate_emotional_wellbeing(self, entries):
        """Calculate emotional wellbeing score"""
        sentiment_scores = [e.sentiment_score for e in entries if e.sentiment_score is not None]
        if not sentiment_scores:
            return 50
        
        avg_sentiment = np.mean(sentiment_scores)
        return (avg_sentiment + 1) * 50  # Convert -1,1 to 0,100
    
    def _calculate_stress_wellbeing(self, entries):
        """Calculate stress wellbeing score"""
        stress_levels = [e.stress_level for e in entries if e.stress_level]
        if not stress_levels:
            return 50
        
        stress_scores = {'low': 100, 'medium': 50, 'high': 0}
        avg_stress_score = np.mean([stress_scores.get(level, 50) for level in stress_levels])
        
        return avg_stress_score
    
    def _calculate_mental_health_wellbeing(self, entries):
        """Calculate mental health wellbeing score"""
        mental_health_concerns = sum(1 for e in entries if e.mental_health_concern)
        total_entries = len(entries)
        
        if total_entries == 0:
            return 50
        
        concern_percentage = (mental_health_concerns / total_entries) * 100
        wellbeing_score = 100 - concern_percentage
        
        return max(0, wellbeing_score)
    
    def _calculate_wellbeing_trend(self, entries):
        """Calculate wellbeing trend over time"""
        if len(entries) < 4:
            return 'insufficient_data'
        
        # Split entries into two halves
        mid_point = len(entries) // 2
        first_half = entries[:mid_point]
        second_half = entries[mid_point:]
        
        # Calculate average sentiment for each half
        first_sentiments = [e.sentiment_score for e in first_half if e.sentiment_score is not None]
        second_sentiments = [e.sentiment_score for e in second_half if e.sentiment_score is not None]
        
        if not first_sentiments or not second_sentiments:
            return 'insufficient_data'
        
        first_avg = np.mean(first_sentiments)
        second_avg = np.mean(second_sentiments)
        
        if second_avg > first_avg + 0.1:
            return 'improving'
        elif second_avg < first_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _analyze_journaling_patterns(self, entries):
        """Analyze journaling patterns"""
        if not entries:
            return {}
        
        # Analyze journal text length patterns
        text_lengths = [len(e.journal_text or '') for e in entries]
        
        return {
            'avg_text_length': np.mean(text_lengths),
            'journaling_frequency': len(entries),
            'consistency_score': 100 - (np.std(text_lengths) / np.mean(text_lengths) * 100) if text_lengths else 0
        }
    
    def _detect_mood_cycles(self, entries):
        """Detect mood cycles and patterns"""
        if len(entries) < 7:
            return {'message': 'Insufficient data for cycle detection'}
        
        # Simple cycle detection based on sentiment scores
        sentiments = [e.sentiment_score for e in entries if e.sentiment_score is not None]
        
        if len(sentiments) < 7:
            return {'message': 'Insufficient sentiment data'}
        
        # Look for patterns in sentiment changes
        sentiment_changes = np.diff(sentiments)
        
        return {
            'cycle_length': len(sentiments),
            'volatility': np.std(sentiment_changes),
            'pattern_type': 'stable' if np.std(sentiment_changes) < 0.5 else 'volatile'
        }
    
    def _analyze_recovery_patterns(self, entries):
        """Analyze recovery patterns from low moods"""
        low_mood_entries = [e for e in entries if e.sentiment_score is not None and e.sentiment_score < -0.3]
        
        if len(low_mood_entries) < 2:
            return {'message': 'Insufficient low mood data'}
        
        recovery_times = []
        for i in range(len(low_mood_entries) - 1):
            if low_mood_entries[i].sentiment_score < -0.3:
                # Find next entry with better sentiment
                for j in range(i + 1, len(entries)):
                    if entries[j].sentiment_score is not None and entries[j].sentiment_score > -0.1:
                        time_diff = (entries[j].timestamp - low_mood_entries[i].timestamp).total_seconds() / 3600
                        recovery_times.append(time_diff)
                        break
        
        return {
            'avg_recovery_time_hours': np.mean(recovery_times) if recovery_times else 0,
            'recovery_episodes': len(recovery_times),
            'recovery_rate': len(recovery_times) / len(low_mood_entries)
        }
    
    def _identify_trigger_patterns(self, entries):
        """Identify trigger patterns"""
        # This is a simplified version - in production, you'd use more sophisticated NLP
        trigger_keywords = {
            'work': ['work', 'job', 'boss', 'colleague', 'meeting', 'deadline'],
            'relationships': ['friend', 'family', 'partner', 'relationship', 'argument'],
            'health': ['sick', 'pain', 'doctor', 'hospital', 'medication'],
            'finances': ['money', 'bills', 'debt', 'expensive', 'cost']
        }
        
        triggers = defaultdict(int)
        
        for entry in entries:
            if entry.journal_text:
                text_lower = entry.journal_text.lower()
                for category, keywords in trigger_keywords.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            triggers[category] += 1
                            break
        
        return dict(triggers)
    
    def _analyze_coping_effectiveness(self, entries):
        """Analyze effectiveness of coping strategies"""
        # Look for positive sentiment after negative entries
        negative_entries = [e for e in entries if e.sentiment_score is not None and e.sentiment_score < -0.2]
        
        coping_success = 0
        for neg_entry in negative_entries:
            # Find next entry within 24 hours
            next_day = neg_entry.timestamp + timedelta(hours=24)
            next_entries = [e for e in entries if e.timestamp > neg_entry.timestamp and e.timestamp <= next_day]
            
            if next_entries:
                next_sentiment = next_entries[0].sentiment_score
                if next_sentiment is not None and next_sentiment > neg_entry.sentiment_score + 0.3:
                    coping_success += 1
        
        return {
            'coping_success_rate': coping_success / len(negative_entries) if negative_entries else 0,
            'negative_episodes': len(negative_entries),
            'successful_coping': coping_success
        }
    
    def _predict_mood_trend(self, entries):
        """Predict future mood trend"""
        if len(entries) < 5:
            return {'message': 'Insufficient data for prediction'}
        
        # Simple linear regression on recent sentiment scores
        recent_entries = entries[-10:]  # Last 10 entries
        sentiments = [e.sentiment_score for e in recent_entries if e.sentiment_score is not None]
        
        if len(sentiments) < 3:
            return {'message': 'Insufficient sentiment data'}
        
        # Calculate trend
        x = np.arange(len(sentiments))
        trend_slope = np.polyfit(x, sentiments, 1)[0]
        
        # Predict next few days
        future_predictions = []
        for i in range(1, 8):  # Next 7 days
            predicted_sentiment = sentiments[-1] + (trend_slope * i)
            future_predictions.append({
                'day': i,
                'predicted_sentiment': max(-1, min(1, predicted_sentiment))
            })
        
        return {
            'trend_slope': trend_slope,
            'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
            'future_predictions': future_predictions,
            'confidence': min(100, len(sentiments) * 10)  # More data = higher confidence
        }
    
    def _identify_risk_factors(self, entries):
        """Identify risk factors for mental health"""
        risk_factors = []
        
        # Recent decline in mood
        recent_entries = entries[-5:] if len(entries) >= 5 else entries
        recent_sentiments = [e.sentiment_score for e in recent_entries if e.sentiment_score is not None]
        
        if len(recent_sentiments) >= 3:
            if np.mean(recent_sentiments) < -0.5:
                risk_factors.append({
                    'factor': 'recent_mood_decline',
                    'severity': 'high',
                    'description': 'Significant decline in mood over recent entries'
                })
        
        # High stress levels
        high_stress_count = sum(1 for e in entries[-10:] if e.stress_level == 'high')
        if high_stress_count >= 3:
            risk_factors.append({
                'factor': 'persistent_high_stress',
                'severity': 'medium',
                'description': 'Consistently high stress levels'
            })
        
        # Mental health concerns
        mental_health_count = sum(1 for e in entries[-10:] if e.mental_health_concern)
        if mental_health_count >= 2:
            risk_factors.append({
                'factor': 'mental_health_concerns',
                'severity': 'high',
                'description': 'Multiple mental health concerns expressed'
            })
        
        return risk_factors
    
    def _identify_intervention_times(self, entries):
        """Identify optimal times for interventions"""
        if len(entries) < 5:
            return {'message': 'Insufficient data for intervention analysis'}
        
        # Analyze patterns to identify when user is most receptive
        intervention_times = []
        
        # Look for patterns in positive responses
        positive_entries = [e for e in entries if e.sentiment_score is not None and e.sentiment_score > 0.3]
        
        if positive_entries:
            # Analyze time patterns of positive entries
            time_patterns = defaultdict(int)
            for entry in positive_entries:
                hour = entry.timestamp.hour
                time_period = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening'
                time_patterns[time_period] += 1
            
            best_time = max(time_patterns, key=time_patterns.get)
            intervention_times.append({
                'type': 'optimal_engagement_time',
                'time': best_time,
                'reason': f'User shows most positive engagement during {best_time}'
            })
        
        return intervention_times
    
    def _analyze_seasonal_patterns(self, entries):
        """Analyze seasonal patterns in mood"""
        if len(entries) < 30:  # Need at least a month of data
            return {'message': 'Insufficient data for seasonal analysis'}
        
        # Group entries by month
        monthly_data = defaultdict(list)
        for entry in entries:
            month = entry.timestamp.month
            if entry.sentiment_score is not None:
                monthly_data[month].append(entry.sentiment_score)
        
        monthly_avg = {}
        for month, scores in monthly_data.items():
            monthly_avg[month] = np.mean(scores)
        
        return {
            'monthly_patterns': monthly_avg,
            'seasonal_trend': 'detected' if len(monthly_avg) >= 3 else 'insufficient_data'
        }
