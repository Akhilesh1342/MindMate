"""
Utility functions and helpers for MindMate
"""

import re
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from functools import wraps
import json

def sanitize_text(text: str, max_length: int = 5000) -> str:
    """Sanitize and clean text input"""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."
    
    return text

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username: str) -> bool:
    """Validate username format"""
    if not username or len(username) < 3 or len(username) > 20:
        return False
    
    # Only alphanumeric characters and underscores
    pattern = r'^[a-zA-Z0-9_]+$'
    return re.match(pattern, username) is not None

def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """Hash password securely"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{password_hash.hex()}"

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    try:
        salt, hash_hex = password_hash.split(':')
        password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return secrets.compare_digest(hash_hex, password_hash_check.hex())
    except ValueError:
        return False

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    if not timestamp:
        return "Never"
    
    now = datetime.utcnow()
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"

def calculate_age_in_days(start_date: datetime, end_date: datetime = None) -> int:
    """Calculate age in days between two dates"""
    if end_date is None:
        end_date = datetime.utcnow()
    
    return (end_date - start_date).days

def normalize_sentiment_score(score: float) -> float:
    """Normalize sentiment score to 0-100 range"""
    if score is None:
        return 50  # Neutral
    
    # Convert from -1,1 range to 0,100 range
    normalized = ((score + 1) / 2) * 100
    return max(0, min(100, normalized))

def calculate_emotional_intensity(sentiment_score: float, confidence: float) -> str:
    """Calculate emotional intensity level"""
    if confidence < 0.3:
        return "low"
    
    abs_score = abs(sentiment_score)
    if abs_score > 0.7:
        return "high"
    elif abs_score > 0.4:
        return "medium"
    else:
        return "low"

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text"""
    if not text:
        return []
    
    # Remove special characters and split into words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter by length and common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Count frequency and return most common
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:10]]

def detect_language(text: str) -> str:
    """Simple language detection"""
    if not text:
        return "unknown"
    
    # Simple heuristic based on common words
    english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    english_count = sum(1 for word in words if word in english_words)
    
    if len(words) > 0 and english_count / len(words) > 0.3:
        return "english"
    else:
        return "unknown"

def create_time_series_data(data: List[Dict], date_field: str, value_field: str, 
                          start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
    """Create time series data with missing dates filled"""
    if not data:
        return []
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    df[date_field] = pd.to_datetime(df[date_field])
    
    # Set date range
    if start_date is None:
        start_date = df[date_field].min()
    if end_date is None:
        end_date = df[date_field].max()
    
    # Create complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Fill missing dates with None values
    df_complete = df.set_index(date_field).reindex(date_range).reset_index()
    df_complete[date_field] = df_complete['index']
    df_complete = df_complete.drop('index', axis=1)
    
    return df_complete.to_dict('records')

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {}
    
    values = [v for v in values if v is not None]
    if not values:
        return {}
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'count': len(values)
    }

def create_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Create confidence interval for a list of values"""
    if not values or len(values) < 2:
        return {}
    
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return {}
    
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    # Calculate margin of error
    margin_error = std / np.sqrt(n) * 1.96  # 95% confidence interval
    
    return {
        'mean': mean,
        'lower_bound': mean - margin_error,
        'upper_bound': mean + margin_error,
        'margin_error': margin_error
    }

def validate_json_data(data: Any, required_fields: List[str]) -> Dict[str, Any]:
    """Validate JSON data and return validation results"""
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(data, dict):
        result['valid'] = False
        result['errors'].append("Data must be a dictionary")
        return result
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            result['valid'] = False
            result['errors'].append(f"Missing required field: {field}")
    
    return result

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely load JSON string"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely dump data to JSON string"""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default

def create_pagination_info(page: int, per_page: int, total: int) -> Dict[str, Any]:
    """Create pagination information"""
    total_pages = (total + per_page - 1) // per_page
    
    return {
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < total_pages else None
    }

def rate_limit_key(identifier: str, endpoint: str) -> str:
    """Generate rate limit key"""
    return f"rate_limit:{identifier}:{endpoint}"

def is_rate_limited(identifier: str, endpoint: str, limit: int, window: int) -> bool:
    """Check if request is rate limited (simplified version)"""
    # In a real implementation, you'd use Redis or similar
    # This is a placeholder for the logic
    return False

def log_activity(user_id: int, activity: str, details: Dict[str, Any] = None):
    """Log user activity (placeholder for logging system)"""
    timestamp = datetime.utcnow()
    log_entry = {
        'timestamp': timestamp.isoformat(),
        'user_id': user_id,
        'activity': activity,
        'details': details or {}
    }
    
    # In a real implementation, you'd write to a log file or database
    print(f"Activity Log: {log_entry}")

def create_backup_filename(prefix: str = "backup") -> str:
    """Create backup filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Validate file extension"""
    if not filename:
        return False
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        import os
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0

def create_error_response(message: str, code: int = 400, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        'error': True,
        'message': message,
        'code': code,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if details:
        response['details'] = details
    
    return response

def create_success_response(message: str, data: Any = None, code: int = 200) -> Dict[str, Any]:
    """Create standardized success response"""
    response = {
        'success': True,
        'message': message,
        'code': code,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    return response

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    continue
            
            raise last_exception
        
        return wrapper
    return decorator

def measure_execution_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        result = func(*args, **kwargs)
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds()
        print(f"{func.__name__} executed in {execution_time:.3f} seconds")
        
        return result
    
    return wrapper
