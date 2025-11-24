from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io
import requests
import json
import time
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Use correct model name
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# API Keys for recommendations
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
GOOGLE_BOOKS_API_KEY = os.getenv('GOOGLE_BOOKS_API_KEY')

# Load emotion detection model
try:
    emotion_model = load_model('emotion_model_final.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    print("✅ Emotion detection model loaded successfully")
except Exception as e:
    print(f"❌ Error loading emotion model: {e}")
    emotion_model = None

# Updated Onboarding questions
ONBOARDING_QUESTIONS = [
    {
        "id": 1,
        "question": "Hi there! 👋 Let's understand your preferences better.\n\nFirst, what language would you like me to use for recommendations and suggestions?\n\n(For example: English, Spanish, French, Hindi, Tamil, Telugu, Malayalam, Kannada, etc.)",
        "emotion": None,
        "category": "language"
    },
    {
        "id": 2,
        "question": "Great! Now, when you're feeling HAPPY, what are your preferences?\n\n📝 Format: music genre, book type, movie genre\n(Example: pop, romance, comedy)\n\nYour answer:",
        "emotion": "happy",
        "category": "all"
    },
    {
        "id": 3,
        "question": "Thanks! Now, when you're feeling SAD or down, what are your preferences?\n\n📝 Format: music genre, book type, movie genre\n(Example: slow songs, fiction, drama)\n\nYour answer:",
        "emotion": "sad",
        "category": "all"
    },
    {
        "id": 4,
        "question": "I appreciate you sharing. When you're feeling ANGRY or frustrated, what are your preferences?\n\n📝 Format: music genre, book type, movie genre\n(Example: rock, thriller, action)\n\nYour answer:",
        "emotion": "angry",
        "category": "all"
    },
    {
        "id": 5,
        "question": "When you're feeling ANXIOUS or FEARFUL, what are your preferences?\n\n📝 Format: music genre, book type, movie genre\n(Example: calm music, self-help, documentary)\n\nYour answer:",
        "emotion": "fear",
        "category": "all"
    },
    {
        "id": 6,
        "question": "Last question! When you want to feel RELAXED or at peace, what are your preferences?\n\n📝 Format: music genre, book type, movie genre\n(Example: jazz, poetry, nature films)\n\nYour answer:",
        "emotion": "neutral",
        "category": "all"
    }
]

# Store user sessions
user_sessions = {}

# Spotify token cache
spotify_token = None
spotify_token_expires = 0


# ==================== HELPER FUNCTIONS ====================

def get_utc_timestamp():
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc).isoformat()


def get_language_config(language):
    """Get language-specific configuration for searching native content"""
    language_configs = {
        'tamil': {
            'music_terms': ['tamil songs', 'tamil music', 'kollywood music', 'tamil'],
            'book_terms': ['tamil books', 'tamil literature'],
            'movie_terms': ['tamil movies', 'kollywood'],
            'market': 'IN',
            'lang_code': 'ta',
            'tmdb_lang': 'ta'
        },
        'hindi': {
            'music_terms': ['hindi songs', 'bollywood music', 'hindi music', 'hindi'],
            'book_terms': ['hindi books', 'hindi literature'],
            'movie_terms': ['hindi movies', 'bollywood'],
            'market': 'IN',
            'lang_code': 'hi',
            'tmdb_lang': 'hi'
        },
        'telugu': {
            'music_terms': ['telugu songs', 'tollywood music', 'telugu'],
            'book_terms': ['telugu books', 'telugu literature'],
            'movie_terms': ['telugu movies', 'tollywood'],
            'market': 'IN',
            'lang_code': 'te',
            'tmdb_lang': 'te'
        },
        'malayalam': {
            'music_terms': ['malayalam songs', 'mollywood music', 'malayalam'],
            'book_terms': ['malayalam books', 'malayalam literature'],
            'movie_terms': ['malayalam movies', 'mollywood'],
            'market': 'IN',
            'lang_code': 'ml',
            'tmdb_lang': 'ml'
        },
        'kannada': {
            'music_terms': ['kannada songs', 'sandalwood music', 'kannada'],
            'book_terms': ['kannada books', 'kannada literature'],
            'movie_terms': ['kannada movies', 'sandalwood'],
            'market': 'IN',
            'lang_code': 'kn',
            'tmdb_lang': 'kn'
        },
        'spanish': {
            'music_terms': ['spanish music', 'musica latina', 'spanish'],
            'book_terms': ['spanish books', 'literatura española'],
            'movie_terms': ['spanish movies', 'cine español'],
            'market': 'ES',
            'lang_code': 'es',
            'tmdb_lang': 'es'
        },
        'french': {
            'music_terms': ['french music', 'musique française', 'french'],
            'book_terms': ['french books', 'littérature française'],
            'movie_terms': ['french movies', 'cinéma français'],
            'market': 'FR',
            'lang_code': 'fr',
            'tmdb_lang': 'fr'
        },
        'marathi': {
            'music_terms': ['marathi songs', 'marathi music', 'marathi'],
            'book_terms': ['marathi books', 'marathi literature'],
            'movie_terms': ['marathi movies', 'marathi cinema'],
            'market': 'IN',
            'lang_code': 'mr',
            'tmdb_lang': 'mr'
        },
        'bengali': {
            'music_terms': ['bengali songs', 'bengali music', 'bengali'],
            'book_terms': ['bengali books', 'bangla literature'],
            'movie_terms': ['bengali movies', 'tollywood bengali'],
            'market': 'IN',
            'lang_code': 'bn',
            'tmdb_lang': 'bn'
        },
        'punjabi': {
            'music_terms': ['punjabi songs', 'punjabi music', 'punjabi'],
            'book_terms': ['punjabi books', 'punjabi literature'],
            'movie_terms': ['punjabi movies', 'pollywood'],
            'market': 'IN',
            'lang_code': 'pa',
            'tmdb_lang': 'pa'
        },
        'english': {
            'music_terms': ['music', 'songs'],
            'book_terms': ['books', 'literature'],
            'movie_terms': ['movies', 'films'],
            'market': 'US',
            'lang_code': 'en',
            'tmdb_lang': 'en'
        }
    }
    
    lang_key = language.lower()
    return language_configs.get(lang_key, language_configs['english'])


def get_spotify_token():
    """Get Spotify access token"""
    global spotify_token, spotify_token_expires
    
    if spotify_token and time.time() < spotify_token_expires:
        return spotify_token
    
    try:
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_response = requests.post(auth_url, {
            'grant_type': 'client_credentials',
            'client_id': SPOTIFY_CLIENT_ID,
            'client_secret': SPOTIFY_CLIENT_SECRET,
        })
        auth_data = auth_response.json()
        spotify_token = auth_data['access_token']
        spotify_token_expires = time.time() + auth_data['expires_in'] - 60
        return spotify_token
    except Exception as e:
        print(f"Error getting Spotify token: {e}")
        return None


def get_user_language(user_id):
    """Get user's preferred language from database"""
    try:
        result = supabase.table('user_profiles')\
            .select('preferred_language')\
            .eq('id', user_id)\
            .execute()
        
        if result.data and len(result.data) > 0:
            lang = result.data[0].get('preferred_language', 'English')
            print(f"🌍 Found language in profile: {lang}")
            return lang
        
        return 'English'
    except Exception as e:
        print(f"❌ Error fetching user language: {e}")
        return 'English'


def get_user_preferences(user_id):
    """Get all user preferences from user_profiles table with better error handling"""
    try:
        print(f"🔍 Fetching preferences for user: {user_id}")
        
        if not user_id or user_id == 'guest_user':
            print("⚠️ Guest user - returning None")
            return None
        
        result = supabase.table('user_profiles')\
            .select('*')\
            .eq('id', user_id)\
            .execute()
        
        if hasattr(result, 'data') and result.data and len(result.data) > 0:
            user_prefs = result.data[0]
            print(f"✅ Found preferences")
            
            # Debug: print all emotion-based preferences
            for emotion in ['happy', 'sad', 'angry', 'fear', 'neutral']:
                music_key = f'{emotion}_music'
                books_key = f'{emotion}_books'
                movies_key = f'{emotion}_movies'
                print(f"   {emotion}: music='{user_prefs.get(music_key)}', books='{user_prefs.get(books_key)}', movies='{user_prefs.get(movies_key)}'")
            
            return user_prefs
        
        print("⚠️ No data found for user")
        return None
        
    except Exception as e:
        print(f"❌ Error fetching preferences: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_genres_from_user_response(user_response):
    """Extract genres from user's response"""
    try:
        prompt = f"""Extract the EXACT genres mentioned by the user from this text: "{user_response}"

The user specified what music, books, and movies they prefer. Extract them precisely.

Return ONLY a JSON object:
{{
    "music": ["genre1"],
    "books": ["type1"],
    "movies": ["genre1"]
}}

Examples:
"rock, thriller, action" -> {{"music": ["rock"], "books": ["thriller"], "movies": ["action"]}}
"pop songs, romance books, comedy films" -> {{"music": ["pop"], "books": ["romance"], "movies": ["comedy"]}}

Keep genres lowercase and simple. If 3 items are given, assume order: music, books, movies."""

        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        extracted = json.loads(result_text)
        print(f"✅ Extracted genres via Gemini: {extracted}")
        return extracted
        
    except Exception as e:
        print(f"⚠️ Gemini extraction failed: {e}, trying fallback parsing...")
        
        try:
            parts = [p.strip() for p in user_response.replace(' and ', ',').split(',')]
            
            if len(parts) >= 3:
                extracted = {
                    'music': [parts[0]],
                    'books': [parts[1]],
                    'movies': [parts[2]]
                }
                print(f"✅ Fallback extraction successful: {extracted}")
                return extracted
            elif len(parts) == 1:
                extracted = {
                    'music': [parts[0]],
                    'books': [parts[0]],
                    'movies': [parts[0]]
                }
                print(f"✅ Fallback extraction (single): {extracted}")
                return extracted
            
        except Exception as fallback_error:
            print(f"❌ Fallback parsing also failed: {fallback_error}")
        
        return None


def validate_language_response(user_answer):
    """Validate if the response is a valid language"""
    validation_prompt = f"""Is this a valid language name? "{user_answer}"

Respond with ONLY a JSON object:
{{
    "valid": true/false,
    "language": "standardized language name or null",
    "feedback": "brief message if invalid"
}}

Examples:
"Tamil" -> {{"valid": true, "language": "Tamil", "feedback": ""}}
"Kannada" -> {{"valid": true, "language": "Kannada", "feedback": ""}}
"""

    try:
        response = model.generate_content(validation_prompt)
        result_text = response.text.strip()
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        return result.get('valid', False), result.get('feedback', ''), result.get('language', 'English')
        
    except Exception as e:
        print(f"Language validation error: {e}")
        common_languages = ['english', 'spanish', 'french', 'hindi', 'tamil', 'telugu', 'kannada', 'malayalam', 'marathi', 'bengali', 'punjabi']
        if user_answer.lower().strip() in common_languages:
            return True, '', user_answer.capitalize()
        return False, 'Please specify a valid language name.', 'English'


def validate_response_with_gemini(user_answer, emotion):
    """Validate if response contains music, books, and movies"""
    validation_prompt = f"""Validate this response for {emotion} preferences: "{user_answer}"

The user should mention their preferences for Music, Books, and Movies.

IMPORTANT: Be VERY lenient with the format. Accept responses like:
- "rock, thriller, action" (assume order: music, books, movies)
- "I like rock music, thriller books, action movies"
- Simple comma-separated: "love, fiction, action"

If the user provides 3 items separated by commas/and, ALWAYS assume they are in order: music, books, movies.

Return JSON:
{{
    "valid": true/false,
    "music": "extracted music genre or null",
    "books": "extracted book type or null",
    "movies": "extracted movie genre or null",
    "missing": ["list of missing categories"],
    "feedback": "message if invalid"
}}"""

    try:
        response = model.generate_content(validation_prompt)
        result_text = response.text.strip()
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        
        is_valid = result.get('valid', False)
        extracted_data = {
            'music': result.get('music'),
            'books': result.get('books'),
            'movies': result.get('movies')
        }
        feedback = result.get('feedback', '')
        
        if extracted_data['music'] and extracted_data['books'] and extracted_data['movies']:
            is_valid = True
            feedback = ''
        
        if not is_valid:
            missing = result.get('missing', [])
            if missing:
                missing_str = ", ".join(missing)
                feedback = f"Please mention {missing_str} for when you feel {emotion}.\n\n📝 Format: music genre, book type, movie genre\nExample: 'rock, thriller, action'"
        
        print(f"✅ Validation result: valid={is_valid}, data={extracted_data}")
        return is_valid, feedback, extracted_data
        
    except Exception as e:
        print(f"⚠️ Validation error: {e}, trying fallback parsing...")
        parts = [p.strip() for p in user_answer.split(',')]
        if len(parts) >= 3:
            print(f"✅ Fallback parsing successful: assuming order music, books, movies")
            return True, '', {
                'music': parts[0],
                'books': parts[1],
                'movies': parts[2]
            }
        
        return False, f"Please mention all three: music genre, book type, and movie genre for when you feel {emotion}.\n\n📝 Format: music, books, movies\nExample: 'rock, thriller, action'", None


def preprocess_face(face_img, target_size=(48, 48)):
    """Preprocess face image for model prediction"""
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, target_size[0], target_size[1], 1)
    
    return face_img


def detect_emotion_from_base64(base64_image):
    """Detect emotion from base64 encoded image"""
    if emotion_model is None:
        return None, 0.0, False, "Emotion model not loaded"
    
    try:
        if ',' in base64_image:
            image_data = base64.b64decode(base64_image.split(',')[1])
        else:
            image_data = base64.b64decode(base64_image)
            
        image = Image.open(io.BytesIO(image_data))
        frame = np.array(image)
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None, 0.0, False, "No face detected"
        
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        face_roi = gray[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        
        predictions = emotion_model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = emotion_labels[emotion_idx]
        
        return emotion, confidence, True, None
        
    except Exception as e:
        return None, 0.0, False, str(e)


# ==================== CHATBOT ENDPOINTS ====================

@app.route('/api/chatbot/start', methods=['POST'])
def start_onboarding():
    """Start a new onboarding session"""
    data = request.json
    user_id = data.get('user_id', 'anonymous')
    
    user_sessions[user_id] = {
        'current_question': 0,
        'responses': [],
        'completed': False,
        'retry_count': 0,
        'language': None
    }
    
    return jsonify({
        'success': True,
        'question': ONBOARDING_QUESTIONS[0]['question'],
        'question_number': 1,
        'total_questions': len(ONBOARDING_QUESTIONS),
        'question_id': ONBOARDING_QUESTIONS[0]['id'],
        'timestamp': get_utc_timestamp()
    })


@app.route('/api/chatbot/answer', methods=['POST'])
def submit_answer():
    """Submit an answer and get the next question"""
    data = request.json
    user_id = data.get('user_id', 'anonymous')
    answer = data.get('answer', '')
    
    if user_id not in user_sessions:
        return jsonify({
            'success': False,
            'error': 'Session not found',
            'timestamp': get_utc_timestamp()
        }), 400
    
    session = user_sessions[user_id]
    current_q_index = session['current_question']
    current_question = ONBOARDING_QUESTIONS[current_q_index]
    
    # Handle language question
    if current_question['category'] == 'language':
        is_valid, feedback, language = validate_language_response(answer)
        
        if not is_valid:
            return jsonify({
                'success': True,
                'completed': False,
                'question': f"🤔 {feedback}\n\n{current_question['question']}",
                'question_number': current_q_index + 1,
                'total_questions': len(ONBOARDING_QUESTIONS),
                'question_id': current_question['id'],
                'retry': True,
                'timestamp': get_utc_timestamp()
            })
        
        session['language'] = language
        try:
            supabase.table('user_profiles').update({
                'preferred_language': language
            }).eq('id', user_id).execute()
            
            print(f"✅ Saved language: {language} for user {user_id}")
        except Exception as e:
            print(f"❌ Error saving language: {e}")
        
        session['responses'].append({
            'question_id': current_question['id'],
            'answer': answer,
            'emotion': None,
            'category': 'language'
        })
    else:
        # Validate media preferences
        is_valid, feedback, extracted_data = validate_response_with_gemini(answer, current_question['emotion'])
        
        if not is_valid:
            return jsonify({
                'success': True,
                'completed': False,
                'question': f"🤔 {feedback}\n\n{current_question['question']}",
                'question_number': current_q_index + 1,
                'total_questions': len(ONBOARDING_QUESTIONS),
                'question_id': current_question['id'],
                'retry': True,
                'timestamp': get_utc_timestamp()
            })
        
        session['responses'].append({
            'question_id': current_question['id'],
            'answer': answer,
            'emotion': current_question['emotion'],
            'extracted': extracted_data
        })
        
        # ✅ SAVE TO NEW DATABASE STRUCTURE (user_profiles columns)
        try:
            emotion = current_question['emotion']
            music_col = f'{emotion}_music'
            books_col = f'{emotion}_books'
            movies_col = f'{emotion}_movies'
            
            # Extract individual preferences
            music_pref = extracted_data.get('music', '')
            books_pref = extracted_data.get('books', '')
            movies_pref = extracted_data.get('movies', '')
            
            update_data = {
                music_col: music_pref,
                books_col: books_pref,
                movies_col: movies_pref
            }
            
            supabase.table('user_profiles').update(update_data).eq('id', user_id).execute()
            print(f"✅ Saved {emotion} preferences: music={music_pref}, books={books_pref}, movies={movies_pref}")
        except Exception as e:
            print(f"❌ Error saving preferences: {e}")
    
    session['current_question'] += 1
    
    # Check if complete
    if session['current_question'] >= len(ONBOARDING_QUESTIONS):
        session['completed'] = True
        
        try:
            user_language = session.get('language', 'English')
            summary_prompt = f"""Create a warm summary in {user_language} (2-3 sentences):
{chr(10).join([f"- {r.get('emotion', 'preference')}: {r['answer']}" for r in session['responses']])}"""
            
            response = model.generate_content(summary_prompt)
            summary = response.text
        except:
            summary = "Thank you! Your preferences have been saved."
        
        # ✅ MARK USER AS ONBOARDED
        try:
            supabase.table('user_profiles').update({
                'is_onboarded': True
            }).eq('id', user_id).execute()
            print(f"✅ Marked user {user_id} as onboarded")
        except Exception as e:
            print(f"❌ Error marking user as onboarded: {e}")
        
        return jsonify({
            'success': True,
            'completed': True,
            'summary': summary,
            'timestamp': get_utc_timestamp()
        })
    
    next_question = ONBOARDING_QUESTIONS[session['current_question']]
    
    return jsonify({
        'success': True,
        'completed': False,
        'question': next_question['question'],
        'question_number': session['current_question'] + 1,
        'total_questions': len(ONBOARDING_QUESTIONS),
        'question_id': next_question['id'],
        'timestamp': get_utc_timestamp()
    })

# ==================== PREFERENCES ENDPOINTS ====================

@app.route('/api/preferences/get-all', methods=['POST'])
def get_all_preferences():
    """Get all user preferences"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'error': 'User ID required'}), 400
    
    try:
        print(f"\n📋 === GET ALL PREFERENCES ===")
        print(f"User ID: {user_id}")
        
        result = supabase.table('user_profiles')\
            .select('*')\
            .eq('id', user_id)\
            .execute()
        
        print(f"Query result: {result}")
        print(f"Data length: {len(result.data) if result.data else 0}")
        
        if result.data and len(result.data) > 0:
            row = result.data[0]
            
            preferences = {
                'preferred_language': row.get('preferred_language') or 'English',
                'happy_music': row.get('happy_music') or '',
                'happy_books': row.get('happy_books') or '',
                'happy_movies': row.get('happy_movies') or '',
                'sad_music': row.get('sad_music') or '',
                'sad_books': row.get('sad_books') or '',
                'sad_movies': row.get('sad_movies') or '',
                'angry_music': row.get('angry_music') or '',
                'angry_books': row.get('angry_books') or '',
                'angry_movies': row.get('angry_movies') or '',
                'fear_music': row.get('fear_music') or '',
                'fear_books': row.get('fear_books') or '',
                'fear_movies': row.get('fear_movies') or '',
                'neutral_music': row.get('neutral_music') or '',
                'neutral_books': row.get('neutral_books') or '',
                'neutral_movies': row.get('neutral_movies') or ''
            }
            
            print(f"✅ Preferences loaded successfully")
            print(f"Sample: happy_movies={preferences['happy_movies']}")
            
            return jsonify({
                'success': True,
                'preferences': preferences
            })
        else:
            print("⚠️ No user profile found")
            return jsonify({'success': False, 'error': 'User profile not found'}), 404
            
    except Exception as e:
        print(f"❌ Error fetching preferences: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/preferences/update-all', methods=['POST'])
def update_all_preferences():
    """Update all user preferences"""
    data = request.json
    user_id = data.get('user_id')
    preferences = data.get('preferences')
    
    if not user_id or not preferences:
        return jsonify({'success': False, 'error': 'User ID and preferences required'}), 400
    
    try:
        print(f"\n💾 === UPDATE ALL PREFERENCES ===")
        print(f"User ID: {user_id}")
        print(f"Preferences received: {preferences}")
        
        # Validate that preferences is a dictionary
        if not isinstance(preferences, dict):
            return jsonify({'success': False, 'error': 'Invalid preferences format'}), 400
        
        # Clean and prepare update data
        update_data = {}
        
        # Language
        if 'preferred_language' in preferences:
            update_data['preferred_language'] = str(preferences['preferred_language']).strip()
        
        # Emotions
        emotions = ['happy', 'sad', 'angry', 'fear', 'neutral']
        categories = ['music', 'books', 'movies']
        
        for emotion in emotions:
            for category in categories:
                key = f'{emotion}_{category}'
                if key in preferences:
                    value = preferences[key]
                    # Convert to string and strip whitespace, or empty string if None
                    update_data[key] = str(value).strip() if value else ''
        
        print(f"Cleaned update data: {update_data}")
        
        # Perform the update
        result = supabase.table('user_profiles')\
            .update(update_data)\
            .eq('id', user_id)\
            .execute()
        
        print(f"Update result: {result}")
        
        # Verify the update
        verify_result = supabase.table('user_profiles')\
            .select('*')\
            .eq('id', user_id)\
            .execute()
        
        if verify_result.data and len(verify_result.data) > 0:
            print(f"✅ Verification successful")
            print(f"Updated row sample: happy_movies={verify_result.data[0].get('happy_movies')}")
        else:
            print("⚠️ Could not verify update")
        
        return jsonify({
            'success': True,
            'message': 'Preferences updated successfully'
        })
        
    except Exception as e:
        print(f"❌ Error updating preferences: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== EMOTION DETECTION ====================

@app.route('/api/emotion/detect', methods=['POST'])
def detect_emotion():
    """Detect emotion"""
    data = request.json
    image_data = data.get('image')
    user_id = data.get('user_id')
    
    if not image_data:
        return jsonify({'success': False, 'error': 'No image'}), 400
    
    emotion, confidence, success, error = detect_emotion_from_base64(image_data)
    
    if not success:
        return jsonify({'success': False, 'error': error}), 400
    
    if user_id and user_id != 'guest_user':
        try:
            supabase.table('emotion_logs').insert({
                'user_id': user_id,
                'emotion': emotion,
                'confidence': confidence,
                'created_at': get_utc_timestamp()
            }).execute()
        except:
            pass
    
    return jsonify({
        'success': True,
        'emotion': emotion,
        'confidence': confidence
    })


@app.route('/api/emotion/history', methods=['POST'])
def get_emotion_history():
    """Get emotion history"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'success': False}), 400
    
    try:
        result = supabase.table('emotion_logs')\
            .select('*').eq('user_id', user_id)\
            .order('created_at', desc=True).limit(10).execute()
        
        return jsonify({'success': True, 'history': result.data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== RECOMMENDATIONS ====================

@app.route('/api/recommendations/music', methods=['POST'])
def get_music_recommendations():
    """Get music recommendations based on user preferences"""
    data = request.json
    emotion = data.get('emotion', 'neutral')
    user_id = data.get('user_id')
    
    print(f"\n🎵 === MUSIC RECOMMENDATION ===")
    print(f"Emotion: {emotion}, User: {user_id}")
    
    try:
        user_language = get_user_language(user_id) if user_id and user_id != 'guest_user' else 'English'
        print(f"🌍 Language: {user_language}")
        
        lang_config = get_language_config(user_language)
        
        # Get user preferences from new structure
        preferences = get_user_preferences(user_id) if user_id and user_id != 'guest_user' else None
        
        if not preferences:
            print("⚠️ No user preferences found")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please complete onboarding to get personalized recommendations',
                'language': user_language
            })
        
        # Get music preference for this emotion
        music_pref = preferences.get(f'{emotion}_music', '')
        
        if not music_pref:
            print(f"⚠️ No music preference for {emotion}")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please set your preferences in settings',
                'language': user_language
            })
        
        print(f"🎸 User's music preference: {music_pref}")
        
        token = get_spotify_token()
        if not token:
            return jsonify({'success': False, 'error': 'Spotify auth failed'}), 500
        
        headers = {'Authorization': f'Bearer {token}'}
        search_url = 'https://api.spotify.com/v1/search'
        playlists = []
        
        # Search with language + user's genre
        for lang_term in lang_config['music_terms'][:2]:
            search_query = f'{lang_term} {music_pref}'
            print(f"   Query: {search_query}")
            
            params = {
                'q': search_query,
                'type': 'playlist',
                'limit': 5,
                'market': lang_config['market']
            }
            
            response = requests.get(search_url, headers=headers, params=params)
            if response.status_code == 200:
                data_resp = response.json()
                for playlist in data_resp.get('playlists', {}).get('items', []):
                    if playlist and playlist['id'] not in [p['id'] for p in playlists]:
                        playlists.append({
                            'id': playlist['id'],
                            'name': playlist['name'],
                            'description': playlist.get('description', ''),
                            'image': playlist['images'][0]['url'] if playlist.get('images') else None,
                            'url': playlist['external_urls']['spotify'],
                            'tracks': playlist['tracks']['total']
                        })
        
        # Also search with just user's genre
        params = {
            'q': music_pref,
            'type': 'playlist',
            'limit': 5,
            'market': lang_config['market']
        }
        response = requests.get(search_url, headers=headers, params=params)
        if response.status_code == 200:
            data_resp = response.json()
            for playlist in data_resp.get('playlists', {}).get('items', []):
                if playlist and playlist['id'] not in [p['id'] for p in playlists]:
                    playlists.append({
                        'id': playlist['id'],
                        'name': playlist['name'],
                        'description': playlist.get('description', ''),
                        'image': playlist['images'][0]['url'] if playlist.get('images') else None,
                        'url': playlist['external_urls']['spotify'],
                        'tracks': playlist['tracks']['total']
                    })
        
        print(f"\n✅ Found {len(playlists)} playlists")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'recommendations': playlists[:10],
            'language': user_language,
            'timestamp': get_utc_timestamp()
        })
        
    except Exception as e:
        print(f"❌ Music error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recommendations/movies', methods=['POST'])
def get_movie_recommendations():
    """Get movie recommendations"""
    try:
        data = request.json
        emotion = data.get('emotion', 'neutral')
        user_id = data.get('user_id')
        
        print(f"\n🎬 === MOVIE RECOMMENDATION START ===")
        print(f"Emotion: {emotion}, User: {user_id}")
        
        # Step 1: Get user language
        try:
            user_language = get_user_language(user_id) if user_id and user_id != 'guest_user' else 'English'
            print(f"🌍 Language: {user_language}")
        except Exception as e:
            print(f"⚠️ Error getting language, using English: {e}")
            user_language = 'English'
        
        lang_config = get_language_config(user_language)
        tmdb_lang = lang_config['tmdb_lang']
        
        # Step 2: Get user preferences
        try:
            preferences = get_user_preferences(user_id) if user_id and user_id != 'guest_user' else None
            print(f"📋 Preferences retrieved: {preferences is not None}")
        except Exception as e:
            print(f"⚠️ Error getting preferences: {e}")
            preferences = None
        
        if not preferences:
            print("⚠️ No user preferences found - returning empty recommendations")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please complete onboarding to get personalized recommendations',
                'language': user_language,
                'timestamp': get_utc_timestamp()
            })
        
        # Step 3: Get movie preference for this emotion
        movie_pref_key = f'{emotion}_movies'
        movie_pref = preferences.get(movie_pref_key, '')
        print(f"🎭 Looking for key: {movie_pref_key}")
        print(f"🎭 Movie preference value: '{movie_pref}'")
        
        if not movie_pref or movie_pref.strip() == '':
            print(f"⚠️ No movie preference set for {emotion}")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': f'Please set your movie preferences for {emotion} mood in settings',
                'language': user_language,
                'timestamp': get_utc_timestamp()
            })
        
        # Step 4: Fetch TMDB genres
        print(f"\n📡 Fetching TMDB genres...")
        genre_map = {}
        
        try:
            if not TMDB_API_KEY:
                raise Exception("TMDB_API_KEY not configured")
            
            genre_url = f'https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language={tmdb_lang}'
            print(f"🔗 Genre URL: {genre_url}")
            
            genre_response = requests.get(genre_url, timeout=10)
            print(f"📊 Genre API Status: {genre_response.status_code}")
            
            if genre_response.status_code != 200:
                raise Exception(f"TMDB API returned {genre_response.status_code}")
            
            genre_data = genre_response.json()
            print(f"📦 Genre data keys: {genre_data.keys()}")
            
            # Build genre map
            for g in genre_data.get('genres', []):
                if g and isinstance(g, dict) and 'name' in g and 'id' in g:
                    genre_name = str(g['name']).lower().strip()
                    genre_map[genre_name] = int(g['id'])
            
            print(f"📋 Genre map created with {len(genre_map)} genres")
            print(f"📋 Available genres: {list(genre_map.keys())[:10]}...")
            
        except Exception as e:
            print(f"❌ CRITICAL: Error fetching TMDB genres: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Failed to fetch movie genres: {str(e)}',
                'timestamp': get_utc_timestamp()
            }), 500
        
        # Step 5: Map user preference to genre IDs
        print(f"\n🔍 Mapping preference '{movie_pref}' to genre IDs...")
        genre_ids = []
        g_lower = movie_pref.lower().strip()
        
        # Try exact match
        if g_lower in genre_map:
            genre_ids.append(genre_map[g_lower])
            print(f"   ✅ Exact match: {g_lower} -> {genre_map[g_lower]}")
        
        # Try partial match
        if not genre_ids:
            for tmdb_genre, gid in genre_map.items():
                if g_lower in tmdb_genre or tmdb_genre in g_lower:
                    genre_ids.append(gid)
                    print(f"   ✅ Partial match: {g_lower} ~ {tmdb_genre} -> {gid}")
                    break
        
        # Try genre variations
        if not genre_ids:
            genre_variations = {
                'love': 'romance',
                'romantic': 'romance',
                'romcom': 'romance',
                'scary': 'horror',
                'funny': 'comedy',
                'comedies': 'comedy',
                'sad': 'drama',
                'dramas': 'drama',
                'action': 'action',
                'adventure': 'adventure',
                'thriller': 'thriller',
                'suspense': 'thriller',
                'mystery': 'mystery',
                'scifi': 'science fiction',
                'sci-fi': 'science fiction',
                'animated': 'animation',
                'cartoon': 'animation'
            }
            
            mapped_genre = genre_variations.get(g_lower)
            if mapped_genre and mapped_genre in genre_map:
                genre_ids.append(genre_map[mapped_genre])
                print(f"   ✅ Variation match: {g_lower} -> {mapped_genre} -> {genre_map[mapped_genre]}")
        
        # Use emotion defaults if no match
        if not genre_ids:
            emotion_defaults = {
                'happy': [35, 10749],  # Comedy, Romance
                'sad': [18],  # Drama
                'angry': [28, 53],  # Action, Thriller
                'fear': [27, 9648],  # Horror, Mystery
                'neutral': [18, 35],  # Drama, Comedy
                'surprise': [12, 878],  # Adventure, Sci-Fi
                'disgust': [35, 18]  # Comedy, Drama
            }
            genre_ids = emotion_defaults.get(emotion, [18])
            print(f"   ⚠️ No match found, using emotion defaults for {emotion}: {genre_ids}")
        
        print(f"🎬 Final genre IDs: {genre_ids}")
        
        # Step 6: Fetch movies from TMDB
        movies = []
        discover_url = 'https://api.themoviedb.org/3/discover/movie'
        
        # Try user's language first
        print(f"\n🔍 Searching {user_language} movies...")
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': '|'.join(map(str, genre_ids)),
            'with_original_language': tmdb_lang,
            'sort_by': 'popularity.desc',
            'vote_average.gte': 5.0,
            'vote_count.gte': 5,
            'page': 1
        }
        
        try:
            response = requests.get(discover_url, params=params, timeout=10)
            print(f"📊 Movie search status: {response.status_code}")
            
            if response.status_code == 200:
                data_resp = response.json()
                results = data_resp.get('results', [])
                print(f"📦 Found {len(results)} movies in {user_language}")
                
                for movie in results[:15]:
                    if not movie or not isinstance(movie, dict):
                        continue
                    
                    movie_id = movie.get('id')
                    poster_path = movie.get('poster_path')
                    
                    if movie_id and poster_path:
                        overview = movie.get('overview', '')
                        description = (overview[:200] + '...') if len(overview) > 200 else (overview or 'No description')
                        
                        movies.append({
                            'id': int(movie_id),
                            'title': str(movie.get('title', 'Unknown Title')),
                            'description': description,
                            'image': f"https://image.tmdb.org/t/p/w500{poster_path}",
                            'rating': float(movie.get('vote_average', 0)),
                            'release_date': str(movie.get('release_date', 'N/A')),
                            'url': f"https://www.themoviedb.org/movie/{movie_id}"
                        })
                
                print(f"✅ Processed {len(movies)} movies from {user_language}")
        
        except Exception as e:
            print(f"⚠️ Error fetching {user_language} movies: {type(e).__name__}: {e}")
        
        # Add English movies if needed
        if len(movies) < 8:
            print(f"\n🔍 Adding English movies (current count: {len(movies)})...")
            params['with_original_language'] = 'en'
            
            try:
                response = requests.get(discover_url, params=params, timeout=10)
                print(f"📊 English movie search status: {response.status_code}")
                
                if response.status_code == 200:
                    data_resp = response.json()
                    results = data_resp.get('results', [])
                    print(f"📦 Found {len(results)} English movies")
                    
                    existing_ids = {m['id'] for m in movies}
                    
                    for movie in results[:15]:
                        if not movie or not isinstance(movie, dict):
                            continue
                        
                        movie_id = movie.get('id')
                        poster_path = movie.get('poster_path')
                        
                        if movie_id and poster_path and movie_id not in existing_ids:
                            overview = movie.get('overview', '')
                            description = (overview[:200] + '...') if len(overview) > 200 else (overview or 'No description')
                            
                            movies.append({
                                'id': int(movie_id),
                                'title': str(movie.get('title', 'Unknown Title')),
                                'description': description,
                                'image': f"https://image.tmdb.org/t/p/w500{poster_path}",
                                'rating': float(movie.get('vote_average', 0)),
                                'release_date': str(movie.get('release_date', 'N/A')),
                                'url': f"https://www.themoviedb.org/movie/{movie_id}"
                            })
                    
                    print(f"✅ Total movies after adding English: {len(movies)}")
            
            except Exception as e:
                print(f"⚠️ Error fetching English movies: {type(e).__name__}: {e}")
        
        print(f"\n✅ FINAL: Returning {len(movies)} movies")
        print(f"🎬 === MOVIE RECOMMENDATION END ===\n")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'recommendations': movies[:10],
            'language': user_language,
            'timestamp': get_utc_timestamp()
        })
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in get_movie_recommendations:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_type': type(e).__name__,
            'timestamp': get_utc_timestamp()
        }), 500


@app.route('/api/recommendations/books', methods=['POST'])
def get_book_recommendations():
    """Get book recommendations"""
    data = request.json
    emotion = data.get('emotion', 'neutral')
    user_id = data.get('user_id')
    
    print(f"\n📚 === BOOK RECOMMENDATION ===")
    print(f"Emotion: {emotion}, User: {user_id}")
    
    try:
        user_language = get_user_language(user_id) if user_id and user_id != 'guest_user' else 'English'
        print(f"🌍 Language: {user_language}")
        
        lang_config = get_language_config(user_language)
        lang_code = lang_config['lang_code']
        
        # Get user preferences from new structure
        preferences = get_user_preferences(user_id) if user_id and user_id != 'guest_user' else None
        
        if not preferences:
            print("⚠️ No user preferences found")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please complete onboarding',
                'language': user_language,
                'timestamp': get_utc_timestamp()
            })
        
        # Get book preference for this emotion
        book_pref = preferences.get(f'{emotion}_books', '')
        
        if not book_pref:
            print(f"⚠️ No book preference for {emotion}")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please set your preferences in settings',
                'language': user_language
            })
        
        print(f"📖 Book preference: {book_pref}")
        
        books = []
        search_url = 'https://www.googleapis.com/books/v1/volumes'
        
        # Search with language-specific terms
        for lang_term in lang_config['book_terms'][:2]:
            search_query = f'{lang_term} {book_pref}'
            print(f"   Query: {search_query}")
            
            params = {
                'q': search_query,
                'langRestrict': lang_code,
                'orderBy': 'relevance',
                'maxResults': 10
            }
            
            if GOOGLE_BOOKS_API_KEY:
                params['key'] = GOOGLE_BOOKS_API_KEY
            
            response = requests.get(search_url, params=params)
            
            if response.status_code == 200:
                data_resp = response.json()
                for item in data_resp.get('items', []):
                    volume_info = item.get('volumeInfo', {})
                    book_id = item['id']
                    
                    if book_id not in [b['id'] for b in books]:
                        books.append({
                            'id': book_id,
                            'title': volume_info.get('title', 'Unknown'),
                            'authors': volume_info.get('authors', ['Unknown']),
                            'description': (volume_info.get('description', 'No description')[:200] + '...') if volume_info.get('description') else 'No description',
                            'image': volume_info.get('imageLinks', {}).get('thumbnail'),
                            'rating': volume_info.get('averageRating', 0),
                            'categories': volume_info.get('categories', []),
                            'url': volume_info.get('infoLink', '#'),
                            'language': volume_info.get('language', lang_code)
                        })
        
        # Search with just genre
        search_query = f'subject:{book_pref}'
        params = {
            'q': search_query,
            'langRestrict': lang_code,
            'orderBy': 'relevance',
            'maxResults': 10
        }
        
        if GOOGLE_BOOKS_API_KEY:
            params['key'] = GOOGLE_BOOKS_API_KEY
        
        response = requests.get(search_url, params=params)
        
        if response.status_code == 200:
            data_resp = response.json()
            for item in data_resp.get('items', []):
                volume_info = item.get('volumeInfo', {})
                book_id = item['id']
                
                if book_id not in [b['id'] for b in books]:
                    books.append({
                        'id': book_id,
                        'title': volume_info.get('title', 'Unknown'),
                        'authors': volume_info.get('authors', ['Unknown']),
                        'description': (volume_info.get('description', 'No description')[:200] + '...') if volume_info.get('description') else 'No description',
                        'image': volume_info.get('imageLinks', {}).get('thumbnail'),
                        'rating': volume_info.get('averageRating', 0),
                        'categories': volume_info.get('categories', []),
                        'url': volume_info.get('infoLink', '#'),
                        'language': volume_info.get('language', lang_code)
                    })
        
        # Add English books if needed
        if len(books) < 8:
            print(f"\n🔍 Adding English books...")
            params = {
                'q': f'subject:{book_pref}',
                'langRestrict': 'en',
                'orderBy': 'relevance',
                'maxResults': 10
            }
            
            if GOOGLE_BOOKS_API_KEY:
                params['key'] = GOOGLE_BOOKS_API_KEY
            
            response = requests.get(search_url, params=params)
            
            if response.status_code == 200:
                data_resp = response.json()
                for item in data_resp.get('items', []):
                    volume_info = item.get('volumeInfo', {})
                    book_id = item['id']
                    
                    if book_id not in [b['id'] for b in books]:
                        books.append({
                            'id': book_id,
                            'title': volume_info.get('title', 'Unknown'),
                            'authors': volume_info.get('authors', ['Unknown']),
                            'description': (volume_info.get('description', 'No description')[:200] + '...') if volume_info.get('description') else 'No description',
                            'image': volume_info.get('imageLinks', {}).get('thumbnail'),
                            'rating': volume_info.get('averageRating', 0),
                            'categories': volume_info.get('categories', []),
                            'url': volume_info.get('infoLink', '#'),
                            'language': 'en'
                        })
        
        print(f"\n✅ Total books: {len(books)}")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'recommendations': books[:10],
            'language': user_language,
            'timestamp': get_utc_timestamp()
        })
        
    except Exception as e:
        print(f"❌ Book error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'timestamp': get_utc_timestamp()
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'emotion-recommendation-api',
        'emotion_model': emotion_model is not None,
        'apis': {
            'spotify': bool(SPOTIFY_CLIENT_ID),
            'tmdb': bool(TMDB_API_KEY),
            'google_books': bool(GOOGLE_BOOKS_API_KEY)
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 EMOTION RECOMMENDATION API - UPDATED VERSION")
    print("="*60)
    print(f"✅ New database structure (emotion columns)")
    print(f"✅ Settings page support")
    print(f"✅ Profile-based preferences")
    print(f"✅ Multi-language support")
    print("="*60 + "\n")
    app.run(debug=True, port=3000)
