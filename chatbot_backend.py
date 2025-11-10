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

# ✅ FIXED: Use correct model name
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

# Updated Onboarding questions with clearer format
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
    """
    Get language-specific configuration for searching native content
    """
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
        result = supabase.table('onboarding_responses')\
            .select('language')\
            .eq('user_id', user_id)\
            .not_.is_('language', 'null')\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0 and result.data[0].get('language'):
            lang = result.data[0]['language']
            print(f"🌍 Found language: {lang}")
            return lang
        
        profile_result = supabase.table('user_profiles')\
            .select('preferred_language')\
            .eq('id', user_id)\
            .execute()
        
        if profile_result.data and len(profile_result.data) > 0:
            lang = profile_result.data[0].get('preferred_language', 'English')
            print(f"🌍 Found language in profile: {lang}")
            return lang
        
        return 'English'
    except Exception as e:
        print(f"❌ Error fetching user language: {e}")
        return 'English'


def get_user_preferences_for_emotion(user_id, emotion):
    """
    Get user's EXACT preferences from onboarding_responses for a specific emotion
    Returns the raw response text from the user
    """
    try:
        result = supabase.table('onboarding_responses')\
            .select('response')\
            .eq('user_id', user_id)\
            .eq('emotion', emotion)\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0:
            user_response = result.data[0]['response']
            print(f"📝 Found user preference for {emotion}: {user_response}")
            return user_response
        
        print(f"⚠️ No preferences found for {emotion}")
        return None
    except Exception as e:
        print(f"❌ Error fetching preferences: {e}")
        return None


def extract_genres_from_user_response(user_response):
    """
    Extract EXACT genres from user's response - IMPROVED with fallback parsing
    """
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
"soft, fiction, romantic" -> {{"music": ["soft"], "books": ["fiction"], "movies": ["romantic"]}}
"motivating, comic, love" -> {{"music": ["motivating"], "books": ["comic"], "movies": ["love"]}}

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
        
        # FALLBACK: Simple comma-separated parsing
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
                # Single item - use for all
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
    """Validate if response contains music, books, and movies - IMPROVED VERSION"""
    validation_prompt = f"""Validate this response for {emotion} preferences: "{user_answer}"

The user should mention their preferences for Music, Books, and Movies.

IMPORTANT: Be VERY lenient with the format. Accept responses like:
- "rock, thriller, action" (assume order: music, books, movies)
- "I like rock music, thriller books, action movies"
- "rock music, thriller novels, action films"
- Simple comma-separated: "love, fiction, action"
- "pop, romance, comedy"
- "soft, fiction, romantic"

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
        
        # Clean JSON
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
        
        # Additional check: if we have all three items, it's valid
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
        # Fallback: try simple parsing for comma-separated values
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
            
            supabase.table('onboarding_responses').insert({
                'user_id': user_id,
                'question': current_question['question'],
                'response': answer,
                'emotion': None,
                'language': language,
                'created_at': get_utc_timestamp()
            }).execute()
            
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
            'emotion': current_question['emotion']
        })
        
        # Save to database
        try:
            supabase.table('onboarding_responses').insert({
                'user_id': user_id,
                'question': current_question['question'],
                'response': answer,
                'emotion': current_question['emotion'],
                'created_at': get_utc_timestamp()
            }).execute()
            print(f"✅ Saved {current_question['emotion']} preferences for user {user_id}")
        except Exception as e:
            print(f"❌ Error saving: {e}")
    
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
        
        try:
            supabase.table('user_profiles').update({
                'is_onboarded': True
            }).eq('id', user_id).execute()
        except:
            pass
        
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


@app.route('/api/chatbot/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    data = request.json
    message = data.get('message', '')
    user_id = data.get('user_id')
    
    if not message:
        return jsonify({'success': False, 'error': 'Message required'}), 400
    
    try:
        user_language = get_user_language(user_id) if user_id and user_id != 'guest_user' else 'English'
        
        prompt = f"""You are a supportive AI companion. Respond in {user_language}.
User: {message}
Be warm and helpful (2-4 sentences):"""
        
        response = model.generate_content(prompt)
        
        return jsonify({
            'success': True,
            'response': response.text,
            'timestamp': get_utc_timestamp()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/preferences/update', methods=['POST'])
def update_preference():
    """Update preference"""
    data = request.json
    user_id = data.get('user_id')
    emotion = data.get('emotion')
    new_preference = data.get('preference')
    
    if not all([user_id, emotion, new_preference]):
        return jsonify({'success': False, 'error': 'Missing fields'}), 400
    
    try:
        existing = supabase.table('onboarding_responses')\
            .select('id').eq('user_id', user_id).eq('emotion', emotion).execute()
        
        if existing.data:
            supabase.table('onboarding_responses')\
                .update({'response': new_preference})\
                .eq('user_id', user_id).eq('emotion', emotion).execute()
        else:
            supabase.table('onboarding_responses').insert({
                'user_id': user_id,
                'emotion': emotion,
                'response': new_preference,
                'created_at': get_utc_timestamp()
            }).execute()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/preferences/get', methods=['POST'])
def get_preferences():
    """Get preferences"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'success': False}), 400
    
    try:
        user_language = get_user_language(user_id)
        result = supabase.table('onboarding_responses')\
            .select('emotion, response').eq('user_id', user_id).execute()
        
        preferences = {r['emotion']: r['response'] for r in result.data if r.get('emotion')}
        
        return jsonify({
            'success': True,
            'language': user_language,
            'preferences': preferences
        })
    except Exception as e:
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


@app.route('/api/chatbot/status', methods=['POST'])
def get_status():
    """Get onboarding status"""
    data = request.json
    user_id = data.get('user_id', 'anonymous')
    
    if user_id not in user_sessions:
        return jsonify({'success': True, 'exists': False})
    
    session = user_sessions[user_id]
    
    return jsonify({
        'success': True,
        'exists': True,
        'completed': session['completed'],
        'current_question': session['current_question'] + 1,
        'total_questions': len(ONBOARDING_QUESTIONS)
    })


@app.route('/api/chatbot/responses', methods=['POST'])
def get_user_responses():
    """Get responses"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'success': False}), 400
    
    try:
        result = supabase.table('onboarding_responses').select('*').eq('user_id', user_id).execute()
        return jsonify({'success': True, 'responses': result.data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== RECOMMENDATIONS USING USER'S EXACT PREFERENCES ====================

@app.route('/api/recommendations/music', methods=['POST'])
def get_music_recommendations():
    """Get music recommendations based on USER'S EXACT PREFERENCES from database"""
    data = request.json
    emotion = data.get('emotion', 'neutral')
    user_id = data.get('user_id')
    
    print(f"\n🎵 === MUSIC RECOMMENDATION (USER PREFERENCES) ===")
    print(f"Emotion: {emotion}, User: {user_id}")
    
    try:
        # Get user language
        user_language = get_user_language(user_id) if user_id and user_id != 'guest_user' else 'English'
        print(f"🌍 Language: {user_language}")
        
        # Get language config
        lang_config = get_language_config(user_language)
        
        # Get user's EXACT preference from database
        user_preference_text = get_user_preferences_for_emotion(user_id, emotion) if user_id and user_id != 'guest_user' else None
        
        if not user_preference_text:
            print("⚠️ No user preferences found, using defaults")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please complete onboarding to get personalized recommendations',
                'language': user_language
            })
        
        # Extract genres from user's response
        extracted = extract_genres_from_user_response(user_preference_text)
        if not extracted or not extracted.get('music'):
            print("❌ Could not extract music genres, using user text directly")
            # Fallback: use the first word from user text
            user_music_genres = [user_preference_text.split(',')[0].strip()]
        else:
            user_music_genres = extracted['music']
        
        print(f"🎸 User's music genres: {user_music_genres}")
        
        token = get_spotify_token()
        if not token:
            return jsonify({'success': False, 'error': 'Spotify auth failed'}), 500
        
        headers = {'Authorization': f'Bearer {token}'}
        search_url = 'https://api.spotify.com/v1/search'
        playlists = []
        
        # Search with LANGUAGE + USER'S GENRES
        print(f"\n🔍 Searching: {user_language} + {user_music_genres}")
        
        # For non-English languages, search with language-specific terms
        if lang_config['music_terms']:
            for lang_term in lang_config['music_terms'][:2]:
                for user_genre in user_music_genres[:2]:
                    search_query = f'{lang_term} {user_genre}'
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
                                print(f"      ✅ {playlist['name']}")
        
        # Also search with just user's genres (for English or additional results)
        for user_genre in user_music_genres:
            params = {
                'q': user_genre,
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
            'genres': user_music_genres,
            'user_preference': user_preference_text,
            'recommendations': playlists[:10],
            'language': user_language,
            'timestamp': get_utc_timestamp()
        })
        
    except Exception as e:
        print(f"❌ Music error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== REPLACE ONLY THESE TWO FUNCTIONS IN YOUR app.py ====================

@app.route('/api/recommendations/movies', methods=['POST'])
def get_movie_recommendations():
    """Get movie recommendations - FIXED NULL HANDLING"""
    data = request.json
    emotion = data.get('emotion', 'neutral')
    user_id = data.get('user_id')
    
    print(f"\n🎬 === MOVIE RECOMMENDATION ===")
    print(f"Emotion: {emotion}, User: {user_id}")
    
    try:
        # Get user language
        user_language = get_user_language(user_id) if user_id and user_id != 'guest_user' else 'English'
        print(f"🌍 Language: {user_language}")
        
        # Get language config
        lang_config = get_language_config(user_language)
        tmdb_lang = lang_config['tmdb_lang']
        
        # Get user's EXACT preference from database
        user_preference_text = get_user_preferences_for_emotion(user_id, emotion) if user_id and user_id != 'guest_user' else None
        
        if not user_preference_text:
            print("⚠️ No user preferences found")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please complete onboarding',
                'language': user_language,
                'timestamp': get_utc_timestamp()
            })
        
        # Extract genres from user's response (same as music)
        extracted = extract_genres_from_user_response(user_preference_text)
        if not extracted or not extracted.get('movies'):
            print("❌ Could not extract movie genres")
            # Fallback: try to get third item from comma-separated
            parts = user_preference_text.split(',')
            user_movie_genres = [parts[2].strip()] if len(parts) >= 3 else ['drama']
        else:
            user_movie_genres = extracted['movies']
        
        print(f"🎭 Movie genres: {user_movie_genres}")
        
        # Get TMDB genre IDs
        genre_url = f'https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language={tmdb_lang}'
        genre_response = requests.get(genre_url)
        
        if genre_response.status_code != 200:
            print(f"❌ TMDB genre API failed: {genre_response.status_code}")
            return jsonify({
                'success': False, 
                'error': 'TMDB API error',
                'timestamp': get_utc_timestamp()
            }), 500
        
        genre_data = genre_response.json()
        
        # ✅ FIX: Filter out None values before creating the genre map
        genre_map = {}
        for g in genre_data.get('genres', []):
            if g and g.get('name') and g.get('id'):  # Ensure both name and id exist
                genre_name = g['name'].lower()
                genre_map[genre_name] = g['id']
        
        print(f"📋 Available TMDB genres: {list(genre_map.keys())}")
        
        # Map user's genres to TMDB IDs (improved matching)
        genre_ids = []
        for user_genre in user_movie_genres:
            g_lower = user_genre.lower().strip()
            print(f"   Trying to match: '{g_lower}'")
            
            # Try exact match
            if g_lower in genre_map:
                genre_ids.append(genre_map[g_lower])
                print(f"   ✅ Exact match: {g_lower} -> {genre_map[g_lower]}")
            else:
                # Try partial match
                matched = False
                for tmdb_genre, gid in genre_map.items():
                    if g_lower in tmdb_genre or tmdb_genre in g_lower:
                        genre_ids.append(gid)
                        print(f"   ✅ Partial match: {g_lower} ~ {tmdb_genre} -> {gid}")
                        matched = True
                        break
                
                if not matched:
                    # Special handling for common genre variations
                    genre_variations = {
                        'love': 'romance',
                        'romantic': 'romance',
                        'scary': 'horror',
                        'funny': 'comedy',
                        'sad': 'drama',
                        'action': 'action',
                        'adventure': 'adventure',
                        'thriller': 'thriller',
                        'crime': 'crime',
                        'mystery': 'mystery',
                        'scifi': 'science fiction',
                        'sci-fi': 'science fiction',
                        'fantasy': 'fantasy',
                        'animation': 'animation',
                        'family': 'family',
                        'war': 'war',
                        'western': 'western',
                        'history': 'history',
                        'historical': 'history',
                        'music': 'music',
                        'documentary': 'documentary'
                    }
                    
                    mapped_genre = genre_variations.get(g_lower)
                    if mapped_genre and mapped_genre in genre_map:
                        genre_ids.append(genre_map[mapped_genre])
                        print(f"   ✅ Variation match: {g_lower} -> {mapped_genre} -> {genre_map[mapped_genre]}")
                    else:
                        print(f"   ⚠️ No match found for: {g_lower}")
        
        # If no matches, use popular genres based on emotion
        if not genre_ids:
            emotion_genre_defaults = {
                'happy': [35, 10749],  # Comedy, Romance
                'sad': [18],  # Drama
                'angry': [28, 53],  # Action, Thriller
                'fear': [27, 9648],  # Horror, Mystery
                'neutral': [18, 35],  # Drama, Comedy
                'surprise': [12, 14],  # Adventure, Fantasy
                'disgust': [80]  # Crime
            }
            genre_ids = emotion_genre_defaults.get(emotion, [18])  # Default to Drama
            print(f"   ⚠️ Using emotion-based defaults: {genre_ids}")
        
        print(f"🎬 Final genre IDs: {genre_ids}")
        
        movies = []
        discover_url = 'https://api.themoviedb.org/3/discover/movie'
        
        # Search in user's language first
        print(f"\n🔍 Searching {user_language} movies...")
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': '|'.join(map(str, genre_ids)),
            'with_original_language': tmdb_lang,
            'sort_by': 'popularity.desc',
            'vote_average.gte': 6.0,
            'vote_count.gte': 10,
            'page': 1
        }
        
        response = requests.get(discover_url, params=params)
        print(f"   TMDB API status: {response.status_code}")
        
        if response.status_code == 200:
            data_resp = response.json()
            results = data_resp.get('results', [])
            print(f"   Found {len(results)} results")
            
            for movie in results[:15]:
                if movie.get('poster_path'):  # Only add if has poster
                    movies.append({
                        'id': movie['id'],
                        'title': movie['title'],
                        'description': movie.get('overview', 'No description')[:200] + ('...' if len(movie.get('overview', '')) > 200 else ''),
                        'image': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                        'rating': movie.get('vote_average', 0),
                        'release_date': movie.get('release_date', 'N/A'),
                        'url': f"https://www.themoviedb.org/movie/{movie['id']}"
                    })
                    print(f"      ✅ {movie['title']}")
        
        # Add English movies if needed
        if len(movies) < 8:
            print(f"\n🔍 Adding English movies (currently have {len(movies)})...")
            params['with_original_language'] = 'en'
            response = requests.get(discover_url, params=params)
            
            if response.status_code == 200:
                data_resp = response.json()
                for movie in data_resp.get('results', [])[:15]:
                    if movie['id'] not in [m['id'] for m in movies] and movie.get('poster_path'):
                        movies.append({
                            'id': movie['id'],
                            'title': movie['title'],
                            'description': movie.get('overview', 'No description')[:200] + ('...' if len(movie.get('overview', '')) > 200 else ''),
                            'image': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                            'rating': movie.get('vote_average', 0),
                            'release_date': movie.get('release_date', 'N/A'),
                            'url': f"https://www.themoviedb.org/movie/{movie['id']}"
                        })
        
        print(f"\n✅ Total movies found: {len(movies)}")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'genres': user_movie_genres,
            'user_preference': user_preference_text,
            'recommendations': movies[:10],
            'language': user_language,
            'timestamp': get_utc_timestamp()
        })
        
    except Exception as e:
        print(f"❌ Movie error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'timestamp': get_utc_timestamp()
        }), 500


@app.route('/api/recommendations/books', methods=['POST'])
def get_book_recommendations():
    """Get book recommendations - FIXED TO MATCH MUSIC ENDPOINT"""
    data = request.json
    emotion = data.get('emotion', 'neutral')
    user_id = data.get('user_id')
    
    print(f"\n📚 === BOOK RECOMMENDATION ===")
    print(f"Emotion: {emotion}, User: {user_id}")
    
    try:
        # Get user language
        user_language = get_user_language(user_id) if user_id and user_id != 'guest_user' else 'English'
        print(f"🌍 Language: {user_language}")
        
        # Get language config
        lang_config = get_language_config(user_language)
        lang_code = lang_config['lang_code']
        
        # Get user's EXACT preference from database
        user_preference_text = get_user_preferences_for_emotion(user_id, emotion) if user_id and user_id != 'guest_user' else None
        
        if not user_preference_text:
            print("⚠️ No user preferences found")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'recommendations': [],
                'message': 'Please complete onboarding',
                'language': user_language,
                'timestamp': get_utc_timestamp()
            })
        
        # Extract genres from user's response (same as music)
        extracted = extract_genres_from_user_response(user_preference_text)
        if not extracted or not extracted.get('books'):
            print("❌ Could not extract book genres")
            # Fallback: try to get second item from comma-separated
            parts = user_preference_text.split(',')
            user_book_genres = [parts[1].strip()] if len(parts) >= 2 else ['fiction']
        else:
            user_book_genres = extracted['books']
        
        print(f"📖 Book genres: {user_book_genres}")
        
        books = []
        search_url = 'https://www.googleapis.com/books/v1/volumes'
        
        # Search with language + genres (like music does)
        print(f"\n🔍 Searching {user_language} books...")
        
        for user_genre in user_book_genres[:3]:
            # Search with language-specific terms
            if lang_config['book_terms']:
                for lang_term in lang_config['book_terms'][:2]:
                    search_query = f'{lang_term} {user_genre}'
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
                                print(f"      ✅ {volume_info.get('title', 'Unknown')}")
            
            # Also search with just the genre (like music does)
            search_query = f'subject:{user_genre}'
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
        
        # Add English books if needed (like music does)
        if len(books) < 8:
            print(f"\n🔍 Adding English books (currently have {len(books)})...")
            for user_genre in user_book_genres[:2]:
                params = {
                    'q': f'subject:{user_genre}',
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
        
        print(f"\n✅ Total books found: {len(books)}")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'genres': user_book_genres,
            'user_preference': user_preference_text,
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
    print("🚀 EMOTION RECOMMENDATION API - FIXED VERSION")
    print("="*60)
    print(f"✅ FIXED: Gemini model name (gemini-1.5-flash-latest)")
    print(f"✅ Uses USER'S EXACT preferences from database")
    print(f"✅ Supports: Tamil, Hindi, Telugu, Malayalam, Kannada, Marathi, Bengali, Punjabi, Spanish, French")
    print(f"✅ Multi-language support based on stored user preferences")
    print(f"✅ Fallback parsing if Gemini extraction fails")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
