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

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Load emotion detection model
try:
    emotion_model = load_model('emotion_model_final.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    print("✅ Emotion detection model loaded successfully")
except Exception as e:
    print(f"❌ Error loading emotion model: {e}")
    emotion_model = None

# Updated Onboarding questions - Specific to Music, Books, and Movies
ONBOARDING_QUESTIONS = [
    {
        "id": 1,
        "question": "Hi there! 👋 Let's understand your preferences better.\n\nWhen you're feeling HAPPY, what genre of music do you listen to, what type of books do you read, and what kind of movies do you watch?\n\n(Please mention all three: music genre, book type, and movie genre)",
        "emotion": "happy",
        "category": "all"
    },
    {
        "id": 2,
        "question": "Thanks! Now, when you're feeling SAD or down, what genre of music helps you, what type of books do you read, and what kind of movies do you watch?\n\n(Please mention all three: music genre, book type, and movie genre)",
        "emotion": "sad",
        "category": "all"
    },
    {
        "id": 3,
        "question": "I appreciate you sharing. When you're feeling ANGRY or frustrated, what genre of music do you listen to, what type of books do you read, and what kind of movies do you watch?\n\n(Please mention all three: music genre, book type, and movie genre)",
        "emotion": "angry",
        "category": "all"
    },
    {
        "id": 4,
        "question": "When you're feeling ANXIOUS or FEARFUL, what genre of music calms you, what type of books help you, and what kind of movies do you watch?\n\n(Please mention all three: music genre, book type, and movie genre)",
        "emotion": "fear",
        "category": "all"
    },
    {
        "id": 5,
        "question": "Last question! When you want to feel RELAXED or at peace, what genre of music do you listen to, what type of books do you read, and what kind of movies do you watch?\n\n(Please mention all three: music genre, book type, and movie genre)",
        "emotion": "neutral",
        "category": "all"
    }
]

# Store user sessions
user_sessions = {}

def validate_response_with_gemini(user_answer, emotion):
    """
    Use Gemini to validate if the user's response mentions music, books, and movies
    Returns: (is_valid: bool, feedback_message: str, extracted_data: dict)
    """
    validation_prompt = f"""You are a helpful assistant validating user responses about their media preferences.

The user was asked: "When feeling {emotion.upper()}, what genre of MUSIC do you listen to, what type of BOOKS do you read, and what kind of MOVIES do you watch?"

User's response: "{user_answer}"

Task:
1. Check if the response mentions preferences for ALL THREE categories: Music, Books, and Movies
2. If all three are mentioned, extract them in this format:
   - Music: [genre/type]
   - Books: [genre/type]
   - Movies: [genre/type]
3. If any category is missing or the response is completely irrelevant, indicate which ones are missing

Response format (STRICTLY follow this JSON format):
{{
    "valid": true/false,
    "music": "extracted music genre or null",
    "books": "extracted book type or null",
    "movies": "extracted movie genre or null",
    "missing": ["list of missing categories"],
    "feedback": "brief message to user if invalid"
}}

Examples:
User: "I listen to pop music, read romance novels, and watch comedies"
{{"valid": true, "music": "pop", "books": "romance", "movies": "comedy", "missing": [], "feedback": ""}}

User: "I like pop music and action movies"
{{"valid": false, "music": "pop", "books": null, "movies": "action", "missing": ["books"], "feedback": "Please also mention what type of books you read."}}

User: "I don't know"
{{"valid": false, "music": null, "books": null, "movies": null, "missing": ["music", "books", "movies"], "feedback": "Could you tell me what genres of music, books, and movies you prefer?"}}

Now validate the user's response."""

    try:
        response = model.generate_content(validation_prompt)
        result_text = response.text.strip()
        
        # Extract JSON from response (remove code blocks if present)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Parse the JSON response
        import json
        result = json.loads(result_text)
        
        is_valid = result.get('valid', False)
        extracted_data = {
            'music': result.get('music'),
            'books': result.get('books'),
            'movies': result.get('movies')
        }
        feedback = result.get('feedback', '')
        
        if not is_valid:
            missing = result.get('missing', [])
            if missing:
                missing_str = ", ".join(missing)
                feedback = f"I noticed you didn't mention {missing_str}. Could you please tell me about all three: music genres, book types, and movie genres you prefer when feeling {emotion}?"
        
        return is_valid, feedback, extracted_data
        
    except Exception as e:
        print(f"Validation error: {e}")
        # Fallback validation - check if response has 3 comma-separated items
        answer_lower = user_answer.lower()
        parts = [p.strip() for p in user_answer.split(',')]
        
        # If user gave 3 parts separated by comma, assume valid
        if len(parts) >= 3:
            return True, "", {
                'music': parts[0],
                'books': parts[1],
                'movies': parts[2]
            }
        
        # Otherwise check for keywords
        has_music = any(word in answer_lower for word in ['music', 'song', 'soft', 'loud', 'pop', 'rock', 'jazz', 'classical', 'hip hop', 'rap', 'country', 'edm', 'electronic', 'indie', 'metal', 'blues'])
        has_books = any(word in answer_lower for word in ['book', 'novel', 'fiction', 'non-fiction', 'mystery', 'thriller', 'romance', 'fantasy', 'biography', 'self-help', 'horror', 'poetry', 'drama'])
        has_movies = any(word in answer_lower for word in ['movie', 'film', 'action', 'comedy', 'drama', 'horror', 'thriller', 'romance', 'sci-fi', 'documentary', 'animated', 'love', 'romantic', 'adventure'])
        
        if has_music and has_books and has_movies:
            return True, "", {'music': user_answer, 'books': user_answer, 'movies': user_answer}
        else:
            missing = []
            if not has_music: missing.append("music")
            if not has_books: missing.append("books")
            if not has_movies: missing.append("movies")
            
            feedback = f"Please mention all three: what genre of music you listen to, what type of books you read, and what kind of movies you watch when feeling {emotion}."
            return False, feedback, None


def preprocess_face(face_img, target_size=(48, 48)):
    """Preprocess face image for model prediction"""
    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    face_img = cv2.resize(face_img, target_size)
    
    # Normalize pixel values
    face_img = face_img / 255.0
    
    # Reshape for model input
    face_img = face_img.reshape(1, target_size[0], target_size[1], 1)
    
    return face_img


def detect_emotion_from_base64(base64_image):
    """
    Detect emotion from base64 encoded image
    Returns: (emotion, confidence, success, error_message)
    """
    if emotion_model is None:
        return None, 0.0, False, "Emotion model not loaded"
    
    try:
        # Decode base64 image
        if ',' in base64_image:
            image_data = base64.b64decode(base64_image.split(',')[1])
        else:
            image_data = base64.b64decode(base64_image)
            
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        frame = np.array(image)
        
        # Convert RGB to BGR (OpenCV format)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, 0.0, False, "No face detected"
        
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess face
        processed_face = preprocess_face(face_roi)
        
        # Predict emotion
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
    
    # Initialize session
    user_sessions[user_id] = {
        'current_question': 0,
        'responses': [],
        'completed': False,
        'retry_count': 0
    }
    
    return jsonify({
        'success': True,
        'question': ONBOARDING_QUESTIONS[0]['question'],
        'question_number': 1,
        'total_questions': len(ONBOARDING_QUESTIONS),
        'question_id': ONBOARDING_QUESTIONS[0]['id']
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
            'error': 'Session not found. Please start onboarding first.'
        }), 400
    
    session = user_sessions[user_id]
    current_q_index = session['current_question']
    current_question = ONBOARDING_QUESTIONS[current_q_index]
    
    # Validate the response using Gemini
    is_valid, feedback, extracted_data = validate_response_with_gemini(answer, current_question['emotion'])
    
    if not is_valid:
        # Invalid response - ask again
        session['retry_count'] += 1
        
        return jsonify({
            'success': True,
            'completed': False,
            'question': f"🤔 {feedback}\n\n{current_question['question']}",
            'question_number': current_q_index + 1,
            'total_questions': len(ONBOARDING_QUESTIONS),
            'question_id': current_question['id'],
            'retry': True
        })
    
    # Valid response - store it
    response_data = {
        'question_id': current_question['id'],
        'question': current_question['question'],
        'answer': answer,
        'emotion': current_question['emotion'],
        'extracted_data': extracted_data
    }
    session['responses'].append(response_data)
    session['retry_count'] = 0
    
    # Save to Supabase database
    try:
        supabase.table('onboarding_responses').insert({
            'user_id': user_id,
            'question': current_question['question'],
            'response': answer,
            'emotion': current_question['emotion']
        }).execute()
        
        print(f"✅ Saved response to database for user {user_id}")
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
    
    # Move to next question
    session['current_question'] += 1
    
    # Check if onboarding is complete
    if session['current_question'] >= len(ONBOARDING_QUESTIONS):
        session['completed'] = True
        
        # Generate a personalized summary using Gemini
        try:
            summary_prompt = f"""Based on these user preferences for music, books, and movies for different emotions, create a brief, warm, and encouraging summary (2-3 sentences):

{chr(10).join([f"- When {r['emotion'].title()}: {r['answer']}" for r in session['responses']])}

Keep it personal, positive, and supportive. Acknowledge their diverse tastes."""
            
            response = model.generate_content(summary_prompt)
            summary = response.text
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = "Thank you for sharing your preferences! I've learned about your favorite music, books, and movies for different moods. I'm here to help you discover more content that matches your emotions!"
        
        # Mark user as onboarded in database
        try:
            supabase.table('user_profiles').update({
                'is_onboarded': True
            }).eq('id', user_id).execute()
            print(f"✅ Marked user {user_id} as onboarded")
        except Exception as e:
            print(f"❌ Error updating user profile: {e}")
        
        return jsonify({
            'success': True,
            'completed': True,
            'summary': summary,
            'responses': session['responses']
        })
    
    # Return next question
    next_question = ONBOARDING_QUESTIONS[session['current_question']]
    
    return jsonify({
        'success': True,
        'completed': False,
        'question': next_question['question'],
        'question_number': session['current_question'] + 1,
        'total_questions': len(ONBOARDING_QUESTIONS),
        'question_id': next_question['id']
    })


@app.route('/api/chatbot/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with user preference awareness"""
    data = request.json
    message = data.get('message', '')
    user_id = data.get('user_id')
    
    if not message:
        return jsonify({
            'success': False,
            'error': 'Message is required'
        }), 400
    
    try:
        # Get user's onboarding preferences if available
        user_context = ""
        if user_id:
            try:
                result = supabase.table('onboarding_responses')\
                    .select('emotion, response')\
                    .eq('user_id', user_id)\
                    .execute()
                
                if result.data:
                    preferences = "\n".join([
                        f"- When {r['emotion']}: {r['response']}" 
                        for r in result.data
                    ])
                    user_context = f"\n\nUser's saved preferences:\n{preferences}"
            except Exception as e:
                print(f"Error fetching user preferences: {e}")
        
        # Enhanced prompt with user context
        prompt = f"""You are a supportive AI companion helping users with their media preferences (music, books, movies) based on emotions.{user_context}

User message: {message}

Instructions:
- If the user asks about their preferences, refer to their saved preferences above
- If they want to change a preference, acknowledge it and tell them you've noted it (but mention they can update it in settings)
- Provide personalized recommendations based on their stated preferences
- If they ask for recommendations, suggest specific titles that match their taste
- Be warm, encouraging, and conversational (2-4 sentences)
- If they're asking about a specific emotion, reference what they previously said they like for that emotion

Respond naturally and helpfully:"""
        
        response = model.generate_content(prompt)
        
        return jsonify({
            'success': True,
            'response': response.text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/preferences/update', methods=['POST'])
def update_preference():
    """Update a specific user preference"""
    data = request.json
    user_id = data.get('user_id')
    emotion = data.get('emotion')  # happy, sad, angry, fear, neutral
    new_preference = data.get('preference')  # New preference text
    
    if not all([user_id, emotion, new_preference]):
        return jsonify({
            'success': False,
            'error': 'Missing required fields'
        }), 400
    
    try:
        # Check if preference exists
        existing = supabase.table('onboarding_responses')\
            .select('id')\
            .eq('user_id', user_id)\
            .eq('emotion', emotion)\
            .execute()
        
        if existing.data:
            # Update existing preference
            supabase.table('onboarding_responses')\
                .update({'response': new_preference})\
                .eq('user_id', user_id)\
                .eq('emotion', emotion)\
                .execute()
            message = f"Updated your preferences for {emotion}"
        else:
            # Insert new preference
            supabase.table('onboarding_responses').insert({
                'user_id': user_id,
                'emotion': emotion,
                'response': new_preference,
                'question': f"Preferences for {emotion}"
            }).execute()
            message = f"Added preferences for {emotion}"
        
        print(f"✅ {message} for user {user_id}")
        
        return jsonify({
            'success': True,
            'message': message
        })
    except Exception as e:
        print(f"❌ Error updating preference: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/preferences/get', methods=['POST'])
def get_preferences():
    """Get all user preferences"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({
            'success': False,
            'error': 'User ID required'
        }), 400
    
    try:
        result = supabase.table('onboarding_responses')\
            .select('emotion, response')\
            .eq('user_id', user_id)\
            .execute()
        
        preferences = {r['emotion']: r['response'] for r in result.data}
        
        return jsonify({
            'success': True,
            'preferences': preferences
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== EMOTION DETECTION ENDPOINTS ====================

@app.route('/api/emotion/detect', methods=['POST'])
def detect_emotion():
    """Detect emotion from uploaded image"""
    data = request.json
    image_data = data.get('image')
    user_id = data.get('user_id')
    
    print(f"🔍 Emotion detection request received from user: {user_id}")
    print(f"🔍 Image data length: {len(image_data) if image_data else 0}")
    
    if not image_data:
        print("❌ No image data provided")
        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400
    
    # Detect emotion
    print("🔍 Starting emotion detection...")
    emotion, confidence, success, error = detect_emotion_from_base64(image_data)
    
    print(f"🔍 Detection result: success={success}, emotion={emotion}, confidence={confidence}, error={error}")
    
    if not success:
        print(f"❌ Detection failed: {error}")
        return jsonify({
            'success': False,
            'error': error
        }), 400
    
    # Save emotion log to database
    if user_id:
        try:
            supabase.table('emotion_logs').insert({
                'user_id': user_id,
                'emotion': emotion,
                'confidence': confidence
            }).execute()
            print(f"✅ Logged emotion {emotion} for user {user_id}")
        except Exception as e:
            print(f"❌ Error logging emotion: {e}")
    
    return jsonify({
        'success': True,
        'emotion': emotion,
        'confidence': confidence
    })


@app.route('/api/emotion/history', methods=['POST'])
def get_emotion_history():
    """Get emotion history for a user"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({
            'success': False,
            'error': 'User ID required'
        }), 400
    
    try:
        result = supabase.table('emotion_logs')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('timestamp', desc=True)\
            .limit(10)\
            .execute()
        
        return jsonify({
            'success': True,
            'history': result.data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chatbot/status', methods=['POST'])
def get_status():
    """Get current onboarding status"""
    data = request.json
    user_id = data.get('user_id', 'anonymous')
    
    if user_id not in user_sessions:
        return jsonify({
            'success': True,
            'exists': False
        })
    
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
    """Get all onboarding responses for a user from database"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({
            'success': False,
            'error': 'User ID is required'
        }), 400
    
    try:
        result = supabase.table('onboarding_responses').select('*').eq('user_id', user_id).execute()
        
        return jsonify({
            'success': True,
            'responses': result.data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'chatbot-api',
        'supabase_connected': supabase is not None,
        'emotion_model_loaded': emotion_model is not None
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)