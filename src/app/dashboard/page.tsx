'use client';

import { useState, useEffect, useRef } from 'react';
import { supabase } from '@/lib/supabase';
import { 
  Camera, 
  X, 
  Smile, 
  Frown, 
  Meh,
  Music,
  Book,
  Film,
  TrendingUp,
  Loader2,
  Video,
  VideoOff,
  ExternalLink,
  Star,
  AlertCircle,
  User,
  Settings,
  LogOut,
  ChevronDown,
  Save,
  ArrowLeft
} from 'lucide-react';

interface EmotionData {
  emotion: string;
  confidence: number;
  timestamp: Date;
}

interface MusicRecommendation {
  id: string;
  name: string;
  description: string;
  image: string | null;
  url: string;
  tracks: number;
}

interface MovieRecommendation {
  id: number;
  title: string;
  description: string;
  image: string | null;
  rating: number;
  release_date: string;
  url: string;
}

interface BookRecommendation {
  id: string;
  title: string;
  authors: string[];
  description: string;
  image: string | null;
  rating: number;
  categories: string[];
  url: string;
}

interface UserPreferences {
  preferred_language: string;
  happy_music: string;
  happy_books: string;
  happy_movies: string;
  sad_music: string;
  sad_books: string;
  sad_movies: string;
  angry_music: string;
  angry_books: string;
  angry_movies: string;
  fear_music: string;
  fear_books: string;
  fear_movies: string;
  neutral_music: string;
  neutral_books: string;
  neutral_movies: string;
}

export default function DashboardPage() {
  const [userId, setUserId] = useState<string>('');
  const [isUserLoaded, setIsUserLoaded] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState<EmotionData | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [emotionHistory, setEmotionHistory] = useState<EmotionData[]>([]);
  const [error, setError] = useState<string>('');
  
  // Profile dropdown and settings state
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [userPreferences, setUserPreferences] = useState<UserPreferences>({
    preferred_language: 'English',
    happy_music: '', happy_books: '', happy_movies: '',
    sad_music: '', sad_books: '', sad_movies: '',
    angry_music: '', angry_books: '', angry_movies: '',
    fear_music: '', fear_books: '', fear_movies: '',
    neutral_music: '', neutral_books: '', neutral_movies: ''
  });
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [settingsMessage, setSettingsMessage] = useState('');
  
  // Recommendations state
  const [musicRecs, setMusicRecs] = useState<MusicRecommendation[]>([]);
  const [movieRecs, setMovieRecs] = useState<MovieRecommendation[]>([]);
  const [bookRecs, setBookRecs] = useState<BookRecommendation[]>([]);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [recsError, setRecsError] = useState<string>('');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const profileMenuRef = useRef<HTMLDivElement>(null);

  // Get user session first
  useEffect(() => {
    getUserSession();
    return () => {
      stopCamera();
    };
  }, []);

  // Load recommendations only after user is loaded
  useEffect(() => {
    if (isUserLoaded && userId) {
      console.log('✅ User loaded, fetching initial recommendations for neutral mood');
      loadRecommendations('neutral');
      if (userId !== 'guest_user') {
        loadUserPreferences();
      }
    }
  }, [isUserLoaded, userId]);

  // Load recommendations when emotion changes
  useEffect(() => {
    if (currentEmotion && userId && isUserLoaded) {
      console.log(`✅ Emotion changed to ${currentEmotion.emotion}, loading recommendations`);
      loadRecommendations(currentEmotion.emotion);
    }
  }, [currentEmotion?.emotion]);

  // Close profile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target as Node)) {
        setShowProfileMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const getUserSession = async () => {
    try {
      console.log('🔍 Getting user session...');
      const { data: { session }, error } = await supabase.auth.getSession();
      
      if (error) {
        console.log('⚠️ No active session, using guest mode');
        setUserId('guest_user');
        setIsUserLoaded(true);
        return;
      }
      
      if (session?.user) {
        console.log('✅ User session found:', session.user.id);
        setUserId(session.user.id);
        setIsUserLoaded(true);
        loadEmotionHistory(session.user.id);
      } else {
        console.log('⚠️ No user session, using guest mode');
        setUserId('guest_user');
        setIsUserLoaded(true);
      }
    } catch (error) {
      console.error('❌ Error getting session:', error);
      setUserId('guest_user');
      setIsUserLoaded(true);
      setError('Could not load user session');
    }
  };

  const loadEmotionHistory = async (uid: string) => {
    if (uid === 'guest_user') {
      console.log('Guest mode - skipping emotion history load');
      return;
    }
    
    try {
      const response = await fetch('http://localhost:5000/api/emotion/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: uid }),
      });
      const data = await response.json();
      if (data.success) {
        setEmotionHistory(data.history.map((h: any) => ({
          emotion: h.emotion,
          confidence: h.confidence,
          timestamp: new Date(h.timestamp)
        })));
      }
    } catch (error) {
      console.error('Error loading emotion history:', error);
    }
  };

  const loadUserPreferences = async () => {
    if (!userId || userId === 'guest_user') return;

    try {
      const response = await fetch('http://localhost:5000/api/preferences/get-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId }),
      });
      
      const data = await response.json();
      if (data.success && data.preferences) {
        setUserPreferences(data.preferences);
      }
    } catch (error) {
      console.error('Error loading preferences:', error);
    }
  };

  const saveUserPreferences = async () => {
    if (!userId || userId === 'guest_user') {
      setSettingsMessage('❌ Please login to save preferences');
      return;
    }

    setIsSavingSettings(true);
    setSettingsMessage('');

    try {
      const response = await fetch('http://localhost:5000/api/preferences/update-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          user_id: userId,
          preferences: userPreferences
        }),
      });
      
      const data = await response.json();
      if (data.success) {
        setSettingsMessage('✅ Preferences saved successfully!');
        // Reload recommendations for current emotion
        if (currentEmotion) {
          loadRecommendations(currentEmotion.emotion);
        }
        setTimeout(() => setSettingsMessage(''), 3000);
      } else {
        setSettingsMessage('❌ Failed to save preferences');
      }
    } catch (error) {
      console.error('Error saving preferences:', error);
      setSettingsMessage('❌ Error saving preferences');
    } finally {
      setIsSavingSettings(false);
    }
  };

  const loadRecommendations = async (emotion: string) => {
    if (!userId) {
      console.log('⚠️ No user ID, skipping recommendations');
      return;
    }

    setLoadingRecs(true);
    setRecsError('');
    console.log(`\n🎯 === LOADING RECOMMENDATIONS ===`);
    console.log(`Emotion: ${emotion}`);
    console.log(`User ID: ${userId}`);
    
    try {
      // Fetch music recommendations
      console.log('🎵 Fetching music...');
      const musicRes = await fetch('http://localhost:5000/api/recommendations/music', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emotion, user_id: userId }),
      });
      
      if (!musicRes.ok) {
        console.error('❌ Music API error:', musicRes.status);
        throw new Error('Music API failed');
      }
      
      const musicData = await musicRes.json();
      console.log('🎵 Music response:', musicData);
      
      if (musicData.success) {
        setMusicRecs(musicData.recommendations || []);
        console.log(`✅ Loaded ${musicData.recommendations?.length || 0} music recommendations`);
      } else {
        console.warn('⚠️ Music API returned success: false');
      }

      // Fetch movie recommendations
      console.log('🎬 Fetching movies...');
      const movieRes = await fetch('http://localhost:5000/api/recommendations/movies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emotion, user_id: userId }),
      });
      
      if (!movieRes.ok) {
        console.error('❌ Movie API error:', movieRes.status);
        throw new Error('Movie API failed');
      }
      
      const movieData = await movieRes.json();
      console.log('🎬 Movie response:', movieData);
      
      if (movieData.success) {
        setMovieRecs(movieData.recommendations || []);
        console.log(`✅ Loaded ${movieData.recommendations?.length || 0} movie recommendations`);
      } else {
        console.warn('⚠️ Movie API returned success: false');
      }

      // Fetch book recommendations
      console.log('📚 Fetching books...');
      const bookRes = await fetch('http://localhost:5000/api/recommendations/books', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emotion, user_id: userId }),
      });
      
      if (!bookRes.ok) {
        console.error('❌ Book API error:', bookRes.status);
        throw new Error('Book API failed');
      }
      
      const bookData = await bookRes.json();
      console.log('📚 Book response:', bookData);
      
      if (bookData.success) {
        setBookRecs(bookData.recommendations || []);
        console.log(`✅ Loaded ${bookData.recommendations?.length || 0} book recommendations`);
      } else {
        console.warn('⚠️ Book API returned success: false');
      }

      console.log('✅ All recommendations loaded successfully');
      
    } catch (error) {
      console.error('❌ Error loading recommendations:', error);
      setRecsError('Could not load recommendations. Please check if the backend is running on http://localhost:5000');
    } finally {
      setLoadingRecs(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsDetecting(true);
        startEmotionDetection();
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera. Please grant permission.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    setIsDetecting(false);
    setCurrentEmotion(null);
  };

  const captureFrame = (): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg');
  };

  const startEmotionDetection = () => {
    detectionIntervalRef.current = setInterval(async () => {
      const frame = captureFrame();
      if (!frame) return;

      try {
        const response = await fetch('http://localhost:5000/api/emotion/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: frame,
            user_id: userId
          }),
        });

        const data = await response.json();
        if (data.success) {
          const newEmotion = {
            emotion: data.emotion,
            confidence: data.confidence,
            timestamp: new Date()
          };
          setCurrentEmotion(newEmotion);
          setEmotionHistory(prev => [newEmotion, ...prev.slice(0, 9)]);
        }
      } catch (error) {
        console.error('Error detecting emotion:', error);
      }
    }, 5000);
  };

  const getEmotionIcon = (emotion: string) => {
    switch (emotion) {
      case 'happy': case 'surprise': return <Smile className="w-8 h-8" />;
      case 'sad': case 'fear': return <Frown className="w-8 h-8" />;
      case 'angry': case 'disgust': return <Meh className="w-8 h-8" />;
      default: return <Meh className="w-8 h-8" />;
    }
  };

  const getEmotionColor = (emotion: string) => {
    switch (emotion) {
      case 'happy': return 'from-green-400 to-green-600';
      case 'sad': return 'from-blue-400 to-blue-600';
      case 'angry': return 'from-red-400 to-red-600';
      case 'fear': return 'from-purple-400 to-purple-600';
      case 'disgust': return 'from-yellow-400 to-yellow-600';
      case 'surprise': return 'from-pink-400 to-pink-600';
      default: return 'from-gray-400 to-gray-600';
    }
  };

  // Show loading state while user is being loaded
  if (!isUserLoaded) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-purple-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Settings Page
  if (showSettings) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
            <button
              onClick={() => setShowSettings(false)}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span className="font-medium">Back to Dashboard</span>
            </button>
            <h1 className="text-2xl font-bold text-gray-800">User Settings</h1>
            <div className="w-32"></div>
          </div>
        </header>

        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Your Preferences</h2>
            
            {/* Language Selection */}
            <div className="mb-8 p-6 bg-purple-50 rounded-xl">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Preferred Language
              </label>
              <select
                value={userPreferences.preferred_language}
                onChange={(e) => setUserPreferences(prev => ({
                  ...prev,
                  preferred_language: e.target.value
                }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="English">English</option>
                <option value="Tamil">Tamil</option>
                <option value="Hindi">Hindi</option>
                <option value="Telugu">Telugu</option>
                <option value="Malayalam">Malayalam</option>
                <option value="Kannada">Kannada</option>
                <option value="Spanish">Spanish</option>
                <option value="French">French</option>
                <option value="Marathi">Marathi</option>
                <option value="Bengali">Bengali</option>
                <option value="Punjabi">Punjabi</option>
              </select>
            </div>

            {/* Emotion Preferences */}
            <div className="space-y-6">
              {['happy', 'sad', 'angry', 'fear', 'neutral'].map(emotion => (
                <div key={emotion} className="border-b pb-6">
                  <h3 className="text-lg font-semibold text-gray-800 capitalize mb-4 flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full bg-gradient-to-br ${getEmotionColor(emotion)}`}></div>
                    When feeling {emotion}
                  </h3>
                  
                  <div className="grid md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">
                        <Music className="w-3 h-3 inline mr-1" />
                        Music Genre
                      </label>
                      <input
                        type="text"
                        value={userPreferences[`${emotion}_music` as keyof UserPreferences] || ''}
                        onChange={(e) => setUserPreferences(prev => ({
                          ...prev,
                          [`${emotion}_music`]: e.target.value
                        }))}
                        placeholder="e.g., pop, rock"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">
                        <Book className="w-3 h-3 inline mr-1" />
                        Book Type
                      </label>
                      <input
                        type="text"
                        value={userPreferences[`${emotion}_books` as keyof UserPreferences] || ''}
                        onChange={(e) => setUserPreferences(prev => ({
                          ...prev,
                          [`${emotion}_books`]: e.target.value
                        }))}
                        placeholder="e.g., fiction, thriller"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">
                        <Film className="w-3 h-3 inline mr-1" />
                        Movie Genre
                      </label>
                      <input
                        type="text"
                        value={userPreferences[`${emotion}_movies` as keyof UserPreferences] || ''}
                        onChange={(e) => setUserPreferences(prev => ({
                          ...prev,
                          [`${emotion}_movies`]: e.target.value
                        }))}
                        placeholder="e.g., comedy, action"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Save Button */}
            <div className="mt-8 flex items-center justify-between">
              {settingsMessage && (
                <p className={`text-sm ${settingsMessage.includes('✅') ? 'text-green-600' : 'text-red-600'}`}>
                  {settingsMessage}
                </p>
              )}
              <button
                onClick={saveUserPreferences}
                disabled={isSavingSettings}
                className="ml-auto bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {isSavingSettings ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5" />
                    Save Changes
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Main Dashboard
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
      {/* Header with Profile Dropdown */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">Emotion Dashboard</h1>
          
          <div className="relative" ref={profileMenuRef}>
            {userId && userId !== 'guest_user' ? (
              <>
                <button
                  onClick={() => setShowProfileMenu(!showProfileMenu)}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-all"
                >
                  <User className="w-5 h-5 text-gray-600" />
                  <span className="text-sm text-gray-700">Profile</span>
                  <ChevronDown className={`w-4 h-4 text-gray-600 transition-transform ${showProfileMenu ? 'rotate-180' : ''}`} />
                </button>

                {showProfileMenu && (
                  <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-200 py-2 z-50">
                    <button
                      onClick={() => {
                        setShowSettings(true);
                        setShowProfileMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-2"
                    >
                      <Settings className="w-4 h-4" />
                      Settings
                    </button>
                    <hr className="my-2 border-gray-200" />
                    <button
                      onClick={() => supabase.auth.signOut()}
                      className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                    >
                      <LogOut className="w-4 h-4" />
                      Logout
                    </button>
                  </div>
                )}
              </>
            ) : (
              <span className="px-4 py-2 text-sm text-gray-600 bg-gray-100 rounded-lg">
                Guest Mode
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Error Display */}
        {(error || recsError) && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-800 font-semibold">Error</p>
              <p className="text-red-700 text-sm">{error || recsError}</p>
              <p className="text-red-600 text-xs mt-1">Make sure the Flask backend is running on http://localhost:5000</p>
            </div>
          </div>
        )}

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Main Content - Left Side */}
          <div className="lg:col-span-3 space-y-6">
            {/* Welcome Section */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-2">
                Welcome to Your Emotion Space
              </h2>
              <p className="text-gray-600">
                {currentEmotion 
                  ? `You're feeling ${currentEmotion.emotion}. Here are personalized recommendations for you!`
                  : 'Start your camera to detect your current emotion and get personalized recommendations.'}
              </p>
              {userId === 'guest_user' && (
                <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-sm text-blue-800">
                    💡 <strong>Guest Mode:</strong> You're using the app without login. 
                    Complete onboarding to save preferences and get personalized suggestions!
                  </p>
                </div>
              )}
            </div>

            {/* Music Recommendations */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="bg-gradient-to-br from-purple-500 to-purple-600 p-3 rounded-xl">
                  <Music className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-800">Music Playlists</h3>
                  <p className="text-sm text-gray-600">
                    {currentEmotion ? `For your ${currentEmotion.emotion} mood` : 'Recommended for you'}
                  </p>
                </div>
              </div>
              
              {loadingRecs ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="w-8 h-8 animate-spin text-purple-600" />
                </div>
              ) : musicRecs.length > 0 ? (
                <div className="grid md:grid-cols-2 gap-4">
                  {musicRecs.slice(0, 4).map((playlist) => (
                    <div key={playlist.id} className="border rounded-lg p-4 hover:shadow-md transition-all group cursor-pointer">
                      <div className="flex gap-3">
                        {playlist.image && (
                          <img src={playlist.image} alt={playlist.name} className="w-16 h-16 rounded-lg object-cover" />
                        )}
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-gray-800 truncate">{playlist.name}</h4>
                          <p className="text-xs text-gray-600 line-clamp-2 mt-1">{playlist.description}</p>
                          <div className="flex items-center justify-between mt-2">
                            <span className="text-xs text-gray-500">{playlist.tracks} tracks</span>
                            <a 
                              href={playlist.url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-purple-600 text-xs flex items-center gap-1 hover:text-purple-700"
                            >
                              Open <ExternalLink className="w-3 h-3" />
                            </a>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-gray-500 mb-2">No music recommendations yet</p>
                  <p className="text-sm text-gray-400">Complete onboarding or start camera for personalized suggestions</p>
                </div>
              )}
            </div>

            {/* Movie Recommendations */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="bg-gradient-to-br from-pink-500 to-pink-600 p-3 rounded-xl">
                  <Film className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-800">Movies</h3>
                  <p className="text-sm text-gray-600">
                    {currentEmotion ? `Perfect for your ${currentEmotion.emotion} state` : 'Recommended for you'}
                  </p>
                </div>
              </div>
              
              {loadingRecs ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="w-8 h-8 animate-spin text-pink-600" />
                </div>
              ) : movieRecs.length > 0 ? (
                <div className="grid md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {movieRecs.slice(0, 8).map((movie) => (
                    <div key={movie.id} className="border rounded-lg overflow-hidden hover:shadow-md transition-all group cursor-pointer">
                      {movie.image && (
                        <img src={movie.image} alt={movie.title} className="w-full h-48 object-cover" />
                      )}
                      <div className="p-3">
                        <h4 className="font-semibold text-sm text-gray-800 truncate" title={movie.title}>{movie.title}</h4>
                        <div className="flex items-center gap-1 mt-1">
                          <Star className="w-3 h-3 text-yellow-500 fill-current" />
                          <span className="text-xs text-gray-600">{movie.rating.toFixed(1)}</span>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">{movie.release_date}</p>
                        <a 
                          href={movie.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-pink-600 text-xs flex items-center gap-1 mt-2 hover:text-pink-700"
                        >
                          More Info <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-gray-500 mb-2">No movie recommendations yet</p>
                  <p className="text-sm text-gray-400">Complete onboarding or start camera for personalized suggestions</p>
                </div>
              )}
            </div>

            {/* Book Recommendations */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-3 rounded-xl">
                  <Book className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-800">Books</h3>
                  <p className="text-sm text-gray-600">
                    {currentEmotion ? `Matches your ${currentEmotion.emotion} mood` : 'Recommended for you'}
                  </p>
                </div>
              </div>
              
              {loadingRecs ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                </div>
              ) : bookRecs.length > 0 ? (
                <div className="grid md:grid-cols-2 gap-4">
                  {bookRecs.slice(0, 6).map((book) => (
                    <div key={book.id} className="border rounded-lg p-4 hover:shadow-md transition-all group cursor-pointer">
                      <div className="flex gap-3">
                        {book.image && (
                          <img src={book.image} alt={book.title} className="w-20 h-28 rounded object-cover flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-sm text-gray-800 line-clamp-2">{book.title}</h4>
                          <p className="text-xs text-gray-600 mt-1">{book.authors.join(', ')}</p>
                          {book.rating > 0 && (
                            <div className="flex items-center gap-1 mt-1">
                              <Star className="w-3 h-3 text-yellow-500 fill-current" />
                              <span className="text-xs text-gray-600">{book.rating.toFixed(1)}</span>
                            </div>
                          )}
                          <a 
                            href={book.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-600 text-xs flex items-center gap-1 mt-2 hover:text-blue-700"
                          >
                            View Book <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-gray-500 mb-2">No book recommendations yet</p>
                  <p className="text-sm text-gray-400">Complete onboarding or start camera for personalized suggestions</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Sidebar - Camera & Detection */}
          <div className="lg:col-span-1 space-y-6">
            {/* Camera Box */}
            <div className="bg-white rounded-2xl shadow-lg p-4 sticky top-4">
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm font-bold text-gray-800 flex items-center gap-2">
                  <Video className="w-4 h-4" />
                  Live Camera
                </h3>
                {!isDetecting ? (
                  <button
                    onClick={startCamera}
                    className="bg-gradient-to-r from-green-500 to-green-600 text-white p-2 rounded-lg hover:from-green-600 hover:to-green-700 transition-all"
                    title="Start Camera"
                  >
                    <Camera className="w-4 h-4" />
                  </button>
                ) : (
                  <button
                    onClick={stopCamera}
                    className="bg-red-500 text-white p-2 rounded-lg hover:bg-red-600 transition-all"
                    title="Stop Camera"
                  >
                    <VideoOff className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Video Feed */}
              <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-[4/3] mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {!isDetecting && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50">
                    <div className="text-center text-white">
                      <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">Camera Off</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Current Emotion Display */}
              {currentEmotion && isDetecting && (
                <div className={`bg-gradient-to-br ${getEmotionColor(currentEmotion.emotion)} rounded-xl p-4 text-white shadow-lg`}>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="bg-white/20 p-3 rounded-full backdrop-blur-sm">
                      {getEmotionIcon(currentEmotion.emotion)}
                    </div>
                    <div>
                      <p className="text-xs opacity-90">Current Emotion</p>
                      <p className="text-xl font-bold capitalize">{currentEmotion.emotion}</p>
                    </div>
                  </div>
                  <div className="bg-white/20 rounded-lg p-2 backdrop-blur-sm">
                    <p className="text-xs mb-1">Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-white/20 rounded-full h-2">
                        <div 
                          className="bg-white h-2 rounded-full transition-all duration-500"
                          style={{ width: `${currentEmotion.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-bold">{(currentEmotion.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              )}

              {!currentEmotion && isDetecting && (
                <div className="bg-gray-100 rounded-xl p-4 text-center">
                  <Loader2 className="w-8 h-8 mx-auto mb-2 text-purple-600 animate-spin" />
                  <p className="text-sm text-gray-600">Detecting emotion...</p>
                </div>
              )}
            </div>

            {/* Emotion History */}
            <div className="bg-white rounded-2xl shadow-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-purple-600" />
                <h3 className="text-sm font-bold text-gray-800">Recent Emotions</h3>
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {emotionHistory.length === 0 && (
                  <p className="text-xs text-gray-500 text-center py-4">No emotions detected yet</p>
                )}
                {emotionHistory.map((emotion, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className={`bg-gradient-to-br ${getEmotionColor(emotion.emotion)} p-1.5 rounded-full text-white`}>
                        {getEmotionIcon(emotion.emotion)}
                      </div>
                      <p className="text-xs font-semibold capitalize">{emotion.emotion}</p>
                    </div>
                    <span className="text-xs text-gray-600">
                      {(emotion.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
