'use client';

import { useState, useEffect, useRef } from 'react';
import { supabase } from '@/lib/supabase';
import { 
  Camera, 
  MessageCircle, 
  X, 
  Send, 
  Smile, 
  Frown, 
  Meh,
  Music,
  Book,
  Film,
  TrendingUp,
  Loader2,
  Video,
  VideoOff
} from 'lucide-react';

interface EmotionData {
  emotion: string;
  confidence: number;
  timestamp: Date;
}

interface ChatMessage {
  role: 'user' | 'bot';
  content: string;
}

export default function DashboardPage() {
  const [userId, setUserId] = useState<string>('');
  const [currentEmotion, setCurrentEmotion] = useState<EmotionData | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [emotionHistory, setEmotionHistory] = useState<EmotionData[]>([]);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    getUserSession();
    return () => {
      stopCamera();
    };
  }, []);

  const getUserSession = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    if (session?.user) {
      setUserId(session.user.id);
      loadEmotionHistory(session.user.id);
    }
  };

  const loadEmotionHistory = async (uid: string) => {
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
          // Add to history
          setEmotionHistory(prev => [newEmotion, ...prev.slice(0, 9)]);
        }
      } catch (error) {
        console.error('Error detecting emotion:', error);
      }
    }, 3000);
  };

  const handleChatSubmit = async () => {
    if (!chatInput.trim() || isChatLoading) return;

    const userMessage: ChatMessage = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    const messageToSend = chatInput;
    setChatInput('');
    setIsChatLoading(true);

    console.log('Sending chat message:', messageToSend);
    console.log('User ID:', userId);

    try {
      const response = await fetch('http://localhost:5000/api/chatbot/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: messageToSend,
          user_id: userId
        }),
      });

      console.log('Response status:', response.status);
      const data = await response.json();
      console.log('Response data:', data);

      if (data.success) {
        setChatMessages(prev => [...prev, { role: 'bot', content: data.response }]);
      } else {
        console.error('Chat error:', data.error);
        setChatMessages(prev => [...prev, { 
          role: 'bot', 
          content: 'Sorry, I encountered an error. Please try again.' 
        }]);
      }
    } catch (error) {
      console.error('Error in chat:', error);
      setChatMessages(prev => [...prev, { 
        role: 'bot', 
        content: 'Sorry, I could not connect to the server. Please check if the backend is running.' 
      }]);
    } finally {
      setIsChatLoading(false);
    }
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">Emotion Dashboard</h1>
          <button
            onClick={() => supabase.auth.signOut()}
            className="px-4 py-2 text-sm text-purple-600 hover:bg-purple-50 rounded-lg transition-all"
          >
            Logout
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Main Content - Left Side */}
          <div className="lg:col-span-3 space-y-6">
            {/* Welcome Section */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-2">
                Welcome to Your Emotion Space
              </h2>
              <p className="text-gray-600">
                Start your camera to detect your current emotion and get personalized recommendations.
              </p>
            </div>

            {/* Recommendations Section */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-6">
                {currentEmotion ? `Recommendations for ${currentEmotion.emotion}` : 'Your Recommendations'}
              </h3>
              
              <div className="space-y-4">
                {/* Music Recommendation */}
                <div className="border-2 border-purple-100 rounded-xl p-6 hover:border-purple-300 transition-all cursor-pointer group">
                  <div className="flex items-start gap-4">
                    <div className="bg-gradient-to-br from-purple-500 to-purple-600 p-4 rounded-xl group-hover:scale-110 transition-all">
                      <Music className="w-8 h-8 text-white" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-lg font-bold text-gray-800 mb-1">Music</h4>
                      <p className="text-gray-600 text-sm mb-2">
                        {currentEmotion 
                          ? `Playlists curated for when you're feeling ${currentEmotion.emotion}` 
                          : 'Start camera to get personalized music recommendations'}
                      </p>
                      {currentEmotion && (
                        <button className="text-purple-600 font-semibold text-sm hover:text-purple-700">
                          View Playlists →
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Books Recommendation */}
                <div className="border-2 border-blue-100 rounded-xl p-6 hover:border-blue-300 transition-all cursor-pointer group">
                  <div className="flex items-start gap-4">
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-4 rounded-xl group-hover:scale-110 transition-all">
                      <Book className="w-8 h-8 text-white" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-lg font-bold text-gray-800 mb-1">Books</h4>
                      <p className="text-gray-600 text-sm mb-2">
                        {currentEmotion 
                          ? `Books that match your ${currentEmotion.emotion} mood` 
                          : 'Start camera to get personalized book recommendations'}
                      </p>
                      {currentEmotion && (
                        <button className="text-blue-600 font-semibold text-sm hover:text-blue-700">
                          Browse Books →
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Movies Recommendation */}
                <div className="border-2 border-pink-100 rounded-xl p-6 hover:border-pink-300 transition-all cursor-pointer group">
                  <div className="flex items-start gap-4">
                    <div className="bg-gradient-to-br from-pink-500 to-pink-600 p-4 rounded-xl group-hover:scale-110 transition-all">
                      <Film className="w-8 h-8 text-white" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-lg font-bold text-gray-800 mb-1">Movies</h4>
                      <p className="text-gray-600 text-sm mb-2">
                        {currentEmotion 
                          ? `Films perfect for your ${currentEmotion.emotion} state` 
                          : 'Start camera to get personalized movie recommendations'}
                      </p>
                      {currentEmotion && (
                        <button className="text-pink-600 font-semibold text-sm hover:text-pink-700">
                          Watch Now →
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
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

              {/* Video Feed - Google Meet Style */}
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

      {/* Floating Chatbot Icon */}
      {!isChatOpen && (
        <button
          onClick={() => setIsChatOpen(true)}
          className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-110 z-50"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      )}

      {/* Chatbot Window */}
      {isChatOpen && (
        <div className="fixed bottom-6 right-6 w-96 h-[500px] bg-white rounded-2xl shadow-2xl flex flex-col z-50">
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-4 rounded-t-2xl flex justify-between items-center">
            <h3 className="text-white font-semibold">AI Assistant</h3>
            <button onClick={() => setIsChatOpen(false)} className="text-white hover:bg-white/20 p-1 rounded">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {chatMessages.length === 0 && (
              <div className="text-center text-gray-500 mt-8">
                <p className="mb-2">👋 Hi! I'm your AI assistant.</p>
                <p className="text-xs">Try asking:</p>
                <div className="text-xs text-left mt-2 space-y-1">
                  <p>• "What are my preferences?"</p>
                  <p>• "Recommend music for happy mood"</p>
                  <p>• "I want to change my sad preferences"</p>
                  <p>• "What should I watch when angry?"</p>
                </div>
              </div>
            )}
            {chatMessages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] p-3 rounded-lg ${
                  msg.role === 'user' ? 'bg-purple-600 text-white' : 'bg-gray-100 text-gray-800'
                }`}>
                  <p className="text-sm">{msg.content}</p>
                </div>
              </div>
            ))}
            {isChatLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 p-3 rounded-lg">
                  <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
                </div>
              </div>
            )}
          </div>

          <div className="p-4 border-t">
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleChatSubmit()}
                placeholder="Type a message..."
                className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <button
                onClick={handleChatSubmit}
                disabled={!chatInput.trim() || isChatLoading}
                className="bg-purple-600 text-white p-2 rounded-lg hover:bg-purple-700 disabled:opacity-50"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}