'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Loader2, Sparkles, Globe, Calendar } from 'lucide-react';
import { supabase } from '@/lib/supabase';

interface Message {
  role: 'user' | 'bot';
  content: string;
  timestamp: string;
}

export default function OnboardingPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [onboardingStarted, setOnboardingStarted] = useState(false);
  const [onboardingCompleted, setOnboardingCompleted] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(1);
  const [totalQuestions, setTotalQuestions] = useState(6);
  const [userId, setUserId] = useState<string>('');
  const [currentUtcTime, setCurrentUtcTime] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Use NEXT_PUBLIC_API_URL during development or default to localhost:8000
  const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/+$/,'') + '/api/chatbot';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update UTC time every second
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentUtcTime(now.toISOString());
    };
    
    updateTime();
    const interval = setInterval(updateTime, 1000);
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const getUserId = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.user) {
        setUserId(session.user.id);
        startOnboarding(session.user.id);
      }
    };
    getUserId();
  }, []);

  const startOnboarding = async (uid: string) => {
    try {
      // ✅ FIXED: Correct fetch syntax
      const response = await fetch(`${API_BASE_URL}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: uid,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setOnboardingStarted(true);
        setTotalQuestions(data.total_questions);
        setCurrentQuestion(data.question_number);
        
        setMessages([
          {
            role: 'bot',
            content: data.question,
            timestamp: data.timestamp,
          },
        ]);
      }
    } catch (error) {
      console.error('Error starting onboarding:', error);
      setMessages([
        {
          role: 'bot',
          content: "Sorry, I'm having trouble connecting. Please try again later.",
          timestamp: new Date().toISOString(),
        },
      ]);
    }
  };

  const formatUtcTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      const hours = date.getUTCHours().toString().padStart(2, '0');
      const minutes = date.getUTCMinutes().toString().padStart(2, '0');
      const seconds = date.getUTCSeconds().toString().padStart(2, '0');
      return `${hours}:${minutes}:${seconds} UTC`;
    } catch (error) {
      return '';
    }
  };

  const formatUtcDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      const day = date.getUTCDate().toString().padStart(2, '0');
      const month = (date.getUTCMonth() + 1).toString().padStart(2, '0');
      const year = date.getUTCFullYear();
      return `${day}/${month}/${year}`;
    } catch (error) {
      return '';
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      if (!onboardingCompleted) {
        // ✅ FIXED: Correct fetch syntax
        const response = await fetch(`${API_BASE_URL}/answer`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            user_id: userId,
            answer: inputValue,
          }),
        });

        const data = await response.json();

        if (data.success) {
          if (data.completed) {
            setOnboardingCompleted(true);
            setMessages((prev) => [
              ...prev,
              {
                role: 'bot',
                content: `🎉 ${data.summary}\n\nYour onboarding is complete! Redirecting to dashboard...`,
                timestamp: data.timestamp,
              },
            ]);
            
            setTimeout(() => {
              window.location.href = '/dashboard';
            }, 3000);
          } else {
            setCurrentQuestion(data.question_number);
            setMessages((prev) => [
              ...prev,
              {
                role: 'bot',
                content: data.question,
                timestamp: data.timestamp,
              },
            ]);
          }
        }
      } else {
        // ✅ FIXED: Correct fetch syntax
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: inputValue,
            user_id: userId,
          }),
        });

        const data = await response.json();

        if (data.success) {
          setMessages((prev) => [
            ...prev,
            {
              role: 'bot',
              content: data.response,
              timestamp: data.timestamp,
            },
          ]);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: 'Sorry, something went wrong. Please try again.',
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl h-[90vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className="bg-white/20 p-2 rounded-xl backdrop-blur-sm">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Emotion Companion</h1>
                <p className="text-purple-100 text-sm">Your personal wellbeing assistant</p>
              </div>
            </div>
            
            {/* UTC Time Display */}
            <div className="bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
              <div className="flex items-center gap-2 text-sm">
                <Globe className="w-4 h-4" />
                <div className="text-right">
                  <div className="font-mono font-semibold">{formatUtcTime(currentUtcTime)}</div>
                  <div className="text-xs text-purple-100">{formatUtcDate(currentUtcTime)}</div>
                </div>
              </div>
            </div>
          </div>
          
          {!onboardingCompleted && onboardingStarted && (
            <div className="mt-4 bg-white/10 backdrop-blur-sm rounded-lg p-3">
              <div className="flex justify-between text-sm mb-2">
                <span>Onboarding Progress</span>
                <span className="font-semibold">{currentQuestion}/{totalQuestions}</span>
              </div>
              <div className="w-full bg-white/20 rounded-full h-2">
                <div
                  className="bg-white h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(currentQuestion / totalQuestions) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-gray-50 to-white">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex gap-3 ${
                message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}
            >
              {/* Avatar */}
              <div
                className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-lg ${
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-purple-500 to-pink-500'
                    : 'bg-gradient-to-br from-blue-500 to-purple-500'
                }`}
              >
                {message.role === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-white" />
                )}
              </div>

              {/* Message Bubble */}
              <div
                className={`flex flex-col max-w-[75%] ${
                  message.role === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`rounded-2xl px-5 py-3 shadow-md ${
                    message.role === 'user'
                      ? 'bg-gradient-to-br from-purple-500 to-pink-500 text-white'
                      : 'bg-white text-gray-800 border border-gray-200'
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </p>
                </div>
                <div className="flex items-center gap-1 mt-1 px-2">
                  <Calendar className="w-3 h-3 text-gray-400" />
                  <span className="text-xs text-gray-400">
                    {formatUtcTime(message.timestamp)}
                  </span>
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white rounded-2xl px-5 py-3 border border-gray-200 shadow-md">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-5 h-5 text-gray-500 animate-spin" />
                  <span className="text-sm text-gray-500">Thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4 bg-white">
          <div className="flex gap-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                onboardingCompleted 
                  ? "Onboarding complete! Redirecting..." 
                  : "Type your answer here..."
              }
              disabled={isLoading || onboardingCompleted}
              className="flex-1 px-5 py-3 rounded-xl border-2 border-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed transition-all text-gray-900 placeholder-gray-400"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading || onboardingCompleted}
              className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 py-3 rounded-xl hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2 font-medium shadow-lg hover:shadow-xl"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <div className="flex items-center justify-between mt-2">
            <p className="text-xs text-gray-500">
              Press Enter to send • Shift + Enter for new line
            </p>
            <p className="text-xs text-gray-400 flex items-center gap-1">
              <Globe className="w-3 h-3" />
              All times in UTC
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
