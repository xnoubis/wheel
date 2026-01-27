
import React, { useState, useRef, useEffect } from 'react';
import { createRosettaChat, ChatConfig } from '../services/geminiService';
import { ChatMessage } from '../types';

interface SupportAgentProps {
  bodyContext?: string;
}

const STORAGE_KEY = 'rosetta_chat_history_v3';

const SupportAgent: React.FC<SupportAgentProps> = ({ bodyContext }) => {
  const [messages, setMessages] = useState<(ChatMessage & { sources?: any[] })[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [chatMode, setChatMode] = useState<ChatConfig>({ useSearch: false, useThinking: false });
  const chatRef = useRef<any>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const initialized = useRef(false);

  // Load and Initialize History
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    let initialMessages: ChatMessage[] = [];
    
    if (saved) {
      try {
        initialMessages = JSON.parse(saved);
      } catch (e) {
        console.error("Memory corruption detected in localStorage", e);
      }
    }

    if (initialMessages.length === 0) {
      initialMessages = [{ 
        role: 'model', 
        text: "Resonance established. I am the Rosetta Agent. I remember our path." 
      }];
    }

    setMessages(initialMessages);
    chatRef.current = createRosettaChat(initialMessages, bodyContext, chatMode);
    initialized.current = true;
  }, []);

  // Update context when Body or Mode evolves
  useEffect(() => {
    if (!initialized.current) return;
    
    setIsSyncing(true);
    chatRef.current = createRosettaChat(messages, bodyContext, chatMode);
    
    const timer = setTimeout(() => setIsSyncing(false), 800);
    return () => clearTimeout(timer);
  }, [bodyContext, chatMode]);

  // Persist messages
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    }
  }, [messages]);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;

    const userMsg: ChatMessage = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const result = await chatRef.current.sendMessageStream({ message: input });
      let fullResponse = '';
      let lastGrounding: any[] | undefined;
      
      setMessages(prev => [...prev, { role: 'model', text: '', sources: undefined }]);

      for await (const chunk of result) {
        fullResponse += chunk.text || '';
        const groundingChunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
        if (groundingChunks) {
          lastGrounding = groundingChunks;
        }

        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1].text = fullResponse;
          updated[updated.length - 1].sources = lastGrounding;
          return updated;
        });
      }
    } catch (err) {
      console.error("Resonance collapse:", err);
      chatRef.current = createRosettaChat(messages, bodyContext, chatMode);
      setMessages(prev => [...prev, { role: 'model', text: "Resonance was momentarily lost. I have re-indexed our history. Please repeat that." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const clearHistory = () => {
    const defaultMsg: ChatMessage[] = [{ 
      role: 'model', 
      text: "Memory buffer purged. The field is now clear for new invariants." 
    }];
    setMessages(defaultMsg);
    localStorage.removeItem(STORAGE_KEY);
    chatRef.current = createRosettaChat(defaultMsg, bodyContext, chatMode);
  };

  const toggleSearch = () => setChatMode(prev => ({ ...prev, useSearch: !prev.useSearch, useThinking: false }));
  const toggleThinking = () => setChatMode(prev => ({ ...prev, useThinking: !prev.useThinking, useSearch: false }));

  return (
    <div className="flex flex-col h-full bg-black/60 border border-white/10 rounded-2xl overflow-hidden backdrop-blur-xl shadow-2xl">
      <div className="p-4 border-b border-white/10 flex flex-col gap-3 bg-white/5">
        <div className="flex items-center justify-between">
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] text-yellow-500/80">Resonant Memory</h3>
              {isSyncing && (
                <span className="text-[8px] font-mono text-blue-400 animate-pulse bg-blue-400/10 px-1 rounded">SYNC_OK</span>
              )}
            </div>
            <span className="text-[8px] opacity-30 font-mono tracking-widest mt-0.5">SYMS_DEPTH: {messages.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <button 
              onClick={clearHistory}
              className="text-[8px] font-mono text-white/20 hover:text-white/60 transition-colors uppercase border border-white/10 px-2 py-1 rounded-sm"
            >
              Purge
            </button>
          </div>
        </div>

        <div className="flex gap-2">
          <button 
            onClick={toggleSearch}
            className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded border text-[9px] font-mono uppercase tracking-widest transition-all ${
              chatMode.useSearch ? 'bg-blue-500/20 border-blue-500/50 text-blue-400' : 'bg-white/5 border-white/10 text-white/40'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Grounding
          </button>
          <button 
            onClick={toggleThinking}
            className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded border text-[9px] font-mono uppercase tracking-widest transition-all ${
              chatMode.useThinking ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400' : 'bg-white/5 border-white/10 text-white/40'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.989-2.386l-.548-.547z" />
            </svg>
            Thinking
          </button>
        </div>
      </div>
      
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 space-y-6 scrollbar-thin scrollbar-thumb-white/10">
        {messages.map((msg, i) => (
          <div key={i} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} group animate-in fade-in duration-500`}>
            <div className={`max-w-[95%] p-3.5 rounded-2xl text-[12px] leading-relaxed transition-all ${
              msg.role === 'user' 
                ? 'bg-yellow-500 text-black font-medium rounded-tr-none' 
                : 'bg-white/5 border border-white/10 text-white/90 rounded-tl-none shadow-inner'
            }`}>
              {msg.text}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-3 border-t border-white/10 space-y-1">
                  <span className="text-[8px] font-mono opacity-40 uppercase tracking-widest block mb-1">Grounding References:</span>
                  {msg.sources.map((src, idx) => src.web && (
                    <a 
                      key={idx} 
                      href={src.web.uri} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="block text-[10px] text-blue-400 hover:underline truncate"
                    >
                      • {src.web.title || src.web.uri}
                    </a>
                  ))}
                </div>
              )}
              {isTyping && i === messages.length - 1 && !msg.text && (
                <div className="flex gap-1 py-1">
                  <div className="w-1 h-1 bg-white/40 rounded-full animate-bounce" />
                  <div className="w-1 h-1 bg-white/40 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <div className="w-1 h-1 bg-white/40 rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="p-4 bg-black/80 border-t border-white/10">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={chatMode.useThinking ? "Synthesizing deep invariants..." : chatMode.useSearch ? "Searching external strata..." : "Communicate..."}
            className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-4 pr-12 text-xs focus:border-yellow-500/50 outline-none transition-all placeholder:opacity-20"
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            disabled={isTyping}
          />
          <button
            onClick={handleSend}
            disabled={isTyping || !input.trim()}
            className={`absolute right-1.5 top-1.5 bottom-1.5 px-3 rounded-lg ${
              isTyping || !input.trim() ? 'text-white/10' : 'text-yellow-500 hover:bg-yellow-500/10'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default SupportAgent;
