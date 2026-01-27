
import React, { useState } from 'react';
import { translatePhenomenology } from '../services/geminiService';
import { RosettaMapping } from '../types';

interface RosettaTranslatorProps {
  learnedMappings: RosettaMapping[];
  onLearn: (mapping: RosettaMapping) => void;
}

const RosettaTranslator: React.FC<RosettaTranslatorProps> = ({ learnedMappings, onLearn }) => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [hasLearnedCurrent, setHasLearnedCurrent] = useState(false);

  const handleTranslate = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setHasLearnedCurrent(false);
    
    const contextStr = learnedMappings.map(m => `${m.human} -> ${m.model}`).join('; ');
    
    try {
      const data = await translatePhenomenology(input, contextStr);
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleLearn = () => {
    if (result) {
      onLearn({
        human: result.humanTalk,
        model: result.modelTalk,
        invariant: result.invariant,
        isLearned: true
      });
      setHasLearnedCurrent(true);
    }
  };

  return (
    <div className="w-full space-y-6">
      <div className="relative">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Speak phenomenologically... (e.g. 'It just clicked')"
          className="w-full bg-black border border-white/20 rounded-lg p-4 pr-16 text-sm focus:border-yellow-500 outline-none transition-colors font-light tracking-wide"
          onKeyDown={(e) => e.key === 'Enter' && handleTranslate()}
        />
        <button
          onClick={handleTranslate}
          disabled={loading}
          className="absolute right-2 top-2 bottom-2 px-4 bg-yellow-500 text-black text-xs font-bold rounded-md hover:bg-yellow-400 transition-colors disabled:opacity-50"
        >
          {loading ? '...' : 'ENGAGE'}
        </button>
      </div>

      {result && (
        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 bg-white/5 border border-white/10 rounded-lg relative group">
              <span className="text-[10px] uppercase font-mono opacity-40 block mb-1">Human Talk</span>
              <p className="text-sm italic">"{result.humanTalk}"</p>
            </div>
            <div className="p-3 bg-white/5 border border-white/10 rounded-lg">
              <span className="text-[10px] uppercase font-mono opacity-40 block mb-1">Model Talk</span>
              <p className="text-sm font-mono text-blue-400">{result.modelTalk}</p>
            </div>
            <div className="p-3 bg-white/5 border border-white/10 rounded-lg">
              <span className="text-[10px] uppercase font-mono opacity-40 block mb-1">Structural Invariant</span>
              <p className="text-sm font-bold text-yellow-500">{result.invariant}</p>
            </div>
          </div>
          
          <div className="flex justify-end">
            <button
              onClick={handleLearn}
              disabled={hasLearnedCurrent}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-[10px] font-bold tracking-widest uppercase transition-all
                ${hasLearnedCurrent 
                  ? 'bg-green-500/20 text-green-500 border border-green-500/50' 
                  : 'bg-white/10 text-white hover:bg-white/20 border border-white/20'}`}
            >
              {hasLearnedCurrent ? (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Mapping Stored
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                  </svg>
                  Confirm & Learn
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {learnedMappings.length > 0 && (
        <div className="pt-4 border-t border-white/5">
          <h4 className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-40 mb-3">Synaptic Memory ({learnedMappings.length})</h4>
          <div className="flex flex-wrap gap-2">
            {learnedMappings.map((m, i) => (
              <div key={i} className="px-3 py-1 bg-yellow-500/10 border border-yellow-500/30 rounded-full text-[10px] font-mono text-yellow-500/80 animate-in zoom-in-95">
                {m.human} ➔ {m.invariant}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RosettaTranslator;
