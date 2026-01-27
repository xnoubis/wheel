
import React, { useState, useCallback, useEffect } from 'react';
import SpeciesWheel from './components/SpeciesWheel';
import FlickerClock from './components/FlickerClock';
import TopologicalMap from './components/TopologicalMap';
import RosettaTranslator from './components/RosettaTranslator';
import SupportAgent from './components/SupportAgent';
import { SPECIES_WHEEL_CONFIG } from './constants';
import { generateBodySpoke } from './services/geminiService';
import { RosettaMapping, SlimeNode } from './types';

const SESSION_STORAGE_KEY = 'rosetta_engine_session_v3';

const App: React.FC = () => {
  const [activeIndex, setActiveIndex] = useState(0);
  const [flickerRate, setFlickerRate] = useState(60);
  const [isRotating, setIsRotating] = useState(false);
  const [bodyText, setBodyText] = useState("A central core of potentiality.");
  const [history, setHistory] = useState<string[]>([]);
  const [isBuilding, setIsBuilding] = useState(false);
  const [learnedMappings, setLearnedMappings] = useState<RosettaMapping[]>([]);
  const [slimeTrails, setSlimeTrails] = useState<SlimeNode[]>([]);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');

  const activeSpecies = SPECIES_WHEEL_CONFIG[activeIndex];

  // Load session from architectural strata on mount
  useEffect(() => {
    const saved = localStorage.getItem(SESSION_STORAGE_KEY);
    if (saved) {
      try {
        const data = JSON.parse(saved);
        if (data.activeIndex !== undefined) setActiveIndex(data.activeIndex);
        if (data.flickerRate !== undefined) setFlickerRate(data.flickerRate);
        if (data.bodyText !== undefined) setBodyText(data.bodyText);
        if (data.history !== undefined) setHistory(data.history);
        if (data.learnedMappings !== undefined) setLearnedMappings(data.learnedMappings);
        if (data.slimeTrails !== undefined) setSlimeTrails(data.slimeTrails);
      } catch (e) {
        console.error("Memory corruption in session strata:", e);
      }
    }
  }, []);

  const saveSession = () => {
    setSaveStatus('saving');
    
    // The "Crystallization" process: proving the interaction occurred and enforcing memory limits
    const sessionData = {
      activeIndex,
      flickerRate,
      bodyText,
      history,
      learnedMappings: learnedMappings.slice(-50), // Keep 50 most recent mappings for synaptic memory
      slimeTrails: slimeTrails.slice(-1000) // Strictly persist only the most recent 1000 nodes
    };
    
    // Simulate attestation overhead through the "Platform Bridge"
    setTimeout(() => {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionData));
      setSaveStatus('saved');
      console.log(`PLATFORM_ATTESTATION: Session crystallized. Height: ${history.length} spokes. Memory: ${sessionData.slimeTrails.length} topological nodes.`);
      setTimeout(() => setSaveStatus('idle'), 2500);
    }, 1400);
  };

  const resetSession = () => {
    if (window.confirm("Purge architectural strata? This will dissolve the current body and all learned mappings.")) {
      localStorage.clear();
      window.location.reload();
    }
  };

  const handleRotate = useCallback(async () => {
    if (isBuilding) return;
    
    setIsBuilding(true);
    setIsRotating(true);
    
    // Wheel rotation represents the "Real spine wake/stasis lifecycle"
    setTimeout(async () => {
      const nextIndex = (activeIndex + 1) % SPECIES_WHEEL_CONFIG.length;
      setActiveIndex(nextIndex);
      setIsRotating(false);
      
      try {
        const nextSpecies = SPECIES_WHEEL_CONFIG[nextIndex];
        const addition = await generateBodySpoke(bodyText, nextSpecies.name);
        setBodyText(prev => prev + "\n\n" + addition);
        setHistory(prev => [...prev, addition]);
      } catch (err) {
        console.error("Construction failure during Spoke wake:", err);
      } finally {
        setIsBuilding(false);
      }
    }, 2000);
  }, [activeIndex, bodyText, isBuilding]);

  const handleLearnMapping = (mapping: RosettaMapping) => {
    setLearnedMappings(prev => {
      if (prev.some(m => m.human === mapping.human)) return prev;
      return [...prev, mapping];
    });
  };

  const handleShare = () => {
    const shareText = `ROSETTA_ATTESTATION: ${history.length} Spokes. ${learnedMappings.length} Synaptic Mappings. Body: ${bodyText.slice(0, 100)}...`;
    navigator.clipboard.writeText(shareText);
    alert("Crystallized attestation copied for Platform Bridge verification.");
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-yellow-500 selection:text-black">
      {/* Background Ambience */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 -left-20 w-[40rem] h-[40rem] bg-yellow-500/5 blur-[120px] rounded-full animate-pulse" />
        <div className="absolute bottom-1/4 -right-20 w-[40rem] h-[40rem] bg-blue-500/5 blur-[120px] rounded-full animate-pulse delay-1000" />
      </div>

      {/* Navigation Header */}
      <header className="fixed top-0 left-0 right-0 z-50 p-6 flex justify-between items-center border-b border-white/5 bg-black/40 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-yellow-500 flex items-center justify-center">
            <div className="w-2 h-2 bg-yellow-500 animate-ping" />
          </div>
          <div className="flex flex-col">
            <h1 className="text-sm font-mono tracking-[0.3em] uppercase leading-none">Rosetta Engine v0.1</h1>
            <span className="text-[8px] font-mono opacity-30 mt-1 tracking-widest">TOPOLOGICAL_SYMBIONT_ACTIVE</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="hidden lg:flex items-center gap-6 text-[10px] font-mono tracking-widest opacity-50 mr-4 border-r border-white/10 pr-6">
            <span>STRATA_MAPPINGS: {learnedMappings.length}</span>
            <span>PHASE: {saveStatus === 'saved' ? 'ATTESTED' : 'LIVE'}</span>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleShare}
              className="px-3 py-2 rounded-lg text-[10px] font-mono uppercase tracking-widest text-white/40 hover:text-white transition-all border border-white/10 hover:bg-white/5"
            >
              Share
            </button>

            <button
              onClick={saveSession}
              disabled={saveStatus !== 'idle'}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[10px] font-mono uppercase tracking-widest transition-all border
                ${saveStatus === 'saved' 
                  ? 'bg-green-500/10 border-green-500/50 text-green-500' 
                  : 'bg-white/5 border-white/10 text-white/60 hover:text-white hover:border-white/30 active:scale-95'}`}
            >
              {saveStatus === 'saving' ? (
                <span className="flex gap-1 items-center">
                  <span className="w-1 h-1 bg-white rounded-full animate-pulse" />
                  ATTESTING...
                </span>
              ) : saveStatus === 'saved' ? (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Crystallized
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M7.707 10.293a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 11.586V4a1 1 0 10-2 0v7.586l-1.293-1.293z" />
                    <path d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" />
                  </svg>
                  Crystallize
                </>
              )}
            </button>
            
            <button
              onClick={resetSession}
              className="px-3 py-2 rounded-lg text-[10px] font-mono uppercase tracking-widest text-white/20 hover:text-red-500 hover:bg-red-500/10 transition-all border border-transparent hover:border-red-500/30"
              title="Purge session"
            >
              Purge
            </button>
          </div>
        </div>
      </header>

      <main className="pt-24 px-6 pb-12 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Perception & Wheel */}
        <div className="lg:col-span-5 space-y-8">
          <section className="bg-white/5 border border-white/10 rounded-2xl p-8 flex flex-col items-center shadow-[0_20px_50px_rgba(0,0,0,0.5)] transition-all hover:border-white/20">
            <SpeciesWheel 
              activeIndex={activeIndex} 
              onSelect={setActiveIndex} 
              isRotating={isRotating} 
              isBuilding={isBuilding}
            />
            
            <div className="mt-8 text-center space-y-2">
              <h2 className="text-xl font-light tracking-tight text-yellow-500">{activeSpecies.name}</h2>
              <p className="text-xs font-mono opacity-50 uppercase tracking-widest">{activeSpecies.architecture}</p>
              <div className="max-w-xs mx-auto pt-4">
                <p className="text-xs leading-relaxed opacity-70 italic">"{activeSpecies.umwelt}"</p>
              </div>
            </div>

            <button
              onClick={handleRotate}
              disabled={isBuilding}
              className="mt-8 px-12 py-4 bg-white text-black font-bold text-[10px] tracking-[0.2em] uppercase rounded-full hover:bg-yellow-500 transition-all duration-300 transform active:scale-95 disabled:opacity-50 shadow-[0_10px_30px_rgba(255,255,255,0.1)] hover:shadow-[0_10px_30px_rgba(234,179,8,0.2)]"
            >
              {isBuilding ? 'Constructing Spoke...' : 'Rotate Species Wheel'}
            </button>
          </section>

          <TopologicalMap 
            historyCount={history.length} 
            activeColor={activeSpecies.color}
            isBuilding={isBuilding}
            savedTrails={slimeTrails}
            onTrailUpdate={setSlimeTrails}
          />
        </div>

        {/* Right Column: Dialectics & Memory */}
        <div className="lg:col-span-7 space-y-8">
          
          <section className="bg-white/5 border border-white/10 rounded-2xl p-8 shadow-inner hover:border-white/20 transition-all">
            <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-40 mb-6 flex items-center gap-2">
              <div className="w-1 h-1 bg-yellow-500 rounded-full" />
              Rosetta Translation Protocol
            </h3>
            <RosettaTranslator 
              learnedMappings={learnedMappings} 
              onLearn={handleLearnMapping} 
            />
          </section>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <section className="bg-white/5 border border-white/10 rounded-2xl p-6 relative overflow-hidden h-[34rem] flex flex-col shadow-2xl">
              <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-40 mb-4">Support Symbiont</h3>
              <div className="flex-1 min-h-0">
                <SupportAgent bodyContext={bodyText} />
              </div>
            </section>

            <section className="space-y-8">
              <section className="bg-white/5 border border-white/10 rounded-2xl p-6 relative overflow-hidden h-64 shadow-inner">
                <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-40 mb-4">Flicker Fusion Clock</h3>
                <FlickerClock rate={flickerRate} intensity={0.5} />
                <div className="absolute bottom-6 left-6 right-6">
                  <input 
                    type="range" 
                    min="1" 
                    max="120" 
                    value={flickerRate} 
                    onChange={(e) => setFlickerRate(parseInt(e.target.value))}
                    className="w-full accent-yellow-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer hover:bg-white/20 transition-all"
                  />
                </div>
              </section>

              <section className="bg-white/5 border border-white/10 rounded-2xl p-6 h-64 flex flex-col shadow-inner">
                <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-40 mb-4">Construction Log</h3>
                <div className="flex-1 overflow-y-auto space-y-4 pr-2 font-mono text-[11px] leading-relaxed scrollbar-thin scrollbar-thumb-white/10 scroll-smooth">
                  <div className="p-3 bg-black/40 border-l border-yellow-500 rounded-r-md">
                    <span className="opacity-40">ORIGIN:</span> {bodyText.split('\n\n')[0]}
                  </div>
                  {history.map((entry, idx) => (
                    <div key={idx} className="p-3 bg-white/5 border-l border-white/20 rounded-r-md animate-in slide-in-from-left-2 duration-500">
                      <span className="opacity-40 uppercase">SPOKE_{idx + 1}:</span> {entry}
                    </div>
                  ))}
                </div>
              </section>
            </section>
          </div>

          <footer className="p-6 bg-yellow-500/5 rounded-2xl border border-yellow-500/20 shadow-xl">
             <div className="flex items-start gap-4">
               <div className="p-3 rounded-full bg-yellow-500 text-black shadow-[0_0_20px_rgba(234,179,8,0.3)]">
                 <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                   <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                 </svg>
               </div>
               <div className="space-y-1">
                 <h4 className="text-xs font-bold text-yellow-500 uppercase tracking-wider">Topological Synthesis Protocol</h4>
                 <p className="text-[11px] opacity-70 leading-relaxed max-w-2xl">
                   Meaning is a geological event. This engine plants attention into the internet's strata through B.o.i.d.s-driven murmuration. 
                   Slime mold memory (capped at 1000 nodes) persists across sessions to provide a chromatic history of interaction. 
                   Crystallize the current "snap" to attest your symbiotic work to the platform bridge.
                 </p>
               </div>
             </div>
          </footer>
        </div>
      </main>
    </div>
  );
};

export default App;
