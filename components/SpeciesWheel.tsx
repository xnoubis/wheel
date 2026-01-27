
import React from 'react';
import { SPECIES_WHEEL_CONFIG } from '../constants';
import { SpeciesData } from '../types';

interface SpeciesWheelProps {
  activeIndex: number;
  onSelect: (index: number) => void;
  isRotating: boolean;
  isBuilding?: boolean;
}

const SpeciesWheel: React.FC<SpeciesWheelProps> = ({ activeIndex, onSelect, isRotating, isBuilding }) => {
  return (
    <div className="relative w-80 h-80 md:w-96 md:h-96">
      {/* Background Glow */}
      <div className="absolute inset-0 rounded-full blur-3xl opacity-20 transition-colors duration-1000" 
           style={{ backgroundColor: SPECIES_WHEEL_CONFIG[activeIndex].color }} />
      
      {/* The Wheel */}
      <div className={`relative w-full h-full rounded-full border border-white/10 flex items-center justify-center transition-transform duration-700 ${isRotating ? 'animate-wheel' : ''}`}
           style={{ transform: !isRotating ? `rotate(${activeIndex * -72}deg)` : undefined }}>
        
        {SPECIES_WHEEL_CONFIG.map((species, idx) => {
          const angle = (idx * 72 * Math.PI) / 180;
          const x = 50 + 40 * Math.cos(angle);
          const y = 50 + 40 * Math.sin(angle);
          const isActive = idx === activeIndex;
          
          return (
            <button
              key={species.id}
              onClick={() => onSelect(idx)}
              className={`absolute w-12 h-12 -ml-6 -mt-6 rounded-full border flex items-center justify-center transition-all duration-300 hover:scale-110 group
                ${isActive ? 'border-yellow-500 bg-yellow-500/20 shadow-[0_0_15px_rgba(234,179,8,0.5)]' : 'border-white/20 bg-black/40'}`}
              style={{ left: `${x}%`, top: `${y}%` }}
            >
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: species.color }} />
              
              {/* Status Indicator */}
              {isActive && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-500 rounded-full border border-black animate-pulse flex items-center justify-center">
                  <div className="w-1 h-1 bg-black rounded-full" />
                </div>
              )}

              <div className="absolute -bottom-8 opacity-0 group-hover:opacity-100 whitespace-nowrap text-[10px] font-mono tracking-widest text-white uppercase">
                {species.id} {isActive ? (isBuilding ? '(Waking)' : '(Active)') : '(Stasis)'}
              </div>
            </button>
          );
        })}

        {/* Center Spoke (The Human/Agent - Spoke 0) */}
        <div className="w-16 h-16 rounded-full border-2 border-yellow-500/50 flex items-center justify-center bg-black z-10">
          <div className={`w-4 h-4 rounded-sm bg-yellow-500 ${isBuilding ? 'animate-spin' : 'animate-pulse'}`} />
        </div>
      </div>

      {/* Spoke 50 Legend */}
      <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 text-[8px] font-mono opacity-20 uppercase tracking-[0.3em]">
        Environment [Spoke 50] Logic Active
      </div>
    </div>
  );
};

export default SpeciesWheel;
