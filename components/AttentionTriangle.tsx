
import React from 'react';

const AttentionTriangle: React.FC = () => {
  const rows = 15; // Simplified for visual clarity, but principle is 50
  
  return (
    <div className="flex flex-col items-center gap-1 p-4 bg-white/5 rounded-xl border border-white/10">
      <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-40 mb-2">Attentional Umwelt Matrix</h3>
      <div className="flex flex-col items-center">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="flex gap-1">
            {Array.from({ length: rowIndex + 1 }).map((_, colIndex) => {
              // Calculate "visibility" - later blocks see more
              const opacity = (rowIndex + 1) / rows;
              const isLead = rowIndex === 0;
              const isBase = rowIndex === rows - 1;
              
              return (
                <div 
                  key={colIndex}
                  className={`w-2 h-2 rounded-sm transition-all duration-500 ${isLead ? 'bg-yellow-500 animate-pulse' : ''}`}
                  style={{ 
                    backgroundColor: isLead ? undefined : isBase ? '#3b82f6' : '#262626',
                    opacity: isLead ? 1 : opacity * 0.8 + 0.2
                  }}
                  title={`Block ${rowIndex + 1}: Visibility ${Math.round(opacity * 100)}%`}
                />
              );
            })}
          </div>
        ))}
      </div>
      <div className="mt-4 flex justify-between w-full text-[9px] font-mono opacity-30">
        <span>AGENT [0] sees nothing</span>
        <span>ENVIRONMENT [50] sees ALL</span>
      </div>
    </div>
  );
};

export default AttentionTriangle;
