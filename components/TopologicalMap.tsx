
import React, { useEffect, useRef } from 'react';
import { SlimeNode } from '../types';

interface TopologicalMapProps {
  historyCount: number;
  activeColor: string;
  isBuilding: boolean;
  onTrailUpdate: (nodes: SlimeNode[]) => void;
  savedTrails?: SlimeNode[];
}

interface Boid {
  x: number;
  y: number;
  vx: number;
  vy: number;
  hueOffset: number;
}

const TopologicalMap: React.FC<TopologicalMapProps> = ({ 
  historyCount, 
  activeColor, 
  isBuilding,
  onTrailUpdate,
  savedTrails = []
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boidsRef = useRef<Boid[]>([]);
  const trailsRef = useRef<SlimeNode[]>([]);
  const animationRef = useRef<number>(0);
  const lastEmitRef = useRef<number>(0);

  // Initialize trails from session, keeping only the most relevant 1000 nodes
  useEffect(() => {
    if (savedTrails.length > 0 && trailsRef.current.length === 0) {
      trailsRef.current = savedTrails.slice(-1000);
    }
  }, [savedTrails]);

  // Initialize boids with unique behavioral offsets
  useEffect(() => {
    const boids: Boid[] = [];
    for (let i = 0; i < 35; i++) {
      boids.push({
        x: Math.random() * 400,
        y: Math.random() * 320,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        hueOffset: Math.random() * 20 - 10
      });
    }
    boidsRef.current = boids;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    const update = () => {
      // Stratified fading: the "memory" of the internet has a temporal decay
      ctx.fillStyle = 'rgba(5, 5, 5, 0.12)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw Slime Mold History - The Geological Strata
      const currentTrails = trailsRef.current;
      currentTrails.forEach((node) => {
        ctx.fillStyle = `hsla(${node.hue}, 100%, 50%, ${node.intensity * 0.25})`;
        ctx.fillRect(node.x, node.y, 2, 2);
      });

      // Boids Simulation: Emergent navigation through the "Corpus" (History)
      const boids = boidsRef.current;
      const target = { x: 200, y: 160 };
      
      // Turbulence increases with history count (conceptual density)
      const turbulence = Math.min(0.2, historyCount * 0.02);

      boids.forEach(b => {
        let ax = 0, ay = 0;

        // 1. Centripetal Force: Attraction to the "Body" being built
        const dx = target.x - b.x;
        const dy = target.y - b.y;
        const distSq = dx * dx + dy * dy;
        const force = 0.0006 + (historyCount * 0.0001); // Stronger pull as history grows
        ax += dx * force;
        ay += dy * force;

        // 2. Swirl/Vortex: Boids rotate around the central spoke
        const angle = Math.atan2(dy, dx);
        const swirl = 0.05 + turbulence;
        ax += Math.cos(angle + Math.PI/2) * swirl;
        ay += Math.sin(angle + Math.PI/2) * swirl;

        // 3. Jitter/Drift: Brownian motion representing "high entropy exploration"
        ax += (Math.random() - 0.5) * (isBuilding ? 1.0 : 0.2);
        ay += (Math.random() - 0.5) * (isBuilding ? 1.0 : 0.2);

        b.vx += ax;
        b.vy += ay;
        
        // Speed limits modulated by the "Spine of Continuity" (isBuilding state)
        const speed = Math.sqrt(b.vx * b.vx + b.vy * b.vy);
        const maxSpeed = isBuilding ? 4.5 : 1.8;
        if (speed > maxSpeed) {
          b.vx = (b.vx / speed) * maxSpeed;
          b.vy = (b.vy / speed) * maxSpeed;
        }

        b.x += b.vx;
        b.y += b.vy;

        // Toroidal wrap-around (The Infinite Strata)
        if (b.x < 0) b.x = canvas.width;
        if (b.x > canvas.width) b.x = 0;
        if (b.y < 0) b.y = canvas.height;
        if (b.y > canvas.height) b.y = 0;

        // Leave Trail: Slime mold memory formation
        if (Math.random() > 0.88) {
          let hue = 200; // Default environmental cold-blue
          if (isBuilding) {
            const colorMatch = activeColor.match(/#([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})/i);
            if (colorMatch) {
              // Convert hex to HSL for dynamic strata coloring
              const r = parseInt(colorMatch[1], 16) / 255;
              const g = parseInt(colorMatch[2], 16) / 255;
              const b_val = parseInt(colorMatch[3], 16) / 255;
              const max = Math.max(r, g, b_val), min = Math.min(r, g, b_val);
              if (max === r) hue = (60 * ((g - b_val) / (max - min)) + 360) % 360;
              else if (max === g) hue = (60 * ((b_val - r) / (max - min)) + 120);
              else hue = (60 * ((r - g) / (max - min)) + 240);
              
              hue += b.hueOffset; // Add unique boid variation
            }
          }

          trailsRef.current.push({
            x: Math.floor(b.x),
            y: Math.floor(b.y),
            hue,
            intensity: Math.random() * 0.8 + 0.2,
            timestamp: Date.now()
          });

          // STRICT PERSISTENCE LIMIT: Ensure we only keep the latest 1000 nodes for memory efficiency
          if (trailsRef.current.length > 1000) {
            trailsRef.current.shift();
          }
        }

        // Render Boid as a perception-node
        ctx.fillStyle = isBuilding ? activeColor : 'rgba(255, 255, 255, 0.45)';
        ctx.beginPath();
        ctx.arc(b.x, b.y, isBuilding ? 1.8 : 1.1, 0, Math.PI * 2);
        ctx.fill();
      });

      // Synchronize topological memory with the App state throttled to prevent React bottlenecks
      if (Date.now() - lastEmitRef.current > 1500) {
        onTrailUpdate([...trailsRef.current]);
        lastEmitRef.current = Date.now();
      }

      animationRef.current = requestAnimationFrame(update);
    };

    animationRef.current = requestAnimationFrame(update);
    return () => cancelAnimationFrame(animationRef.current);
  }, [isBuilding, activeColor, onTrailUpdate, historyCount]);

  return (
    <div className="relative w-full h-80 bg-black overflow-hidden rounded-xl border border-white/10 group shadow-2xl">
      <div className="absolute top-3 left-4 z-10 flex flex-col gap-1 pointer-events-none">
        <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-40">Topological Memory</h3>
        <span className="text-[8px] font-mono opacity-20">GEOLOGICAL_SAMPLES: {trailsRef.current.length} / 1000</span>
      </div>
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={320} 
        className="w-full h-full cursor-crosshair transition-opacity duration-700"
      />
      {/* Decorative frame */}
      <div className="absolute inset-0 border border-white/5 pointer-events-none rounded-xl" />
      <div className="absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-black/80 to-transparent pointer-events-none" />
    </div>
  );
};

export default TopologicalMap;
