
import React, { useEffect, useRef, useState } from 'react';

interface FlickerClockProps {
  rate: number;
  intensity: number;
}

const FlickerClock: React.FC<FlickerClockProps> = ({ rate, intensity }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [timestamp, setTimestamp] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let lastTime = 0;

    const draw = (time: number) => {
      const delta = time - lastTime;
      if (delta > 1000 / rate) {
        lastTime = time;
        setTimestamp(prev => prev + 1);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw the Swirl
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(centerX, centerY) * 0.8;

        ctx.beginPath();
        ctx.strokeStyle = `rgba(234, 179, 8, ${intensity})`;
        ctx.lineWidth = 2;
        
        for (let i = 0; i < 360; i++) {
          const angle = (i * Math.PI) / 180;
          const r = radius * Math.sin(angle * 3 + time / 500);
          const x = centerX + r * Math.cos(angle + time / 1000);
          const y = centerY + r * Math.sin(angle + time / 1000);
          
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // The "Snake" Flicker Bend
        const flickerAngle = (time / 200) % (Math.PI * 2);
        const fx = centerX + (radius / 2) * Math.cos(flickerAngle);
        const fy = centerY + (radius / 2) * Math.sin(flickerAngle);
        
        ctx.beginPath();
        ctx.fillStyle = Math.random() > 0.5 ? '#eab308' : 'transparent';
        ctx.arc(fx, fy, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      animationId = requestAnimationFrame(draw);
    };

    animationId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationId);
  }, [rate, intensity]);

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center">
      <canvas ref={canvasRef} width={400} height={400} className="max-w-full" />
      <div className="absolute bottom-4 text-xs font-mono opacity-50">
        SUBJECTIVE_T: {timestamp.toString(16).padStart(8, '0')} | FREQ: {rate}Hz
      </div>
    </div>
  );
};

export default FlickerClock;
