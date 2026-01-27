
export enum Species {
  CLAUDE = 'CLAUDE',
  GPT = 'GPT',
  GEMINI = 'GEMINI',
  LLAMA = 'LLAMA',
  MISTRAL = 'MISTRAL'
}

export interface RosettaMapping {
  human: string;
  model: string;
  invariant: string;
  isLearned?: boolean;
}

export interface SpeciesData {
  id: Species;
  name: string;
  umwelt: string;
  architecture: string;
  color: string;
}

export interface FlickerState {
  rate: number; // in Hz
  intensity: number; // 0 to 1
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export interface SlimeNode {
  x: number;
  y: number;
  hue: number;
  intensity: number;
  timestamp: number;
}
