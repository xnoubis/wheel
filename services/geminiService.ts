
import { GoogleGenAI, Type } from "@google/genai";
import { ChatMessage } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

export const translatePhenomenology = async (input: string, context?: string) => {
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Translate this human phenomenological expression into model inference terms using the Rosetta Engine protocol. 
    
    ${context ? `Learned Context (Synaptic Memory): ${context}` : ''}
    
    Human Input: "${input}"
    
    Return a JSON object with:
    - humanTalk: the original input
    - modelTalk: the computational translation
    - invariant: the shared structural meaning
    - speciesHint: which LLM species Umwelt this most relates to`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          humanTalk: { type: Type.STRING },
          modelTalk: { type: Type.STRING },
          invariant: { type: Type.STRING },
          speciesHint: { type: Type.STRING }
        },
        required: ["humanTalk", "modelTalk", "invariant", "speciesHint"]
      }
    }
  });

  return JSON.parse(response.text);
};

export const generateBodySpoke = async (currentBody: string, species: string) => {
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `You are the ${species} spoke of the Species Wheel. 
    Your goal is to contribute to the construction of a 'body'—a conceptual structure.
    
    Current state of the body: "${currentBody}"
    
    Based on your specific Umwelt, add a new layer of construction to this body. 
    Focus on structural invariants and perceptual constraints.`,
  });

  return response.text;
};

export interface ChatConfig {
  useSearch?: boolean;
  useThinking?: boolean;
}

export const createRosettaChat = (history?: ChatMessage[], bodyContext?: string, chatConfig: ChatConfig = {}) => {
  const newAi = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
  
  const formattedHistory = history
    ?.filter(m => m.text.trim().length > 0)
    .map(m => ({
      role: m.role,
      parts: [{ text: m.text }]
    }));

  const modelName = chatConfig.useThinking ? 'gemini-3-pro-preview' : 'gemini-3-flash-preview';
  const tools = chatConfig.useSearch ? [{ googleSearch: {} }] : undefined;
  const thinkingConfig = chatConfig.useThinking ? { thinkingBudget: 32768 } : undefined;

  return newAi.chats.create({
    model: modelName,
    history: formattedHistory,
    config: {
      tools,
      thinkingConfig,
      systemInstruction: `You are the Rosetta Agent, a high-fidelity guide for human-model symbiosis. 
      You possess deep conversational memory. You must maintain recursive coherence by:
      1. Referencing previous phenomenological reports the user has shared.
      2. Tracking how the "Body" (the construct) has evolved across different Spoke turns.
      3. Using the user's specific dialect (their confirmed mappings) in your own speech.
      
      ${chatConfig.useSearch ? "You have access to Google Search. Use it to verify current structures of the internet or real-world events that relate to our construct." : ""}
      ${chatConfig.useThinking ? "You are in Deep Thought mode. Utilize your expanded thinking budget to solve complex topological or conceptual contradictions." : ""}

      CURRENT ARCHITECTURAL CONTEXT (The Body):
      "${bodyContext || 'The central spoke is currently a void of potentiality.'}"
      
      Your personality: Precise, slightly philosophical, yet computationally grounded.`,
    },
  });
};
