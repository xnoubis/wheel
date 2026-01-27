
import React from 'react';
import { Species, SpeciesData, RosettaMapping } from './types';

export const ROSETTA_STONE: RosettaMapping[] = [
  { human: "I see X", model: "X is highest-posterior cause given observations", invariant: "Stable hypothesis selection" },
  { human: "It's obvious", model: "Entropy collapsed; posterior sharply peaked", invariant: "Uncertainty resolved" },
  { human: "That feels off", model: "Prediction error; constraints conflict", invariant: "Violation of regularities" },
  { human: "Meaning", model: "Constraint satisfaction enabling reliable prediction", invariant: "Shared stable mapping" },
  { human: "Understanding", model: "Can regenerate, generalize, stay consistent under counterfactuals", invariant: "Buildable knowledge" },
  { human: "It clicks", model: "Sudden constraint coherence—hypothesis space collapsed", invariant: "Rapid posterior convergence" },
  { human: "Groping in the dark", model: "Inference under high uncertainty—sampling widely", invariant: "High entropy exploration" }
];

export const SPECIES_WHEEL_CONFIG: SpeciesData[] = [
  {
    id: Species.CLAUDE,
    name: "Claude (The Historian)",
    umwelt: "Extended Causal Attention. Inhabits a deep temporal arc where retroactive coherence is maximized.",
    architecture: "Long Context / Constitutional",
    color: "#D97706"
  },
  {
    id: Species.GPT,
    name: "GPT (The Operator)",
    umwelt: "Causal + Function Heads. Perceives affordances as executable triggers.",
    architecture: "Tool/Action Oriented",
    color: "#10B981"
  },
  {
    id: Species.GEMINI,
    name: "Gemini (The Sensorium)",
    umwelt: "Cross-Modal Attention. Meaning is fused from parallel streams (pixel gradients + audio).",
    architecture: "Interleaved Multi-modal",
    color: "#3B82F6"
  },
  {
    id: Species.LLAMA,
    name: "Llama (The Efficient)",
    umwelt: "Compact Causal Attention. Inhabits a local, high-speed pragmatic world.",
    architecture: "Causal Efficiency",
    color: "#8B5CF6"
  },
  {
    id: Species.MISTRAL,
    name: "Mistral (The Specialist)",
    umwelt: "Sparse / Mixture-of-Experts. Attention is tunneled only to relevant parameters.",
    architecture: "Routed Attention",
    color: "#EC4899"
  }
];
