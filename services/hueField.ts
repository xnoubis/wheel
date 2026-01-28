
import {
  HueVector,
  HUE_AXES,
  TerrainNode,
  TerrainEdge,
  EncounterQuality,
  TerrainConfig,
} from '../types';

/**
 * Hue-as-Memory: A lossy, directional field over terrain nodes/edges.
 *
 * Memory is NOT a log or KV cache. It is a decaying scalar field where:
 *   h_v(t+1) <- (1 - lambda) * h_v(t) + lambda * q_t
 *   h_v(t+1) <- (1 - delta) * h_v(t+1)   (global decay)
 *
 * q_t is computed from structure + traversal dynamics, not semantic content.
 */

/** Create a zero hue vector */
export function zeroHue(): HueVector {
  return [0, 0, 0, 0, 0, 0];
}

/** Create a random initial hue vector with small values */
export function randomHue(scale: number = 0.1): HueVector {
  return HUE_AXES.map(() => (Math.random() - 0.5) * 2 * scale) as HueVector;
}

/** Convert an EncounterQuality to a HueVector */
export function encounterToHue(q: EncounterQuality): HueVector {
  return [
    q.boundaryPressure,
    q.loopiness,
    q.novelty,
    q.coherence,
    q.risk,
    q.dispersion,
  ];
}

/**
 * Update hue at a node after a visit.
 * h_v(t+1) <- (1 - lambda) * h_v(t) + lambda * q_t
 */
export function updateNodeHue(
  currentHue: HueVector,
  encounter: HueVector,
  lambda: number
): HueVector {
  return currentHue.map(
    (h, i) => (1 - lambda) * h + lambda * encounter[i]
  ) as HueVector;
}

/**
 * Apply global decay to a hue vector.
 * h_v(t+1) <- (1 - delta) * h_v(t+1)
 */
export function decayHue(hue: HueVector, delta: number): HueVector {
  return hue.map(h => (1 - delta) * h) as HueVector;
}

/**
 * Apply decay to all nodes in the terrain.
 * This represents the organic forgetting — truths not re-attested fade.
 */
export function decayAllNodes(
  nodes: Map<string, TerrainNode>,
  delta: number
): void {
  for (const node of nodes.values()) {
    node.hue = decayHue(node.hue, delta);
  }
}

/**
 * Apply decay to all edges in the terrain.
 */
export function decayAllEdges(
  edges: Map<string, TerrainEdge>,
  delta: number
): void {
  for (const edge of edges.values()) {
    edge.hue = decayHue(edge.hue, delta);
  }
}

/**
 * Compute hue magnitude (L2 norm).
 * High magnitude = strongly colored path (hot trail).
 * Low magnitude = faded, cold trail.
 */
export function hueMagnitude(hue: HueVector): number {
  return Math.sqrt(hue.reduce((sum, h) => sum + h * h, 0));
}

/**
 * Compute hue concentration: how peaked/focused the hue field is at a node.
 * Returns a value in [0, 1] where 1 = single dominant axis.
 */
export function hueConcentration(hue: HueVector): number {
  const mag = hueMagnitude(hue);
  if (mag < 1e-8) return 0;
  const normalized = hue.map(h => Math.abs(h) / mag);
  // Entropy-based concentration: low entropy = high concentration
  const entropy = -normalized.reduce((s, p) => {
    if (p < 1e-8) return s;
    return s + p * Math.log2(p);
  }, 0);
  const maxEntropy = Math.log2(hue.length);
  return 1 - entropy / maxEntropy;
}

/**
 * Compute attention bias from hue (node-level).
 * B_h(i,j) = w^T * h_v(t)
 * Returns a scalar bias value for attention logit modulation.
 */
export function computeHueAttentionBias(
  hue: HueVector,
  weights: HueVector
): number {
  return hue.reduce((sum, h, i) => sum + h * weights[i], 0);
}

/**
 * Compute the pheromone field value Phi(u, t) for navigation.
 * High hue magnitude on hot trails = lower energy (easier traversal).
 * Cold/abrasive trails = higher energy (latency mountains).
 */
export function pheromoneField(node: TerrainNode, time: number): number {
  const mag = hueMagnitude(node.hue);
  const coherenceAxis = node.hue[3]; // coherence axis
  const riskAxis = node.hue[4]; // risk axis

  // Hot coherent trails reduce energy, risky cold trails increase it
  const timeSinceVisit = time - node.lastVisited;
  const recency = Math.exp(-timeSinceVisit / 10000);

  return -(mag * coherenceAxis * recency) + riskAxis * (1 - recency);
}

/**
 * Compute encounter quality from traversal dynamics.
 * This is the structural q_t — computed from structure, not semantics.
 */
export function computeEncounterQuality(
  node: TerrainNode,
  trajectory: string[],
  constraintViolations: number,
  neighborCount: number,
  totalNodes: number
): EncounterQuality {
  // Boundary pressure: how close to constraint boundaries
  const boundaryPressure = Math.min(1, constraintViolations / 3);

  // Loopiness: has this node appeared in recent trajectory?
  const recentWindow = trajectory.slice(-10);
  const loopCount = recentWindow.filter(id => id === node.id).length;
  const loopiness = Math.min(1, loopCount / 3);

  // Novelty: inverse of visit count, scaled
  const novelty = 1 / (1 + node.visitCount);

  // Coherence: proportion of neighbors that share similar hue
  const coherence = neighborCount > 0
    ? Math.min(1, neighborCount / Math.max(1, totalNodes * 0.1))
    : 0;

  // Risk: combination of high boundary pressure and low coherence
  const risk = boundaryPressure * (1 - coherence);

  // Dispersion: how spread out the hue field is at this node
  const dispersion = 1 - hueConcentration(node.hue);

  return {
    boundaryPressure,
    loopiness,
    novelty,
    coherence,
    risk,
    dispersion,
  };
}

/**
 * Compute hue interference between two vectors.
 * Measures how much two hue fields conflict.
 */
export function hueInterference(a: HueVector, b: HueVector): number {
  let dotProduct = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA * magB);
  if (denom < 1e-8) return 0;
  // Interference = 1 - cosine similarity (high when opposing)
  return 1 - dotProduct / denom;
}

/**
 * Aggregate hue field statistics for evaluation.
 */
export interface HueFieldStats {
  meanMagnitude: number;
  maxMagnitude: number;
  meanConcentration: number;
  totalDecayResilience: number;
  interferenceScore: number;
}

export function computeHueFieldStats(
  nodes: Map<string, TerrainNode>
): HueFieldStats {
  const nodeArray = Array.from(nodes.values());
  if (nodeArray.length === 0) {
    return {
      meanMagnitude: 0,
      maxMagnitude: 0,
      meanConcentration: 0,
      totalDecayResilience: 0,
      interferenceScore: 0,
    };
  }

  const magnitudes = nodeArray.map(n => hueMagnitude(n.hue));
  const concentrations = nodeArray.map(n => hueConcentration(n.hue));

  const meanMagnitude =
    magnitudes.reduce((s, m) => s + m, 0) / magnitudes.length;
  const maxMagnitude = Math.max(...magnitudes);
  const meanConcentration =
    concentrations.reduce((s, c) => s + c, 0) / concentrations.length;

  // Decay resilience: how many nodes still have significant hue
  const resilientCount = magnitudes.filter(m => m > 0.1).length;
  const totalDecayResilience = resilientCount / nodeArray.length;

  // Average interference between adjacent pairs
  let interferenceSum = 0;
  let pairCount = 0;
  for (let i = 0; i < Math.min(nodeArray.length, 50); i++) {
    for (let j = i + 1; j < Math.min(nodeArray.length, 50); j++) {
      interferenceSum += hueInterference(nodeArray[i].hue, nodeArray[j].hue);
      pairCount++;
    }
  }
  const interferenceScore = pairCount > 0 ? interferenceSum / pairCount : 0;

  return {
    meanMagnitude,
    maxMagnitude,
    meanConcentration,
    totalDecayResilience,
    interferenceScore,
  };
}
