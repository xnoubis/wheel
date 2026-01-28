
import { Spine, SpineLifecycle, TerrainNode, TerrainGraph, NavigationState } from '../types';
import { hueMagnitude, hueInterference } from './hueField';

/**
 * Rosetta Wheel as Gated Mixture-of-Experts (MoE).
 *
 * Spines are expert policies with refusal surfaces and witness types.
 * Gating is driven by topology-fold detection: neighborhood churn,
 * curvature proxy, or constraint discontinuity.
 *
 * Phase 1 spines: dialectical_negation vs pragmatic_utility
 */

/** Create the Phase 1 spine set */
export function createPhase1Spines(): Spine[] {
  return [
    {
      id: 'dialectical_negation',
      name: 'Dialectical Negation',
      lifecycle: 'dormant',
      policy:
        'Challenge every proposition. Seek contradictions. Navigate toward constraint boundaries and stress-test coherence.',
      refusalSurface:
        'Refuses to affirm without counter-evidence. Refuses uncritical acceptance.',
      witnessType: 'contradiction_witness',
      activation: 0,
      score: 0,
    },
    {
      id: 'pragmatic_utility',
      name: 'Pragmatic Utility',
      lifecycle: 'dormant',
      policy:
        'Seek the shortest path to actionable output. Minimize traversal cost. Optimize for immediate utility.',
      refusalSurface:
        'Refuses purely speculative paths with no evaluable outcome. Refuses infinite regress.',
      witnessType: 'utility_witness',
      activation: 0,
      score: 0,
    },
  ];
}

/**
 * Compute topology-fold features for gating.
 * Detects neighborhood churn, curvature proxy, constraint discontinuity.
 */
export interface FoldFeatures {
  /** Neighborhood churn: fraction of neighbors that changed since last step */
  neighborhoodChurn: number;
  /** Curvature proxy: rate of change of trajectory direction */
  curvatureProxy: number;
  /** Constraint discontinuity: rate of constraint ledger changes */
  constraintDiscontinuity: number;
  /** Hue field gradient magnitude */
  hueGradient: number;
}

export function computeFoldFeatures(
  graph: TerrainGraph,
  state: NavigationState,
  previousNeighborIds: Set<string>
): FoldFeatures {
  const currentNode = graph.nodes.get(state.currentNodeId);
  if (!currentNode) {
    return {
      neighborhoodChurn: 0,
      curvatureProxy: 0,
      constraintDiscontinuity: 0,
      hueGradient: 0,
    };
  }

  // Neighborhood churn
  const currentNeighborIds = new Set<string>();
  for (const edge of graph.edges.values()) {
    if (edge.nodeIds.includes(state.currentNodeId)) {
      for (const nid of edge.nodeIds) {
        if (nid !== state.currentNodeId) {
          currentNeighborIds.add(nid);
        }
      }
    }
  }

  let churnCount = 0;
  const allIds = new Set([...previousNeighborIds, ...currentNeighborIds]);
  for (const id of allIds) {
    if (!previousNeighborIds.has(id) || !currentNeighborIds.has(id)) {
      churnCount++;
    }
  }
  const neighborhoodChurn =
    allIds.size > 0 ? churnCount / allIds.size : 0;

  // Curvature proxy: direction change in last 3 trajectory steps
  let curvatureProxy = 0;
  const traj = state.trajectory;
  if (traj.length >= 3) {
    const n2 = graph.nodes.get(traj[traj.length - 3]);
    const n1 = graph.nodes.get(traj[traj.length - 2]);
    const n0 = currentNode;
    if (n2 && n1) {
      // Use embedding space for direction
      const d1 = n1.embedding.map((v, i) => v - n2.embedding[i]);
      const d2 = n0.embedding.map((v, i) => v - n1.embedding[i]);
      const dot = d1.reduce((s, v, i) => s + v * d2[i], 0);
      const mag1 = Math.sqrt(d1.reduce((s, v) => s + v * v, 0));
      const mag2 = Math.sqrt(d2.reduce((s, v) => s + v * v, 0));
      if (mag1 > 1e-8 && mag2 > 1e-8) {
        const cosAngle = dot / (mag1 * mag2);
        curvatureProxy = 1 - Math.max(-1, Math.min(1, cosAngle));
      }
    }
  }

  // Constraint discontinuity
  const recentConstraints = state.constraintLedger.slice(-5);
  const resolvedRecent = recentConstraints.filter(c => c.resolved).length;
  const constraintDiscontinuity =
    recentConstraints.length > 0
      ? resolvedRecent / recentConstraints.length
      : 0;

  // Hue gradient: average interference between current node and neighbors
  let hueGradient = 0;
  let gradientCount = 0;
  for (const nid of currentNeighborIds) {
    const neighbor = graph.nodes.get(nid);
    if (neighbor) {
      hueGradient += hueInterference(currentNode.hue, neighbor.hue);
      gradientCount++;
    }
  }
  hueGradient = gradientCount > 0 ? hueGradient / gradientCount : 0;

  return {
    neighborhoodChurn,
    curvatureProxy,
    constraintDiscontinuity,
    hueGradient,
  };
}

/**
 * Gating mechanism: softmax over fold features.
 * Pr(spine=i | x_t) = softmax(W * f_fold(x_t))
 */
export function computeSpineGating(
  spines: Spine[],
  foldFeatures: FoldFeatures
): number[] {
  // Gating weights for each spine (Phase 1: 2 spines)
  // Dialectical negation activates on high churn, high curvature, high gradient
  // Pragmatic utility activates on low churn, low curvature, high constraint resolution
  const featureVec = [
    foldFeatures.neighborhoodChurn,
    foldFeatures.curvatureProxy,
    foldFeatures.constraintDiscontinuity,
    foldFeatures.hueGradient,
  ];

  const gatingWeights: number[][] = [
    // dialectical_negation: attracted to turbulence
    [2.0, 1.5, -1.0, 1.8],
    // pragmatic_utility: attracted to stability
    [-1.5, -1.0, 2.0, -0.5],
  ];

  const logits = gatingWeights.map(w =>
    w.reduce((sum, wi, i) => sum + wi * featureVec[i], 0)
  );

  // Softmax
  const maxLogit = Math.max(...logits);
  const expValues = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = expValues.reduce((s, v) => s + v, 0);

  return expValues.map(v => v / sumExp);
}

/**
 * Transition spine lifecycle based on navigation state.
 */
export function transitionSpineLifecycle(
  spine: Spine,
  isActive: boolean,
  hasSnapped: boolean,
  hasWitnessed: boolean
): SpineLifecycle {
  if (!isActive) return 'dormant';

  switch (spine.lifecycle) {
    case 'dormant':
      return 'waking';
    case 'waking':
      return 'mapping';
    case 'mapping':
      return hasSnapped ? 'snapping' : 'mapping';
    case 'snapping':
      return hasWitnessed ? 'witnessing' : 'snapping';
    case 'witnessing':
      return 'returning';
    case 'returning':
      return 'dormant';
    default:
      return 'dormant';
  }
}

/**
 * Update all spines with gating probabilities and lifecycle transitions.
 */
export function updateSpines(
  spines: Spine[],
  foldFeatures: FoldFeatures,
  activeSpineIndex: number,
  hasSnapped: boolean,
  hasWitnessed: boolean
): Spine[] {
  const activations = computeSpineGating(spines, foldFeatures);

  return spines.map((spine, i) => {
    const isActive = i === activeSpineIndex;
    const newLifecycle = transitionSpineLifecycle(
      spine,
      isActive,
      hasSnapped,
      hasWitnessed
    );

    return {
      ...spine,
      activation: activations[i],
      lifecycle: newLifecycle,
    };
  });
}

/**
 * Select the dominant spine based on gating probabilities.
 * Returns the index of the spine with highest activation.
 */
export function selectDominantSpine(spines: Spine[]): number {
  let maxActivation = -Infinity;
  let maxIdx = 0;
  for (let i = 0; i < spines.length; i++) {
    if (spines[i].activation > maxActivation) {
      maxActivation = spines[i].activation;
      maxIdx = i;
    }
  }
  return maxIdx;
}
