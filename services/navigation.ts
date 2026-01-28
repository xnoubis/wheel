
import {
  TerrainNode,
  TerrainEdge,
  TerrainGraph,
  NavigationEnergy,
  NavigationState,
  ConstraintEntry,
  HueVector,
  NavigationBoid,
} from '../types';
import {
  pheromoneField,
  hueMagnitude,
  updateNodeHue,
  decayHue,
  encounterToHue,
  computeEncounterQuality,
  zeroHue,
  randomHue,
} from './hueField';

/**
 * Terrain Navigation: Constrained Gradient Descent.
 *
 * Navigation is a trajectory v_t -> v_{t+1} minimizing energy:
 *   E(v -> u | s_t) = alpha * Phi(u,t) + beta * C(u|s_t) + gamma * N(u,t) + eta * F(u,t)
 *
 * Selection: P(u|s_t) proportional to exp(-E(v_t -> u | s_t))
 */

let nodeIdCounter = 0;

/** Create a new terrain node */
export function createNode(
  label: string,
  nodeType: TerrainNode['nodeType'],
  embeddingDim: number
): TerrainNode {
  const id = `node_${++nodeIdCounter}_${Date.now()}`;
  const embedding: number[] = [];
  for (let i = 0; i < embeddingDim; i++) {
    embedding.push((Math.random() - 0.5) * 2);
  }
  return {
    id,
    embedding,
    hue: randomHue(0.05),
    lastVisited: 0,
    visitCount: 0,
    nodeType,
    label,
  };
}

/** Create an edge between nodes */
export function createEdge(
  nodeIds: string[],
  edgeType: TerrainEdge['edgeType'],
  weight: number = 1.0
): TerrainEdge {
  const id = `edge_${nodeIds.sort().join('_')}_${Date.now()}`;
  return {
    id,
    nodeIds,
    edgeType,
    hue: zeroHue(),
    weight,
  };
}

/** Initialize an empty terrain graph */
export function createTerrainGraph(): TerrainGraph {
  return {
    nodes: new Map(),
    edges: new Map(),
  };
}

/** Add a node to the terrain */
export function addNode(graph: TerrainGraph, node: TerrainNode): void {
  graph.nodes.set(node.id, node);
}

/** Add an edge to the terrain */
export function addEdge(graph: TerrainGraph, edge: TerrainEdge): void {
  graph.edges.set(edge.id, edge);
}

/** Get neighbors of a node (nodes connected by any edge) */
export function getNeighbors(
  graph: TerrainGraph,
  nodeId: string
): TerrainNode[] {
  const neighbors: TerrainNode[] = [];
  const seen = new Set<string>();

  for (const edge of graph.edges.values()) {
    if (edge.nodeIds.includes(nodeId)) {
      for (const nid of edge.nodeIds) {
        if (nid !== nodeId && !seen.has(nid)) {
          const node = graph.nodes.get(nid);
          if (node) {
            neighbors.push(node);
            seen.add(nid);
          }
        }
      }
    }
  }
  return neighbors;
}

/** Compute cosine distance between embeddings */
function embeddingDistance(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA * magB);
  if (denom < 1e-8) return 1;
  return 1 - dot / denom;
}

/**
 * Recompute edges from embeddings via approximate kNN.
 * This allows the topology to 'fold' as neighborhoods change.
 */
export function recomputeEdges(
  graph: TerrainGraph,
  k: number = 3
): void {
  const nodes = Array.from(graph.nodes.values());
  // Clear existing similarity edges
  for (const [id, edge] of graph.edges) {
    if (edge.edgeType === 'similarity') {
      graph.edges.delete(id);
    }
  }

  // Compute kNN for each node
  for (const node of nodes) {
    const distances = nodes
      .filter(n => n.id !== node.id)
      .map(n => ({
        node: n,
        dist: embeddingDistance(node.embedding, n.embedding),
      }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, k);

    for (const { node: neighbor, dist } of distances) {
      const edgeId = `sim_${[node.id, neighbor.id].sort().join('_')}`;
      if (!graph.edges.has(edgeId)) {
        graph.edges.set(edgeId, {
          id: edgeId,
          nodeIds: [node.id, neighbor.id],
          edgeType: 'similarity',
          hue: zeroHue(),
          weight: 1 - dist,
        });
      }
    }
  }
}

/**
 * Compute constraint pressure C(u|s_t).
 * Measures violations, boundary proximity, loop pressure.
 */
function constraintPressure(
  node: TerrainNode,
  state: NavigationState
): number {
  // Count unresolved constraints
  const unresolvedCount = state.constraintLedger.filter(
    c => !c.resolved
  ).length;

  // Loop pressure: how often this node appears in recent trajectory
  const recentTrajectory = state.trajectory.slice(-15);
  const loopCount = recentTrajectory.filter(id => id === node.id).length;

  // Boundary proximity: based on hue boundary-pressure axis
  const boundaryAxis = node.hue[0];

  return unresolvedCount * 0.3 + loopCount * 0.5 + Math.abs(boundaryAxis) * 0.2;
}

/**
 * Compute novelty term N(u, t).
 * Anti-collapse: encourages visiting unvisited or rarely-visited nodes.
 */
function noveltyTerm(node: TerrainNode, time: number): number {
  const timeSinceVisit = time - node.lastVisited;
  const visitDecay = 1 / (1 + node.visitCount);
  const recencyBonus = 1 - Math.exp(-timeSinceVisit / 5000);
  return -(visitDecay * recencyBonus); // Negative = attractive
}

/**
 * Compute friction F(u, t).
 * Models latency, cost, blocked transitions.
 */
function frictionTerm(
  node: TerrainNode,
  currentNode: TerrainNode,
  graph: TerrainGraph
): number {
  // Distance in embedding space = travel cost
  const dist = embeddingDistance(currentNode.embedding, node.embedding);

  // Check if there's a direct edge (lower friction)
  let hasDirectEdge = false;
  for (const edge of graph.edges.values()) {
    if (
      edge.nodeIds.includes(currentNode.id) &&
      edge.nodeIds.includes(node.id)
    ) {
      hasDirectEdge = true;
      break;
    }
  }

  return dist * (hasDirectEdge ? 0.5 : 1.5);
}

/**
 * Compute full navigation energy E(v -> u | s_t).
 */
export function computeNavigationEnergy(
  currentNode: TerrainNode,
  candidateNode: TerrainNode,
  state: NavigationState,
  graph: TerrainGraph,
  weights: [number, number, number, number],
  time: number
): NavigationEnergy {
  const [alpha, beta, gamma, eta] = weights;

  const phi = pheromoneField(candidateNode, time);
  const constraint = constraintPressure(candidateNode, state);
  const novelty = noveltyTerm(candidateNode, time);
  const friction = frictionTerm(candidateNode, currentNode, graph);

  const total =
    alpha * phi + beta * constraint + gamma * novelty + eta * friction;

  return { phi, constraint, novelty, friction, total };
}

/**
 * Select next node using softmax over negative energies.
 * P(u|s_t) proportional to exp(-E(v_t -> u | s_t))
 */
export function selectNextNode(
  currentNode: TerrainNode,
  candidates: TerrainNode[],
  state: NavigationState,
  graph: TerrainGraph,
  weights: [number, number, number, number],
  time: number
): { node: TerrainNode; energy: NavigationEnergy; probability: number } | null {
  if (candidates.length === 0) return null;

  const energies = candidates.map(c => ({
    node: c,
    energy: computeNavigationEnergy(currentNode, c, state, graph, weights, time),
  }));

  // Softmax over negative energies
  const negEnergies = energies.map(e => -e.energy.total);
  const maxNegE = Math.max(...negEnergies);
  const expValues = negEnergies.map(e => Math.exp(e - maxNegE));
  const sumExp = expValues.reduce((s, v) => s + v, 0);
  const probabilities = expValues.map(v => v / sumExp);

  // Sample from distribution
  let r = Math.random();
  let selectedIdx = probabilities.length - 1;
  for (let i = 0; i < probabilities.length; i++) {
    r -= probabilities[i];
    if (r <= 0) {
      selectedIdx = i;
      break;
    }
  }

  return {
    node: energies[selectedIdx].node,
    energy: energies[selectedIdx].energy,
    probability: probabilities[selectedIdx],
  };
}

/**
 * Compute action distribution entropy H(A|S_t).
 * Used for snap detection.
 */
export function computeActionEntropy(
  currentNode: TerrainNode,
  candidates: TerrainNode[],
  state: NavigationState,
  graph: TerrainGraph,
  weights: [number, number, number, number],
  time: number
): number {
  if (candidates.length <= 1) return 0;

  const energies = candidates.map(c =>
    computeNavigationEnergy(currentNode, c, state, graph, weights, time)
  );

  const negEnergies = energies.map(e => -e.total);
  const maxNegE = Math.max(...negEnergies);
  const expValues = negEnergies.map(e => Math.exp(e - maxNegE));
  const sumExp = expValues.reduce((s, v) => s + v, 0);
  const probabilities = expValues.map(v => v / sumExp);

  // Shannon entropy
  let entropy = 0;
  for (const p of probabilities) {
    if (p > 1e-10) {
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

/** Create initial navigation state */
export function createNavigationState(startNodeId: string): NavigationState {
  return {
    currentNodeId: startNodeId,
    trajectory: [startNodeId],
    step: 0,
    actionEntropy: 0,
    constraintLedger: [],
    activeSpine: 0,
  };
}

/**
 * Execute one navigation step.
 * Returns updated state and the energy of the transition.
 */
export function navigateStep(
  state: NavigationState,
  graph: TerrainGraph,
  config: {
    weights: [number, number, number, number];
    lambda: number;
    delta: number;
  },
  time: number
): {
  newState: NavigationState;
  selectedNode: TerrainNode | null;
  energy: NavigationEnergy | null;
} {
  const currentNode = graph.nodes.get(state.currentNodeId);
  if (!currentNode) {
    return { newState: state, selectedNode: null, energy: null };
  }

  const neighbors = getNeighbors(graph, state.currentNodeId);
  if (neighbors.length === 0) {
    return { newState: state, selectedNode: null, energy: null };
  }

  // Compute action entropy before selection
  const actionEntropy = computeActionEntropy(
    currentNode,
    neighbors,
    state,
    graph,
    config.weights,
    time
  );

  // Select next node
  const selection = selectNextNode(
    currentNode,
    neighbors,
    state,
    graph,
    config.weights,
    time
  );

  if (!selection) {
    return { newState: { ...state, actionEntropy }, selectedNode: null, energy: null };
  }

  const { node: nextNode, energy } = selection;

  // Update hue at visited node
  const encounter = computeEncounterQuality(
    nextNode,
    state.trajectory,
    state.constraintLedger.filter(c => !c.resolved).length,
    neighbors.length,
    graph.nodes.size
  );
  nextNode.hue = updateNodeHue(
    nextNode.hue,
    encounterToHue(encounter),
    config.lambda
  );
  nextNode.lastVisited = time;
  nextNode.visitCount++;

  // Update edge hue along the traversal
  for (const edge of graph.edges.values()) {
    if (
      edge.nodeIds.includes(currentNode.id) &&
      edge.nodeIds.includes(nextNode.id)
    ) {
      edge.hue = updateNodeHue(
        edge.hue,
        encounterToHue(encounter),
        config.lambda * 0.5
      );
    }
  }

  const newState: NavigationState = {
    ...state,
    currentNodeId: nextNode.id,
    trajectory: [...state.trajectory, nextNode.id],
    step: state.step + 1,
    actionEntropy,
  };

  return { newState, selectedNode: nextNode, energy };
}

/**
 * Initialize a Boids swarm for visualization of navigation.
 */
export function initNavigationBoids(count: number, width: number, height: number): NavigationBoid[] {
  const boids: NavigationBoid[] = [];
  for (let i = 0; i < count; i++) {
    boids.push({
      id: i,
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
      separation: 0.5 + Math.random() * 0.5,
      alignment: 0.3 + Math.random() * 0.4,
      cohesion: 0.4 + Math.random() * 0.3,
      nearestNodeId: null,
    });
  }
  return boids;
}

/**
 * Update Boids with terrain-aware forces.
 * Three rules from the Mnemic Substrate:
 *   Separation: Maintain distinct conceptual identities
 *   Alignment: Match prevailing semantic current
 *   Cohesion: Steer toward proven slime mold paths
 */
export function updateNavigationBoids(
  boids: NavigationBoid[],
  hueField: { x: number; y: number; magnitude: number }[],
  target: { x: number; y: number },
  isActive: boolean,
  width: number,
  height: number
): void {
  const perceptionRadius = 60;
  const maxSpeed = isActive ? 4.0 : 1.5;

  for (const boid of boids) {
    let sepX = 0, sepY = 0;
    let aliX = 0, aliY = 0;
    let cohX = 0, cohY = 0;
    let neighborCount = 0;

    // Boid-to-boid forces
    for (const other of boids) {
      if (other.id === boid.id) continue;
      const dx = other.x - boid.x;
      const dy = other.y - boid.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < perceptionRadius && dist > 0) {
        neighborCount++;
        // Separation: don't collapse into bias
        sepX -= dx / (dist * dist);
        sepY -= dy / (dist * dist);
        // Alignment: match velocity
        aliX += other.vx;
        aliY += other.vy;
        // Cohesion: steer to center
        cohX += other.x;
        cohY += other.y;
      }
    }

    let ax = 0, ay = 0;

    if (neighborCount > 0) {
      // Separation
      ax += sepX * boid.separation * 2.0;
      ay += sepY * boid.separation * 2.0;
      // Alignment
      aliX /= neighborCount;
      aliY /= neighborCount;
      ax += (aliX - boid.vx) * boid.alignment * 0.1;
      ay += (aliY - boid.vy) * boid.alignment * 0.1;
      // Cohesion toward group center
      cohX = cohX / neighborCount - boid.x;
      cohY = cohY / neighborCount - boid.y;
      ax += cohX * boid.cohesion * 0.01;
      ay += cohY * boid.cohesion * 0.01;
    }

    // Pheromone attraction: steer toward high-magnitude hue field regions
    for (const point of hueField) {
      const dx = point.x - boid.x;
      const dy = point.y - boid.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 80 && dist > 5) {
        const attraction = point.magnitude / (dist * 0.5);
        ax += (dx / dist) * attraction * 0.3;
        ay += (dy / dist) * attraction * 0.3;
      }
    }

    // Centripetal toward target
    const tdx = target.x - boid.x;
    const tdy = target.y - boid.y;
    ax += tdx * 0.0005;
    ay += tdy * 0.0005;

    // Jitter
    ax += (Math.random() - 0.5) * (isActive ? 0.8 : 0.15);
    ay += (Math.random() - 0.5) * (isActive ? 0.8 : 0.15);

    boid.vx += ax;
    boid.vy += ay;

    // Speed limit
    const speed = Math.sqrt(boid.vx * boid.vx + boid.vy * boid.vy);
    if (speed > maxSpeed) {
      boid.vx = (boid.vx / speed) * maxSpeed;
      boid.vy = (boid.vy / speed) * maxSpeed;
    }

    boid.x += boid.vx;
    boid.y += boid.vy;

    // Toroidal wrap
    if (boid.x < 0) boid.x += width;
    if (boid.x > width) boid.x -= width;
    if (boid.y < 0) boid.y += height;
    if (boid.y > height) boid.y -= height;
  }
}
