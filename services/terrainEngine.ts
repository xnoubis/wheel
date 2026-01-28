
import {
  TerrainGraph,
  TerrainNode,
  TerrainConfig,
  NavigationState,
  SnapEvent,
  StereogramToken,
  OperatorSpec,
  Spine,
  HueVector,
} from '../types';
import {
  decayAllNodes,
  decayAllEdges,
  computeHueFieldStats,
  HueFieldStats,
} from './hueField';
import {
  createTerrainGraph,
  createNode,
  createEdge,
  addNode,
  addEdge,
  recomputeEdges,
  createNavigationState,
  navigateStep,
  getNeighbors,
} from './navigation';
import {
  createPhase1Spines,
  computeFoldFeatures,
  updateSpines,
  selectDominantSpine,
  FoldFeatures,
} from './spineGating';
import {
  createSnapDetectorState,
  detectSnap,
  computeEvaluation,
  SnapDetectorState,
  EvalResult,
} from './snapDetector';
import {
  buildOperatorSpec,
  mintStereogramToken,
} from './stereogramToken';

/** Default Phase 1 configuration */
export const DEFAULT_TERRAIN_CONFIG: TerrainConfig = {
  lambda: 0.3,      // Hue update learning rate
  delta: 0.02,      // Global decay rate
  energyWeights: [1.0, 1.5, 0.8, 0.5], // [alpha, beta, gamma, eta]
  snap: {
    epsilon: 0.3,   // Min entropy reduction for snap
    K: 3,           // Min stable steps
    eta: 0.05,      // Min eval gain
  },
  embeddingDim: 8,  // Latent dimension
  hueK: 6,          // Number of hue axes
};

/** Full engine state */
export interface TerrainEngineState {
  graph: TerrainGraph;
  navState: NavigationState;
  spines: Spine[];
  snapDetector: SnapDetectorState;
  config: TerrainConfig;
  /** All snap events detected */
  snapEvents: SnapEvent[];
  /** All minted tokens */
  tokens: StereogramToken[];
  /** Current fold features */
  foldFeatures: FoldFeatures;
  /** Previous neighbor IDs for churn detection */
  previousNeighborIds: Set<string>;
  /** Current evaluation result */
  evalResult: EvalResult | null;
  /** Hue field statistics */
  hueStats: HueFieldStats;
  /** Step counter */
  totalSteps: number;
  /** Whether the engine is running */
  running: boolean;
}

/**
 * Initialize the terrain engine with a seed topology.
 * Phase 1: Creates a toy terrain with initial nodes and edges.
 */
export function initializeEngine(
  config: TerrainConfig = DEFAULT_TERRAIN_CONFIG,
  seedLabels: string[] = [
    'potentiality_core',
    'relational_manifold',
    'transmodal_integument',
    'mnemic_substrate',
    'constraint_boundary_alpha',
    'constraint_boundary_beta',
    'dialectical_basin',
    'pragmatic_basin',
    'novelty_frontier',
    'coherence_attractor',
    'loop_trap',
    'boundary_crossing',
  ]
): TerrainEngineState {
  const graph = createTerrainGraph();

  // Create seed nodes
  const nodes: TerrainNode[] = seedLabels.map(label => {
    const nodeType: TerrainNode['nodeType'] =
      label.includes('constraint') ? 'constraint' :
      label.includes('boundary') || label.includes('frontier') ? 'constraint' :
      label.includes('basin') || label.includes('attractor') || label.includes('trap') ? 'concept' :
      'conversation_state';
    return createNode(label, nodeType, config.embeddingDim);
  });

  for (const node of nodes) {
    addNode(graph, node);
  }

  // Create initial constraint edges
  const nodeIds = nodes.map(n => n.id);
  // Connect sequential nodes for basic structure
  for (let i = 0; i < nodeIds.length - 1; i++) {
    addEdge(graph, createEdge(
      [nodeIds[i], nodeIds[i + 1]],
      'constraint',
      0.8
    ));
  }
  // Close the loop
  addEdge(graph, createEdge(
    [nodeIds[nodeIds.length - 1], nodeIds[0]],
    'provenance',
    0.5
  ));

  // Add cross-connections for richer topology
  addEdge(graph, createEdge([nodeIds[0], nodeIds[3]], 'witness', 0.7));
  addEdge(graph, createEdge([nodeIds[1], nodeIds[6]], 'similarity', 0.6));
  addEdge(graph, createEdge([nodeIds[2], nodeIds[8]], 'similarity', 0.5));
  addEdge(graph, createEdge([nodeIds[4], nodeIds[5]], 'constraint', 0.9));
  addEdge(graph, createEdge([nodeIds[6], nodeIds[7]], 'witness', 0.6));

  // Recompute similarity edges from embeddings
  recomputeEdges(graph, 3);

  // Initialize navigation at the potentiality core
  const startNode = nodes[0];
  const navState = createNavigationState(startNode.id);

  // Initialize spines
  const spines = createPhase1Spines();

  return {
    graph,
    navState,
    spines,
    snapDetector: createSnapDetectorState(),
    config,
    snapEvents: [],
    tokens: [],
    foldFeatures: {
      neighborhoodChurn: 0,
      curvatureProxy: 0,
      constraintDiscontinuity: 0,
      hueGradient: 0,
    },
    previousNeighborIds: new Set(),
    evalResult: null,
    hueStats: computeHueFieldStats(graph.nodes),
    totalSteps: 0,
    running: false,
  };
}

/**
 * Execute one full engine tick:
 *   1. Compute fold features & update spine gating
 *   2. Navigate one step via constrained gradient descent
 *   3. Apply global hue decay
 *   4. Run snap detection
 *   5. Mint token if snap detected
 *   6. Update statistics
 */
export async function engineTick(
  state: TerrainEngineState
): Promise<{
  state: TerrainEngineState;
  snapEvent: SnapEvent | null;
  token: StereogramToken | null;
}> {
  const time = Date.now();
  const { graph, navState, spines, snapDetector, config } = state;

  // 1. Compute fold features and update spine gating
  const foldFeatures = computeFoldFeatures(
    graph,
    navState,
    state.previousNeighborIds
  );

  // Track current neighbors for next step's churn detection
  const currentNeighborIds = new Set<string>();
  for (const edge of graph.edges.values()) {
    if (edge.nodeIds.includes(navState.currentNodeId)) {
      for (const nid of edge.nodeIds) {
        if (nid !== navState.currentNodeId) {
          currentNeighborIds.add(nid);
        }
      }
    }
  }

  // Determine if snap/witness occurred for lifecycle transitions
  const hasSnapped = state.snapEvents.length > 0;
  const hasWitnessed = state.tokens.length > 0;

  const updatedSpines = updateSpines(
    spines,
    foldFeatures,
    navState.activeSpine,
    hasSnapped,
    hasWitnessed
  );

  // Select dominant spine
  const dominantSpineIdx = selectDominantSpine(updatedSpines);

  // 2. Navigate one step
  const { newState, selectedNode, energy } = navigateStep(
    { ...navState, activeSpine: dominantSpineIdx },
    graph,
    {
      weights: config.energyWeights,
      lambda: config.lambda,
      delta: config.delta,
    },
    time
  );

  // 3. Apply global hue decay
  decayAllNodes(graph.nodes, config.delta);
  decayAllEdges(graph.edges, config.delta);

  // 4. Run snap detection
  const snapEvent = detectSnap(snapDetector, newState, graph, config.snap, time);

  // 5. Mint token if snap detected
  let token: StereogramToken | null = null;
  if (snapEvent) {
    const activeSpine = updatedSpines[dominantSpineIdx];
    const operatorSpec = await buildOperatorSpec(
      `phase1_terrain:${graph.nodes.size}nodes:${graph.edges.size}edges`,
      activeSpine,
      'local_gradient',
      `seed_${time}`,
      'phase1_eval_suite_v1'
    );

    const evalResult = computeEvaluation(newState, graph);
    const evalScores = [
      evalResult.pathEfficiency,
      evalResult.basinEscape,
      evalResult.trajectoryDiversity,
      evalResult.constraintResolutionRate,
      evalResult.compositeScore,
    ];

    token = await mintStereogramToken(
      operatorSpec,
      graph,
      newState,
      evalScores,
      snapEvent
    );
  }

  // 6. Periodically recompute topology
  const totalSteps = state.totalSteps + 1;
  if (totalSteps % 20 === 0) {
    recomputeEdges(graph, 3);
  }

  // Add constraint entries from navigation dynamics
  if (selectedNode && energy) {
    // If high constraint pressure, add a constraint entry
    if (energy.constraint > 1.5) {
      newState.constraintLedger.push({
        id: `constraint_${totalSteps}_${time}`,
        type: energy.friction > 1.0 ? 'boundary' : 'coherence',
        severity: energy.constraint,
        resolved: false,
        timestamp: time,
      });
    }
    // Resolve constraints when novelty leads to progress
    if (energy.novelty < -0.5 && newState.constraintLedger.length > 0) {
      const unresolved = newState.constraintLedger.filter(c => !c.resolved);
      if (unresolved.length > 0) {
        unresolved[0].resolved = true;
      }
    }
  }

  // Update statistics
  const hueStats = computeHueFieldStats(graph.nodes);
  const evalResult = computeEvaluation(newState, graph);

  const newEngineState: TerrainEngineState = {
    ...state,
    graph,
    navState: newState,
    spines: updatedSpines,
    snapDetector,
    snapEvents: snapEvent
      ? [...state.snapEvents, snapEvent]
      : state.snapEvents,
    tokens: token ? [...state.tokens, token] : state.tokens,
    foldFeatures,
    previousNeighborIds: currentNeighborIds,
    evalResult,
    hueStats,
    totalSteps,
  };

  return { state: newEngineState, snapEvent, token };
}

/**
 * Add a new node to the terrain dynamically (e.g., from user interaction).
 */
export function injectNode(
  state: TerrainEngineState,
  label: string,
  nodeType: TerrainNode['nodeType'] = 'conversation_state'
): TerrainEngineState {
  const node = createNode(label, nodeType, state.config.embeddingDim);
  addNode(state.graph, node);

  // Connect to nearest existing nodes
  recomputeEdges(state.graph, 3);

  return { ...state, hueStats: computeHueFieldStats(state.graph.nodes) };
}

/**
 * Get a summary of the engine state for display.
 */
export interface EngineSummary {
  totalNodes: number;
  totalEdges: number;
  totalSteps: number;
  currentNode: string;
  trajectoryLength: number;
  activeSpine: string;
  spineLifecycle: string;
  spineActivations: { name: string; activation: number }[];
  actionEntropy: number;
  snapCount: number;
  tokenCount: number;
  hueStats: HueFieldStats;
  evalResult: EvalResult | null;
  foldFeatures: FoldFeatures;
  unresolvedConstraints: number;
}

export function getEngineSummary(state: TerrainEngineState): EngineSummary {
  const activeSpine = state.spines[state.navState.activeSpine] || state.spines[0];
  const currentNode = state.graph.nodes.get(state.navState.currentNodeId);

  return {
    totalNodes: state.graph.nodes.size,
    totalEdges: state.graph.edges.size,
    totalSteps: state.totalSteps,
    currentNode: currentNode?.label || 'unknown',
    trajectoryLength: state.navState.trajectory.length,
    activeSpine: activeSpine.name,
    spineLifecycle: activeSpine.lifecycle,
    spineActivations: state.spines.map(s => ({
      name: s.name,
      activation: s.activation,
    })),
    actionEntropy: state.navState.actionEntropy,
    snapCount: state.snapEvents.length,
    tokenCount: state.tokens.length,
    hueStats: state.hueStats,
    evalResult: state.evalResult,
    foldFeatures: state.foldFeatures,
    unresolvedConstraints: state.navState.constraintLedger.filter(
      c => !c.resolved
    ).length,
  };
}
