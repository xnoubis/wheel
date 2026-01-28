
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

// ─── Rosetta Terrain Engine Types (v0.2) ───────────────────────────

/** Hue axes: boundary-pressure, loopiness, novelty, coherence, risk, dispersion */
export const HUE_AXES = [
  'boundary_pressure',
  'loopiness',
  'novelty',
  'coherence',
  'risk',
  'dispersion',
] as const;

export type HueAxis = typeof HUE_AXES[number];

/** k-dimensional hue vector (k=6 for Phase 1) */
export type HueVector = [number, number, number, number, number, number];

/** A node in the terrain hyper-graph */
export interface TerrainNode {
  id: string;
  /** Latent embedding coordinates */
  embedding: number[];
  /** Multi-dimensional hue field at this node */
  hue: HueVector;
  /** When this node was last visited */
  lastVisited: number;
  /** Visit count for frequency tracking */
  visitCount: number;
  /** Metadata: type of content this node represents */
  nodeType: 'chunk' | 'concept' | 'conversation_state' | 'constraint';
  /** Human-readable label */
  label: string;
}

/** A hyper-edge relating k >= 2 nodes */
export interface TerrainEdge {
  id: string;
  /** Node IDs this edge connects */
  nodeIds: string[];
  /** Edge type */
  edgeType: 'similarity' | 'constraint' | 'witness' | 'provenance';
  /** Edge-level hue (pheromone on the connection) */
  hue: HueVector;
  /** Weight / strength of connection */
  weight: number;
}

/** The full terrain graph */
export interface TerrainGraph {
  nodes: Map<string, TerrainNode>;
  edges: Map<string, TerrainEdge>;
}

/** Navigation energy components */
export interface NavigationEnergy {
  /** Pheromone/hue bias */
  phi: number;
  /** Constraint pressure */
  constraint: number;
  /** Novelty term (anti-collapse) */
  novelty: number;
  /** Friction (latency/cost) */
  friction: number;
  /** Total energy */
  total: number;
}

/** Navigation state at time t */
export interface NavigationState {
  /** Current node */
  currentNodeId: string;
  /** Trajectory history (node IDs) */
  trajectory: string[];
  /** Current step */
  step: number;
  /** Action distribution entropy H(A|S_t) */
  actionEntropy: number;
  /** Accumulated constraint ledger */
  constraintLedger: ConstraintEntry[];
  /** Active spine index */
  activeSpine: number;
}

/** A constraint in the ledger */
export interface ConstraintEntry {
  id: string;
  type: 'boundary' | 'loop' | 'coherence' | 'refusal';
  severity: number;
  resolved: boolean;
  timestamp: number;
}

/** Spine lifecycle states */
export type SpineLifecycle = 'dormant' | 'waking' | 'mapping' | 'snapping' | 'witnessing' | 'returning';

/** A spine (expert policy) in the gated MoE */
export interface Spine {
  id: string;
  name: string;
  /** Current lifecycle state */
  lifecycle: SpineLifecycle;
  /** Policy description */
  policy: string;
  /** Refusal surface description */
  refusalSurface: string;
  /** Witness type */
  witnessType: string;
  /** Activation probability from gating */
  activation: number;
  /** Accumulated score from evaluations */
  score: number;
}

/** Quality-of-encounter vector for hue updates */
export interface EncounterQuality {
  boundaryPressure: number;
  loopiness: number;
  novelty: number;
  coherence: number;
  risk: number;
  dispersion: number;
}

/** Snap event: nonlinear entropy reduction */
export interface SnapEvent {
  /** Entropy change: H(A|S_t) - H(A|S_{t+1}) */
  delta_entropy: number;
  /** Number of stable steps maintained */
  stability_steps: number;
  /** Evaluation gain */
  eval_gain: number;
  /** Description of structural change */
  structural_delta: string;
  /** ISO 8601 timestamp */
  timestamp: string;
}

/** Operator specification O := (C, P, V, R, T) */
export interface OperatorSpec {
  /** Operator hash (SHA-256 hex) */
  operator_hash: string;
  /** Constraint spec */
  constraints: string;
  /** Policy/spine configuration */
  policy: string;
  /** Visibility regime */
  visibility: string;
  /** Randomness regime */
  randomness: string;
  /** Test suite ID/description */
  test_suite: string;
}

/** Replication log for stereogram tokens */
export interface ReplicationLog {
  trials: number;
  pass_fail: { passed: number; failed: number };
  reproducibility_class: string;
}

/** Stereogram token: ZK co-instantiation proof */
export interface StereogramToken {
  /** Hash of the operator spec */
  operator_hash: string;
  /** Hash of terrain state (optional) */
  terrain_hash: string | null;
  /** Merkle root of transcript commitments (optional) */
  transcript_commitment_root: string | null;
  /** Platform signature (hex) */
  platform_signature: string;
  /** User signature (optional, hex) */
  user_signature: string | null;
  /** Test suite scores */
  scores: number[];
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Replication log */
  replication_log: ReplicationLog;
}

/** Terrain engine configuration */
export interface TerrainConfig {
  /** Hue update learning rate */
  lambda: number;
  /** Global decay rate */
  delta: number;
  /** Energy weights [alpha, beta, gamma, eta] */
  energyWeights: [number, number, number, number];
  /** Snap detector thresholds */
  snap: {
    /** Minimum entropy reduction for snap */
    epsilon: number;
    /** Minimum stable steps */
    K: number;
    /** Minimum eval gain */
    eta: number;
  };
  /** Latent dimension */
  embeddingDim: number;
  /** Number of hue axes */
  hueK: number;
}

/** Boid for swarm navigation */
export interface NavigationBoid {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** Current conceptual identity strength (separation) */
  separation: number;
  /** Alignment with semantic current */
  alignment: number;
  /** Cohesion toward proven paths */
  cohesion: number;
  /** Which node this boid is currently near */
  nearestNodeId: string | null;
}
