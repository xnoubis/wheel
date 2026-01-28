
import {
  NavigationState,
  SnapEvent,
  TerrainGraph,
  TerrainConfig,
  ConstraintEntry,
} from '../types';

/**
 * Snap Detector: Identifies nonlinear entropy reduction events.
 *
 * A snap is NOT semantic matching. It is a structural divergence event where:
 *   1. Sharp entropy reduction: delta_H > epsilon
 *   2. Stability for K steps: entropy stays low
 *   3. Eval gain: measurable improvement in evaluation suite
 *
 * Snap detector stages:
 *   1. State Delta: constraint-ledger distance D(Lambda_{t+1}, Lambda_t) > threshold
 *   2. Gain: measurable improvement in evaluation suite
 */

/** History buffer for entropy tracking */
export interface EntropyHistory {
  /** Entropy values over time */
  values: number[];
  /** Timestamps */
  timestamps: number[];
  /** Maximum buffer size */
  maxSize: number;
}

/** Evaluation suite result */
export interface EvalResult {
  /** Path efficiency: ratio of optimal to actual path length */
  pathEfficiency: number;
  /** Basin escape: number of unique basins visited */
  basinEscape: number;
  /** Trajectory diversity: unique nodes / total steps */
  trajectoryDiversity: number;
  /** Constraint resolution rate */
  constraintResolutionRate: number;
  /** Composite score */
  compositeScore: number;
}

/** Create a new entropy history buffer */
export function createEntropyHistory(maxSize: number = 100): EntropyHistory {
  return { values: [], timestamps: [], maxSize };
}

/** Record an entropy observation */
export function recordEntropy(
  history: EntropyHistory,
  entropy: number,
  timestamp: number
): void {
  history.values.push(entropy);
  history.timestamps.push(timestamp);
  if (history.values.length > history.maxSize) {
    history.values.shift();
    history.timestamps.shift();
  }
}

/**
 * Detect entropy drop: delta_H = H(A|S_t) - H(A|S_{t+1}).
 * Returns the drop magnitude (positive = entropy decreased).
 */
export function detectEntropyDrop(history: EntropyHistory): number {
  if (history.values.length < 2) return 0;
  const prev = history.values[history.values.length - 2];
  const curr = history.values[history.values.length - 1];
  return prev - curr; // Positive means entropy dropped
}

/**
 * Check stability: has entropy stayed below threshold for K consecutive steps?
 */
export function checkStability(
  history: EntropyHistory,
  K: number,
  threshold: number
): { stable: boolean; stableSteps: number } {
  if (history.values.length < K) {
    return { stable: false, stableSteps: 0 };
  }

  const recentValues = history.values.slice(-K);
  let stableSteps = 0;

  for (let i = recentValues.length - 1; i >= 0; i--) {
    if (recentValues[i] <= threshold) {
      stableSteps++;
    } else {
      break;
    }
  }

  return { stable: stableSteps >= K, stableSteps };
}

/**
 * Compute evaluation suite scores.
 * Success = system structure changes to yield new capability under constraint.
 */
export function computeEvaluation(state: NavigationState, graph: TerrainGraph): EvalResult {
  const trajectory = state.trajectory;

  // Path efficiency: unique nodes / total steps
  const uniqueNodes = new Set(trajectory);
  const pathEfficiency =
    trajectory.length > 0 ? uniqueNodes.size / trajectory.length : 0;

  // Basin escape: count distinct "regions" visited
  // Approximate by counting how many times trajectory direction changed significantly
  let directionChanges = 0;
  for (let i = 2; i < trajectory.length; i++) {
    if (trajectory[i] !== trajectory[i - 2] && trajectory[i - 1] !== trajectory[i]) {
      directionChanges++;
    }
  }
  const basinEscape = Math.min(1, directionChanges / Math.max(1, trajectory.length * 0.3));

  // Trajectory diversity
  const trajectoryDiversity = uniqueNodes.size / Math.max(1, graph.nodes.size);

  // Constraint resolution rate
  const totalConstraints = state.constraintLedger.length;
  const resolvedConstraints = state.constraintLedger.filter(c => c.resolved).length;
  const constraintResolutionRate =
    totalConstraints > 0 ? resolvedConstraints / totalConstraints : 1;

  // Composite score
  const compositeScore =
    0.25 * pathEfficiency +
    0.25 * basinEscape +
    0.25 * trajectoryDiversity +
    0.25 * constraintResolutionRate;

  return {
    pathEfficiency,
    basinEscape,
    trajectoryDiversity,
    constraintResolutionRate,
    compositeScore,
  };
}

/**
 * Compute constraint-ledger structural distance.
 * D(Lambda_{t+1}, Lambda_t) measuring the magnitude of structural change.
 */
export function constraintLedgerDistance(
  previous: ConstraintEntry[],
  current: ConstraintEntry[]
): number {
  // Count new constraints added
  const prevIds = new Set(previous.map(c => c.id));
  const newConstraints = current.filter(c => !prevIds.has(c.id)).length;

  // Count newly resolved constraints
  const prevResolved = new Set(
    previous.filter(c => c.resolved).map(c => c.id)
  );
  const newlyResolved = current.filter(
    c => c.resolved && !prevResolved.has(c.id)
  ).length;

  // Count severity changes
  let severityDelta = 0;
  for (const curr of current) {
    const prev = previous.find(p => p.id === curr.id);
    if (prev) {
      severityDelta += Math.abs(curr.severity - prev.severity);
    }
  }

  return newConstraints * 0.4 + newlyResolved * 0.4 + severityDelta * 0.2;
}

/** Accumulated snap detector state */
export interface SnapDetectorState {
  entropyHistory: EntropyHistory;
  previousEvalScore: number;
  previousConstraintLedger: ConstraintEntry[];
  snapEvents: SnapEvent[];
  /** Rolling eval scores for gain computation */
  evalHistory: number[];
  /** Whether a snap candidate is being tracked */
  candidateActive: boolean;
  candidateStartStep: number;
}

/** Create initial snap detector state */
export function createSnapDetectorState(): SnapDetectorState {
  return {
    entropyHistory: createEntropyHistory(100),
    previousEvalScore: 0,
    previousConstraintLedger: [],
    snapEvents: [],
    evalHistory: [],
    candidateActive: false,
    candidateStartStep: 0,
  };
}

/**
 * Run snap detection for the current step.
 * Returns a SnapEvent if snap conditions are met, null otherwise.
 */
export function detectSnap(
  detectorState: SnapDetectorState,
  navState: NavigationState,
  graph: TerrainGraph,
  config: TerrainConfig['snap'],
  time: number
): SnapEvent | null {
  // Record current entropy
  recordEntropy(detectorState.entropyHistory, navState.actionEntropy, time);

  // Stage 1: Entropy drop detection
  const entropyDrop = detectEntropyDrop(detectorState.entropyHistory);

  // Stage 2: Evaluation
  const evalResult = computeEvaluation(navState, graph);
  detectorState.evalHistory.push(evalResult.compositeScore);
  if (detectorState.evalHistory.length > 50) {
    detectorState.evalHistory.shift();
  }

  const evalGain = evalResult.compositeScore - detectorState.previousEvalScore;

  // Stage 3: Constraint ledger distance
  const ledgerDist = constraintLedgerDistance(
    detectorState.previousConstraintLedger,
    navState.constraintLedger
  );

  // Update previous state
  detectorState.previousEvalScore = evalResult.compositeScore;
  detectorState.previousConstraintLedger = [...navState.constraintLedger];

  // Check snap conditions: ALL must be true
  const entropyCondition = entropyDrop > config.epsilon;
  const evalCondition = evalGain > config.eta;

  if (entropyCondition && evalCondition) {
    if (!detectorState.candidateActive) {
      // Start tracking a candidate snap
      detectorState.candidateActive = true;
      detectorState.candidateStartStep = navState.step;
    }
  }

  // Check stability condition for active candidate
  if (detectorState.candidateActive) {
    const stabilityThreshold =
      detectorState.entropyHistory.values.length > 0
        ? Math.min(...detectorState.entropyHistory.values.slice(-config.K)) * 1.2
        : 1.0;

    const { stable, stableSteps } = checkStability(
      detectorState.entropyHistory,
      config.K,
      stabilityThreshold
    );

    if (stable) {
      // SNAP confirmed!
      detectorState.candidateActive = false;

      const snapEvent: SnapEvent = {
        delta_entropy: entropyDrop,
        stability_steps: stableSteps,
        eval_gain: evalGain,
        structural_delta: `ledger_dist=${ledgerDist.toFixed(4)};path_eff=${evalResult.pathEfficiency.toFixed(4)};constraints_resolved=${navState.constraintLedger.filter(c => c.resolved).length}`,
        timestamp: new Date(time).toISOString(),
      };

      detectorState.snapEvents.push(snapEvent);
      return snapEvent;
    }

    // Check if candidate has expired (too many steps without stability)
    if (navState.step - detectorState.candidateStartStep > config.K * 3) {
      detectorState.candidateActive = false;
    }
  }

  return null;
}

/**
 * Compute snap rate: snaps per N navigation steps.
 */
export function computeSnapRate(
  detectorState: SnapDetectorState,
  windowSteps: number
): number {
  if (detectorState.snapEvents.length === 0) return 0;
  const recentSnaps = detectorState.snapEvents.filter(
    s => Date.now() - new Date(s.timestamp).getTime() < windowSteps * 1000
  );
  return recentSnaps.length;
}
