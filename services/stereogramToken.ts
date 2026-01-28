
import {
  OperatorSpec,
  StereogramToken,
  ReplicationLog,
  SnapEvent,
  NavigationState,
  TerrainGraph,
  Spine,
} from '../types';

/**
 * Stereogram Tokens: Zero-Knowledge Co-instantiation Proofs.
 *
 * Tokens prove:
 *   - A specific operator (O) was instantiated under constraints
 *   - Passed a test suite with measured scores
 *   - Without revealing raw content/transcript
 *
 * Operator spec: O := (C, P, V, R, T)
 *   C: constraint spec
 *   P: policy/spine config
 *   V: visibility regime
 *   R: randomness regime
 *   T: evaluation suite
 *
 * MVP: commitment + selective disclosure + signatures (SNARK later)
 */

/**
 * SHA-256 hash using Web Crypto API.
 * All hashes are hex-encoded SHA-256 per spec.
 */
async function sha256Hex(data: string): Promise<string> {
  const encoder = new TextEncoder();
  const buffer = encoder.encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * HMAC-SHA256 for platform signatures.
 * In MVP, this uses a deterministic key derived from operator+timestamp.
 * In production, this would be a proper signing key.
 */
async function hmacSha256Hex(key: string, data: string): Promise<string> {
  const encoder = new TextEncoder();
  const keyData = encoder.encode(key);
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    keyData,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const signature = await crypto.subtle.sign(
    'HMAC',
    cryptoKey,
    encoder.encode(data)
  );
  const sigArray = Array.from(new Uint8Array(signature));
  return sigArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Build the operator specification from current system state.
 */
export async function buildOperatorSpec(
  constraints: string,
  spine: Spine,
  visibility: string,
  randomSeed: string,
  testSuiteId: string
): Promise<OperatorSpec> {
  const canonicalString = [
    constraints,
    `${spine.id}:${spine.policy}`,
    visibility,
    randomSeed,
    testSuiteId,
  ].join('||');

  const operatorHash = await sha256Hex(canonicalString);

  return {
    operator_hash: operatorHash,
    constraints,
    policy: `${spine.id}:${spine.lifecycle}:${spine.activation.toFixed(4)}`,
    visibility,
    randomness: randomSeed,
    test_suite: testSuiteId,
  };
}

/**
 * Compute terrain hash from current graph state.
 * Hash of node IDs + hue magnitudes + edge structure.
 */
export async function computeTerrainHash(
  graph: TerrainGraph
): Promise<string> {
  const nodeEntries = Array.from(graph.nodes.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([id, node]) => `${id}:${node.hue.join(',')}`);

  const edgeEntries = Array.from(graph.edges.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([id, edge]) => `${id}:${edge.nodeIds.join('+')}:${edge.weight}`);

  const data = [...nodeEntries, '---', ...edgeEntries].join('\n');
  return sha256Hex(data);
}

/**
 * Compute transcript commitment root (Merkle root of trajectory).
 */
export async function computeTranscriptCommitment(
  trajectory: string[]
): Promise<string> {
  if (trajectory.length === 0) return sha256Hex('empty_transcript');

  // Build leaf hashes
  const leaves = await Promise.all(
    trajectory.map((nodeId, i) => sha256Hex(`${i}:${nodeId}`))
  );

  // Simple binary Merkle tree
  let currentLevel = leaves;
  while (currentLevel.length > 1) {
    const nextLevel: string[] = [];
    for (let i = 0; i < currentLevel.length; i += 2) {
      if (i + 1 < currentLevel.length) {
        nextLevel.push(
          await sha256Hex(currentLevel[i] + currentLevel[i + 1])
        );
      } else {
        nextLevel.push(currentLevel[i]);
      }
    }
    currentLevel = nextLevel;
  }

  return currentLevel[0];
}

/**
 * Run replication trials for reproducibility attestation.
 * Simulates N trials of the evaluation suite and records pass/fail.
 */
export function runReplicationTrials(
  evalScores: number[],
  threshold: number,
  nTrials: number
): ReplicationLog {
  let passed = 0;
  let failed = 0;

  for (let i = 0; i < nTrials; i++) {
    // Select a score with bootstrap sampling
    const idx = Math.floor(Math.random() * evalScores.length);
    const score = evalScores[idx];
    if (score >= threshold) {
      passed++;
    } else {
      failed++;
    }
  }

  const passRate = passed / nTrials;
  let reproducibilityClass: string;
  if (passRate >= 0.95) reproducibilityClass = 'highly_reproducible';
  else if (passRate >= 0.80) reproducibilityClass = 'reproducible';
  else if (passRate >= 0.60) reproducibilityClass = 'partially_reproducible';
  else reproducibilityClass = 'non_reproducible';

  return {
    trials: nTrials,
    pass_fail: { passed, failed },
    reproducibility_class: reproducibilityClass,
  };
}

/**
 * Mint a Stereogram Token.
 * This is the core attestation function.
 */
export async function mintStereogramToken(
  operatorSpec: OperatorSpec,
  graph: TerrainGraph,
  navState: NavigationState,
  evalScores: number[],
  snapEvent: SnapEvent | null,
  platformKey: string = 'rosetta_platform_v0.2'
): Promise<StereogramToken> {
  const timestamp = new Date().toISOString();

  // Compute commitments
  const terrainHash = await computeTerrainHash(graph);
  const transcriptRoot = await computeTranscriptCommitment(
    navState.trajectory
  );

  // Platform signature: HMAC over operator_hash + scores + timestamp
  const signatureData = [
    operatorSpec.operator_hash,
    JSON.stringify(evalScores),
    timestamp,
    snapEvent ? snapEvent.delta_entropy.toString() : 'no_snap',
  ].join('|');
  const platformSignature = await hmacSha256Hex(platformKey, signatureData);

  // Replication log
  const replicationLog = runReplicationTrials(
    evalScores,
    0.5, // threshold
    Math.max(10, evalScores.length * 3) // at least 10 trials
  );

  return {
    operator_hash: operatorSpec.operator_hash,
    terrain_hash: terrainHash,
    transcript_commitment_root: transcriptRoot,
    platform_signature: platformSignature,
    user_signature: null, // User signs separately if desired
    scores: evalScores,
    timestamp,
    replication_log: replicationLog,
  };
}

/**
 * Verify a stereogram token's internal consistency.
 * Checks that hashes and signatures are well-formed.
 */
export function verifyTokenStructure(token: StereogramToken): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  // Check required fields
  if (!token.operator_hash || token.operator_hash.length !== 64) {
    errors.push('Invalid operator_hash: must be 64-char hex SHA-256');
  }
  if (!token.platform_signature || token.platform_signature.length !== 64) {
    errors.push('Invalid platform_signature: must be 64-char hex');
  }
  if (!token.timestamp) {
    errors.push('Missing timestamp');
  } else {
    const parsed = Date.parse(token.timestamp);
    if (isNaN(parsed)) {
      errors.push('Invalid timestamp: must be ISO 8601');
    }
  }
  if (!Array.isArray(token.scores) || token.scores.length === 0) {
    errors.push('Scores must be a non-empty array');
  }
  if (!token.replication_log) {
    errors.push('Missing replication_log');
  } else {
    if (token.replication_log.trials <= 0) {
      errors.push('Replication trials must be > 0');
    }
  }

  // Validate optional hashes if present
  if (token.terrain_hash && token.terrain_hash.length !== 64) {
    errors.push('Invalid terrain_hash: must be 64-char hex SHA-256');
  }
  if (
    token.transcript_commitment_root &&
    token.transcript_commitment_root.length !== 64
  ) {
    errors.push('Invalid transcript_commitment_root: must be 64-char hex SHA-256');
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Create a selective disclosure view of a token.
 * Reveals only the fields specified, redacting others.
 */
export function selectiveDisclosure(
  token: StereogramToken,
  revealFields: (keyof StereogramToken)[]
): Partial<StereogramToken> & { _redacted: string[] } {
  const disclosed: any = { _redacted: [] };

  for (const key of Object.keys(token) as (keyof StereogramToken)[]) {
    if (revealFields.includes(key)) {
      disclosed[key] = token[key];
    } else {
      disclosed._redacted.push(key);
    }
  }

  return disclosed;
}
