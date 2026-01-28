"""
Rosetta Phase 2 Engine
======================
Corpus-as-terrain navigation with 768-dim embeddings.
Dragon Curve fractal exploration, snap detection, token minting.

Built to pass verify_phase2.py spec.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ============================================================================
# EMBEDDING SIMULATION (Replace with real encoder in production)
# ============================================================================

def compute_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Simulate embedding computation.
    In production: use sentence-transformers or similar.

    For now: deterministic pseudo-embedding based on text hash.
    """
    # Create deterministic seed from text
    text_hash = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(text_hash[:4], 'big')
    rng = random.Random(seed)

    # Generate embedding with some structure
    embedding = []
    for i in range(dim):
        # Mix of random + text-derived features
        base = rng.gauss(0, 1)
        # Add some text-based signal
        char_sum = sum(ord(c) for c in text[:min(100, len(text))])
        text_bias = math.sin(char_sum * 0.01 + i * 0.1) * 0.3
        embedding.append(base + text_bias)

    # Normalize
    mag = math.sqrt(sum(x*x for x in embedding))
    return [x / (mag + 1e-10) for x in embedding]


# ============================================================================
# DRAGON CURVE FRACTAL LOGIC
# ============================================================================

def get_dragon_turn(step: int) -> int:
    """Dragon Curve turn direction: 1=Right, -1=Left"""
    if step <= 0:
        return 1
    return 1 if (((step & -step) << 1) & step) == 0 else -1

def compute_fractal_depth(step: int) -> int:
    """Folding depth at current step"""
    if step <= 0:
        return 0
    depth = 0
    while step % 2 == 0:
        step //= 2
        depth += 1
    return depth


# ============================================================================
# HUE VECTOR (Chromatic Residue Memory)
# ============================================================================

@dataclass
class HueVector:
    """5-axis hue representation for terrain bruising"""
    boundary_pressure: float = 0.0
    loopiness: float = 0.0
    novelty: float = 0.0
    coherence: float = 0.0
    risk: float = 0.0

    def to_array(self) -> List[float]:
        return [self.boundary_pressure, self.loopiness, self.novelty,
                self.coherence, self.risk]

    def decay(self, delta: float = 0.05) -> 'HueVector':
        return HueVector(
            boundary_pressure=self.boundary_pressure * (1 - delta),
            loopiness=self.loopiness * (1 - delta),
            novelty=self.novelty * (1 - delta),
            coherence=self.coherence * (1 - delta),
            risk=self.risk * (1 - delta)
        )

    def update(self, quality: 'HueVector', lambd: float = 0.3) -> 'HueVector':
        return HueVector(
            boundary_pressure=(1-lambd)*self.boundary_pressure + lambd*quality.boundary_pressure,
            loopiness=(1-lambd)*self.loopiness + lambd*quality.loopiness,
            novelty=(1-lambd)*self.novelty + lambd*quality.novelty,
            coherence=(1-lambd)*self.coherence + lambd*quality.coherence,
            risk=(1-lambd)*self.risk + lambd*quality.risk
        )

    def magnitude(self) -> float:
        arr = self.to_array()
        return math.sqrt(sum(x*x for x in arr))


# ============================================================================
# TERRAIN NODE (Corpus Chunk)
# ============================================================================

@dataclass
class TerrainNode:
    """A chunk of corpus as terrain node"""
    id: str
    content: str
    embedding: List[float]
    hue: HueVector = field(default_factory=HueVector)
    visit_count: int = 0


# ============================================================================
# SNAP EVENT
# ============================================================================

@dataclass
class SnapEvent:
    """Nonlinear entropy reduction event"""
    delta_entropy: float
    stability_steps: int
    eval_gain: float
    structural_delta: str
    timestamp: str
    fractal_iteration: int
    folding_depth: int
    dragon_sequence: List[int]
    eigen_resonance: float = 0.0  # Alignment with terrain eigenvector

    def to_dict(self) -> dict:
        return {
            "delta_entropy": self.delta_entropy,
            "stability_steps": self.stability_steps,
            "eval_gain": self.eval_gain,
            "structural_delta": self.structural_delta,
            "timestamp": self.timestamp,
            "fractal_iteration": self.fractal_iteration,
            "folding_depth": self.folding_depth,
            "dragon_sequence": self.dragon_sequence,
            "eigen_resonance": self.eigen_resonance
        }


# ============================================================================
# STEREOGRAM TOKEN
# ============================================================================

@dataclass
class StereogramToken:
    """ZK co-instantiation proof with dragon attestation"""
    operator_hash: str
    terrain_hash: str
    snap_event: SnapEvent
    dragon_attestation: dict  # commitment_hash, sequence_length, verification_seed
    scores: List[float]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "operator_hash": self.operator_hash,
            "terrain_hash": self.terrain_hash,
            "snap_event": self.snap_event.to_dict(),
            "dragon_attestation": self.dragon_attestation,
            "scores": self.scores,
            "timestamp": self.timestamp
        }


# ============================================================================
# EXPLORATION RESULT
# ============================================================================

@dataclass
class ExplorationResult:
    """Result of running the engine"""
    steps_taken: int
    snaps_detected: int
    tokens_minted: int
    tokens: List[StereogramToken]
    visited_nodes: List[str]
    entropy_history: List[float]
    dragon_sequence: List[int]

    def to_dict(self) -> dict:
        return {
            "steps_taken": self.steps_taken,
            "snaps_detected": self.snaps_detected,
            "tokens_minted": self.tokens_minted,
            "tokens": [t.to_dict() for t in self.tokens],
            "visited_nodes": self.visited_nodes,
            "entropy_history": self.entropy_history,
            "dragon_sequence": self.dragon_sequence
        }


# ============================================================================
# ROSETTA ENGINE
# ============================================================================

class RosettaEngine:
    """
    Phase 2 Rosetta Engine

    Treats corpus as navigable terrain. Uses Dragon Curve fractal
    traversal with hue-as-memory. Detects snaps and mints tokens.
    """

    def __init__(self, dim: int = 768, k_neighbors: int = 10):
        """
        Initialize engine.

        Args:
            dim: Embedding dimension (768 for transformer models)
            k_neighbors: Number of nearest neighbors per node
        """
        self.dim = dim
        self.k_neighbors = k_neighbors

        # Terrain
        self.nodes: Dict[str, TerrainNode] = {}
        self.adjacency: Dict[str, List[str]] = {}

        # Navigation state
        self.current_node: Optional[str] = None
        self.visited_path: List[str] = []
        self.dragon_turns: List[int] = []
        self.entropy_history: List[float] = []

        # Snap detection - tuned for corpus terrain
        self.snap_epsilon = 0.003      # Very low for high-dim embedding space
        self.snap_stability = 2        # Shorter stability window
        self.snap_gain_threshold = 0.01  # Lower gain requirement
        self.baseline_eval = 0.0

        # Results
        self.snaps: List[SnapEvent] = []
        self.tokens: List[StereogramToken] = []

        # Energy weights
        self.alpha = 0.25  # hue bias
        self.beta = 0.25   # similarity
        self.gamma = 0.25  # novelty
        self.zeta = 0.25   # fractal bias

    def ingest(self, chunks: List[str]) -> int:
        """
        Ingest corpus chunks as terrain nodes.

        Returns number of nodes created.
        """
        # Clear existing terrain
        self.nodes.clear()
        self.adjacency.clear()
        self.visited_path.clear()
        self.dragon_turns.clear()
        self.entropy_history.clear()
        self.snaps.clear()
        self.tokens.clear()
        self.baseline_eval = 0.0

        # Create nodes with embeddings
        for i, chunk in enumerate(chunks):
            node_id = f"chunk_{i}"
            embedding = compute_embedding(chunk, self.dim)
            self.nodes[node_id] = TerrainNode(
                id=node_id,
                content=chunk[:200],  # Store truncated for memory
                embedding=embedding
            )
            self.adjacency[node_id] = []

        # Build k-NN adjacency graph
        self._build_adjacency()

        # Set initial position
        if self.nodes:
            self.current_node = list(self.nodes.keys())[0]
            self.visited_path = [self.current_node]

        return len(self.nodes)

    def _build_adjacency(self):
        """Build k-NN adjacency graph from embeddings"""
        node_ids = list(self.nodes.keys())

        for node_id in node_ids:
            node = self.nodes[node_id]

            # Compute distances to all other nodes
            distances = []
            for other_id in node_ids:
                if other_id == node_id:
                    continue
                other = self.nodes[other_id]
                dist = self._cosine_distance(node.embedding, other.embedding)
                distances.append((other_id, dist))

            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[1])
            neighbors = [d[0] for d in distances[:self.k_neighbors]]
            self.adjacency[node_id] = neighbors

    def _cosine_distance(self, a: List[float], b: List[float]) -> float:
        """Cosine distance (1 - cosine similarity)"""
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x*x for x in a))
        mag_b = math.sqrt(sum(x*x for x in b))
        if mag_a < 1e-10 or mag_b < 1e-10:
            return 1.0
        similarity = dot / (mag_a * mag_b)
        return 1.0 - similarity

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity"""
        return 1.0 - self._cosine_distance(a, b)

    def run(self, steps: int = 100, temperature: float = 0.8) -> ExplorationResult:
        """
        Run fractal exploration.

        Returns ExplorationResult with snaps and tokens.
        """
        if not self.nodes:
            return ExplorationResult(
                steps_taken=0, snaps_detected=0, tokens_minted=0,
                tokens=[], visited_nodes=[], entropy_history=[], dragon_sequence=[]
            )

        eval_score = 0.0

        for step in range(steps):
            # Get dragon turn for this step
            turn = get_dragon_turn(len(self.visited_path))
            self.dragon_turns.append(turn)

            # Navigate
            next_node = self._select_next_node(temperature)
            entropy = self._compute_entropy()

            # Update state
            self.visited_path.append(next_node)
            self.current_node = next_node
            self.entropy_history.append(entropy)
            self.nodes[next_node].visit_count += 1

            # Update hue (bruise the terrain)
            self._update_hue(next_node)

            # Compute eval score (based on novelty and coherence)
            eval_score += self._compute_step_value(next_node)

            # Check for snap
            snap = self._check_snap(eval_score, step)
            if snap:
                self.snaps.append(snap)
                token = self._mint_token(snap, eval_score)
                self.tokens.append(token)

            # Global decay
            self._apply_decay()

        return ExplorationResult(
            steps_taken=steps,
            snaps_detected=len(self.snaps),
            tokens_minted=len(self.tokens),
            tokens=self.tokens,
            visited_nodes=self.visited_path,
            entropy_history=self.entropy_history,
            dragon_sequence=self.dragon_turns
        )

    def _select_next_node(self, temperature: float) -> str:
        """Select next node using fractal-biased energy function"""
        neighbors = self.adjacency.get(self.current_node, [])
        if not neighbors:
            return self.current_node

        # Compute fractal target
        current_emb = self.nodes[self.current_node].embedding
        step = len(self.visited_path)
        fractal_target = self._compute_fractal_target(current_emb, step)

        # Compute energies
        energies = {}
        for neighbor in neighbors:
            node = self.nodes[neighbor]

            # Hue energy
            hue_e = node.hue.magnitude()

            # Similarity energy (prefer similar)
            sim = self._cosine_similarity(current_emb, node.embedding)
            sim_e = 1.0 - sim

            # Novelty energy (prefer unvisited)
            novelty_e = -1.0 / (1 + node.visit_count)

            # Fractal energy (prefer close to fractal target)
            fractal_e = self._cosine_distance(node.embedding, fractal_target)

            # Total energy
            E = (self.alpha * hue_e +
                 self.beta * sim_e +
                 self.gamma * novelty_e +
                 self.zeta * fractal_e)
            energies[neighbor] = E

        # Softmax selection
        min_e = min(energies.values())
        exp_vals = {n: math.exp(-(e - min_e) / temperature) for n, e in energies.items()}
        total = sum(exp_vals.values())
        probs = {n: v / total for n, v in exp_vals.items()}

        # Sample
        r = random.random()
        cumulative = 0.0
        for n, p in probs.items():
            cumulative += p
            if r <= cumulative:
                return n
        return neighbors[-1]

    def _compute_fractal_target(self, current_emb: List[float], step: int) -> List[float]:
        """Compute fractal target in embedding space"""
        turn = get_dragon_turn(step)
        depth = compute_fractal_depth(step)

        # Displacement magnitude decreases with depth
        magnitude = 0.3 / (1 + depth * 0.2)

        # Create target by displacing along pseudo-random axes
        target = list(current_emb)
        axis1 = step % self.dim
        axis2 = (step * 7) % self.dim

        target[axis1] += turn * magnitude
        target[axis2] += turn * magnitude * 0.5

        return target

    def _compute_entropy(self) -> float:
        """Compute action entropy based on energy distribution"""
        neighbors = self.adjacency.get(self.current_node, [])
        if len(neighbors) <= 1:
            return 0.0

        # Use actual energy computation for entropy
        current_emb = self.nodes[self.current_node].embedding
        step = len(self.visited_path)
        fractal_target = self._compute_fractal_target(current_emb, step)

        energies = []
        for neighbor in neighbors:
            node = self.nodes[neighbor]
            hue_e = node.hue.magnitude()
            sim = self._cosine_similarity(current_emb, node.embedding)
            sim_e = 1.0 - sim
            novelty_e = -1.0 / (1 + node.visit_count)
            fractal_e = self._cosine_distance(node.embedding, fractal_target)
            E = self.alpha * hue_e + self.beta * sim_e + self.gamma * novelty_e + self.zeta * fractal_e
            energies.append(E)

        # Softmax probabilities
        min_e = min(energies)
        exp_vals = [math.exp(-(e - min_e) / 0.8) for e in energies]
        total = sum(exp_vals)
        probs = [v / total for v in exp_vals]

        # Shannon entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        return entropy

    def _update_hue(self, node_id: str):
        """Update hue (bruise terrain) at visited node"""
        node = self.nodes[node_id]

        quality = HueVector(
            boundary_pressure=0.1,
            loopiness=min(node.visit_count * 0.2, 1.0),
            novelty=1.0 / (1 + node.visit_count),
            coherence=0.5,
            risk=0.0
        )
        node.hue = node.hue.update(quality)

    def _compute_step_value(self, node_id: str) -> float:
        """Compute value of visiting this node"""
        node = self.nodes[node_id]

        # Novelty bonus
        novelty = 0.1 / (1 + node.visit_count * 0.5)

        # Coherence bonus (similarity to recent path)
        if len(self.visited_path) > 2:
            prev = self.nodes[self.visited_path[-2]]
            sim = self._cosine_similarity(node.embedding, prev.embedding)
            coherence = sim * 0.05
        else:
            coherence = 0.05

        return novelty + coherence

    def _check_snap(self, eval_score: float, step: int) -> Optional[SnapEvent]:
        """Check for snap event: nonlinear entropy reduction + stability + eval gain"""
        if len(self.entropy_history) < self.snap_stability + 1:
            return None

        # Entropy delta
        recent = self.entropy_history[-1]
        prior = self.entropy_history[-(self.snap_stability + 1)]
        delta_h = prior - recent

        if delta_h < self.snap_epsilon:
            return None

        # Stability check: variance in recent entropy must be low
        stable_region = self.entropy_history[-self.snap_stability:]
        mean_stable = sum(stable_region) / len(stable_region)
        variance = sum((e - mean_stable)**2 for e in stable_region) / len(stable_region)
        if variance > 0.5:
            return None

        # Eval gain check
        gain = eval_score - self.baseline_eval
        if gain < self.snap_gain_threshold:
            return None

        self.baseline_eval = eval_score

        # Compute eigen resonance (alignment with mean embedding)
        mean_emb = self._compute_mean_embedding()
        current_emb = self.nodes[self.current_node].embedding
        eigen_resonance = self._cosine_similarity(current_emb, mean_emb)

        # Fractal context
        fractal_iter = len(self.visited_path)
        fold_depth = compute_fractal_depth(fractal_iter)
        dragon_seq = self.dragon_turns[-min(10, len(self.dragon_turns)):]

        return SnapEvent(
            delta_entropy=delta_h,
            stability_steps=self.snap_stability,
            eval_gain=gain,
            structural_delta=f"step_{step}_depth_{fold_depth}_nodes_{len(self.nodes)}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            fractal_iteration=fractal_iter,
            folding_depth=fold_depth,
            dragon_sequence=dragon_seq,
            eigen_resonance=eigen_resonance
        )

    def _mint_token(self, snap: SnapEvent, eval_score: float) -> StereogramToken:
        """Mint a stereogram token from a snap event"""
        # Compute operator hash: H(constraints || policy || visibility || randomness || test_suite)
        operator_data = json.dumps({
            "constraints": f"corpus_terrain_{len(self.nodes)}_nodes",
            "policy": "fractal_dragon_curve",
            "visibility": "local_gradient",
            "randomness": f"seed_{snap.fractal_iteration}",
            "test_suite": "phase2_eval_v1"
        }, sort_keys=True)
        operator_hash = hashlib.sha256(operator_data.encode()).hexdigest()

        # Compute terrain hash: H(node_ids + hue_magnitudes + adjacency)
        terrain_data = json.dumps({
            "nodes": sorted(self.nodes.keys()),
            "hue_magnitudes": {nid: round(n.hue.magnitude(), 6)
                              for nid, n in sorted(self.nodes.items())},
            "adjacency_checksum": hashlib.sha256(
                json.dumps(self.adjacency, sort_keys=True).encode()
            ).hexdigest()[:16]
        }, sort_keys=True)
        terrain_hash = hashlib.sha256(terrain_data.encode()).hexdigest()

        # Dragon attestation: commitment to fractal traversal
        dragon_data = json.dumps({
            "sequence": self.dragon_turns,
            "total_steps": len(self.visited_path),
            "folding_depth": snap.folding_depth
        }, sort_keys=True)
        dragon_commitment = hashlib.sha256(dragon_data.encode()).hexdigest()

        dragon_attestation = {
            "commitment_hash": dragon_commitment,
            "sequence_length": len(self.dragon_turns),
            "verification_seed": hashlib.sha256(
                f"{operator_hash}:{terrain_hash}:{snap.timestamp}".encode()
            ).hexdigest()[:32],
            "folding_depth": snap.folding_depth,
            "eigen_resonance": round(snap.eigen_resonance, 6)
        }

        # Scores: evaluation metrics
        unique_visited = len(set(self.visited_path))
        total_visited = len(self.visited_path)
        scores = [
            round(unique_visited / max(1, len(self.nodes)), 4),   # coverage
            round(unique_visited / max(1, total_visited), 4),     # efficiency
            round(snap.delta_entropy, 6),                          # snap strength
            round(snap.eigen_resonance, 6),                        # resonance
            round(eval_score, 4)                                   # cumulative eval
        ]

        return StereogramToken(
            operator_hash=operator_hash,
            terrain_hash=terrain_hash,
            snap_event=snap,
            dragon_attestation=dragon_attestation,
            scores=scores,
            timestamp=snap.timestamp
        )

    def _apply_decay(self):
        """Apply global hue decay to all nodes (organic forgetting)"""
        for node in self.nodes.values():
            node.hue = node.hue.decay(delta=0.05)

    def _compute_mean_embedding(self) -> List[float]:
        """Compute mean embedding across all nodes (terrain eigenvector proxy)"""
        if not self.nodes:
            return [0.0] * self.dim

        mean = [0.0] * self.dim
        for node in self.nodes.values():
            for i in range(self.dim):
                mean[i] += node.embedding[i]

        n = len(self.nodes)
        mean = [x / n for x in mean]

        # Normalize
        mag = math.sqrt(sum(x*x for x in mean))
        if mag > 1e-10:
            mean = [x / mag for x in mean]

        return mean

    def get_terrain_stats(self) -> dict:
        """Get current terrain statistics"""
        hue_mags = [n.hue.magnitude() for n in self.nodes.values()]
        visit_counts = [n.visit_count for n in self.nodes.values()]
        unique_visited = len(set(self.visited_path))

        return {
            "total_nodes": len(self.nodes),
            "total_edges": sum(len(v) for v in self.adjacency.values()),
            "steps_taken": len(self.visited_path),
            "unique_visited": unique_visited,
            "coverage": unique_visited / max(1, len(self.nodes)),
            "mean_hue_magnitude": sum(hue_mags) / max(1, len(hue_mags)),
            "max_hue_magnitude": max(hue_mags) if hue_mags else 0.0,
            "mean_visit_count": sum(visit_counts) / max(1, len(visit_counts)),
            "max_visit_count": max(visit_counts) if visit_counts else 0,
            "snap_count": len(self.snaps),
            "token_count": len(self.tokens),
            "dragon_turns": len(self.dragon_turns),
            "entropy_trend": (
                self.entropy_history[-1] - self.entropy_history[0]
                if len(self.entropy_history) > 1 else 0.0
            )
        }


# ============================================================================
# CONVENIENCE: Run from command line
# ============================================================================

if __name__ == "__main__":
    # Demo corpus
    corpus = [
        "The attention mechanism computes weighted sums over value vectors.",
        "Transformer architectures use multi-head self-attention layers.",
        "Gradient descent optimizes parameters by following the loss surface.",
        "Neural networks learn hierarchical feature representations.",
        "Backpropagation chains partial derivatives through computation graphs.",
        "Embedding spaces map discrete tokens to continuous vectors.",
        "The softmax function normalizes logits into probability distributions.",
        "Layer normalization stabilizes training of deep networks.",
        "Residual connections enable gradient flow in deep architectures.",
        "Positional encodings inject sequence order information.",
        "The cross-entropy loss measures divergence between distributions.",
        "Dropout regularization prevents overfitting by random masking.",
        "Batch normalization reduces internal covariate shift.",
        "The learning rate controls the step size of parameter updates.",
        "Weight decay adds L2 regularization to prevent large weights.",
        "Knowledge distillation transfers learned patterns between models.",
        "Mixture-of-experts routes inputs to specialized sub-networks.",
        "Sparse attention patterns reduce quadratic complexity.",
        "Quantization reduces model precision for efficient inference.",
        "Reinforcement learning from human feedback aligns model behavior.",
    ]

    engine = RosettaEngine(dim=768, k_neighbors=5)
    n_nodes = engine.ingest(corpus)
    print(f"Ingested {n_nodes} corpus chunks as terrain nodes")

    result = engine.run(steps=200, temperature=0.8)
    print(f"\nExploration complete:")
    print(f"  Steps: {result.steps_taken}")
    print(f"  Snaps detected: {result.snaps_detected}")
    print(f"  Tokens minted: {result.tokens_minted}")
    print(f"  Unique nodes visited: {len(set(result.visited_nodes))}/{n_nodes}")
    print(f"  Dragon turns: {len(result.dragon_sequence)}")

    stats = engine.get_terrain_stats()
    print(f"\nTerrain stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if result.tokens:
        print(f"\nFirst token:")
        print(json.dumps(result.tokens[0].to_dict(), indent=2))
