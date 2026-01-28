"""
verify_phase2.py
================
Verification suite for Rosetta Phase 2 Engine.

Tests:
  1. Structural: classes, methods, fields exist with correct signatures
  2. Embedding: 768-dim, normalized, deterministic
  3. Dragon Curve: correct turn sequence and fractal depth
  4. Hue Vector: decay, update, magnitude
  5. Ingestion: corpus -> terrain graph with k-NN adjacency
  6. Navigation: energy-based selection, entropy computation
  7. Snap Detection: entropy drop + stability + eval gain
  8. Token Minting: SHA-256 hashes, dragon attestation, scores
  9. Full Run: end-to-end exploration produces valid results
"""

import hashlib
import json
import math
import sys
import traceback
from typing import List

# ── Import the engine ──────────────────────────────────────────────────────
try:
    from rosetta_phase2_engine import (
        RosettaEngine,
        HueVector,
        TerrainNode,
        SnapEvent,
        StereogramToken,
        ExplorationResult,
        compute_embedding,
        get_dragon_turn,
        compute_fractal_depth,
    )
except ImportError as e:
    print(f"FAIL: Cannot import rosetta_phase2_engine: {e}")
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Test Corpus ────────────────────────────────────────────────────────────

CORPUS = [
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


# ============================================================================
# 1. STRUCTURAL CHECKS
# ============================================================================

def test_structural():
    section("1. Structural Checks")

    # RosettaEngine exists and has required methods
    engine = RosettaEngine()
    check("RosettaEngine class exists", True)
    check("RosettaEngine.ingest method", hasattr(engine, 'ingest') and callable(engine.ingest))
    check("RosettaEngine.run method", hasattr(engine, 'run') and callable(engine.run))
    check("RosettaEngine.get_terrain_stats method", hasattr(engine, 'get_terrain_stats') and callable(engine.get_terrain_stats))
    check("RosettaEngine.dim == 768", engine.dim == 768)
    check("RosettaEngine.k_neighbors == 10", engine.k_neighbors == 10)

    # HueVector
    hv = HueVector()
    check("HueVector class exists", True)
    check("HueVector has 5 axes",
          all(hasattr(hv, a) for a in ['boundary_pressure', 'loopiness', 'novelty', 'coherence', 'risk']))
    check("HueVector.to_array returns list of 5", len(hv.to_array()) == 5)
    check("HueVector.decay returns HueVector", isinstance(hv.decay(), HueVector))
    check("HueVector.update returns HueVector", isinstance(hv.update(HueVector()), HueVector))
    check("HueVector.magnitude returns float", isinstance(hv.magnitude(), float))

    # SnapEvent
    snap = SnapEvent(
        delta_entropy=0.1, stability_steps=3, eval_gain=0.05,
        structural_delta="test", timestamp="2025-01-01T00:00:00Z",
        fractal_iteration=10, folding_depth=2, dragon_sequence=[1, -1, 1]
    )
    check("SnapEvent class exists", True)
    check("SnapEvent.to_dict returns dict", isinstance(snap.to_dict(), dict))
    d = snap.to_dict()
    required_snap_fields = ['delta_entropy', 'stability_steps', 'eval_gain',
                            'structural_delta', 'timestamp', 'fractal_iteration',
                            'folding_depth', 'dragon_sequence']
    check("SnapEvent.to_dict has all required fields",
          all(f in d for f in required_snap_fields),
          f"Missing: {[f for f in required_snap_fields if f not in d]}")

    # StereogramToken
    token = StereogramToken(
        operator_hash="a" * 64, terrain_hash="b" * 64,
        snap_event=snap, dragon_attestation={"commitment_hash": "c" * 64},
        scores=[0.5, 0.6], timestamp="2025-01-01T00:00:00Z"
    )
    check("StereogramToken class exists", True)
    check("StereogramToken.to_dict returns dict", isinstance(token.to_dict(), dict))
    td = token.to_dict()
    required_token_fields = ['operator_hash', 'terrain_hash', 'snap_event',
                             'dragon_attestation', 'scores', 'timestamp']
    check("StereogramToken.to_dict has all required fields",
          all(f in td for f in required_token_fields),
          f"Missing: {[f for f in required_token_fields if f not in td]}")

    # ExplorationResult
    er = ExplorationResult(
        steps_taken=0, snaps_detected=0, tokens_minted=0,
        tokens=[], visited_nodes=[], entropy_history=[], dragon_sequence=[]
    )
    check("ExplorationResult class exists", True)
    check("ExplorationResult.to_dict returns dict", isinstance(er.to_dict(), dict))


# ============================================================================
# 2. EMBEDDING CHECKS
# ============================================================================

def test_embedding():
    section("2. Embedding Checks")

    emb = compute_embedding("test input", dim=768)
    check("Embedding returns list", isinstance(emb, list))
    check("Embedding dimension is 768", len(emb) == 768)

    # Check normalization
    mag = math.sqrt(sum(x*x for x in emb))
    check("Embedding is normalized (magnitude ~1.0)", abs(mag - 1.0) < 0.01,
          f"magnitude={mag:.6f}")

    # Check determinism
    emb2 = compute_embedding("test input", dim=768)
    check("Embedding is deterministic",
          all(abs(a - b) < 1e-10 for a, b in zip(emb, emb2)))

    # Different inputs produce different embeddings
    emb3 = compute_embedding("different input", dim=768)
    diff = sum(abs(a - b) for a, b in zip(emb, emb3))
    check("Different inputs produce different embeddings", diff > 0.1,
          f"diff={diff:.6f}")

    # Custom dimension
    emb_custom = compute_embedding("test", dim=128)
    check("Custom dimension works (128)", len(emb_custom) == 128)


# ============================================================================
# 3. DRAGON CURVE CHECKS
# ============================================================================

def test_dragon_curve():
    section("3. Dragon Curve Checks")

    # Known dragon curve sequence for first 8 turns: R R L R R L L
    # In our encoding: 1=R, -1=L
    # The sequence for steps 1..7 should be: 1, 1, -1, 1, 1, -1, -1
    expected_first_7 = [1, 1, -1, 1, 1, -1, -1]
    actual = [get_dragon_turn(i) for i in range(1, 8)]
    check("Dragon turn sequence (steps 1-7)", actual == expected_first_7,
          f"expected={expected_first_7}, got={actual}")

    # Step 0 returns 1
    check("get_dragon_turn(0) == 1", get_dragon_turn(0) == 1)

    # All turns are either 1 or -1
    turns_100 = [get_dragon_turn(i) for i in range(100)]
    check("All turns are 1 or -1", all(t in (1, -1) for t in turns_100))

    # Fractal depth
    check("compute_fractal_depth(0) == 0", compute_fractal_depth(0) == 0)
    check("compute_fractal_depth(1) == 0", compute_fractal_depth(1) == 0)
    check("compute_fractal_depth(2) >= 1", compute_fractal_depth(2) >= 1)
    check("compute_fractal_depth(4) >= 2", compute_fractal_depth(4) >= 2)
    check("compute_fractal_depth(8) >= 3", compute_fractal_depth(8) >= 3)

    # Depth increases with powers of 2
    depths = [compute_fractal_depth(2**i) for i in range(6)]
    check("Fractal depth monotonically increases for powers of 2",
          all(depths[i] <= depths[i+1] for i in range(len(depths)-1)),
          f"depths={depths}")


# ============================================================================
# 4. HUE VECTOR CHECKS
# ============================================================================

def test_hue_vector():
    section("4. Hue Vector Checks")

    hv = HueVector(boundary_pressure=1.0, loopiness=0.5, novelty=0.3,
                    coherence=0.8, risk=0.2)

    # Magnitude
    expected_mag = math.sqrt(1.0**2 + 0.5**2 + 0.3**2 + 0.8**2 + 0.2**2)
    check("HueVector magnitude correct", abs(hv.magnitude() - expected_mag) < 1e-6,
          f"expected={expected_mag:.6f}, got={hv.magnitude():.6f}")

    # Decay reduces magnitude
    decayed = hv.decay(delta=0.1)
    check("Decay reduces magnitude", decayed.magnitude() < hv.magnitude(),
          f"original={hv.magnitude():.4f}, decayed={decayed.magnitude():.4f}")

    # Decay with delta=0 is identity
    no_decay = hv.decay(delta=0.0)
    check("Decay delta=0 preserves values",
          abs(no_decay.magnitude() - hv.magnitude()) < 1e-10)

    # Update blends toward quality
    quality = HueVector(boundary_pressure=0.0, loopiness=0.0, novelty=1.0,
                        coherence=1.0, risk=0.0)
    updated = hv.update(quality, lambd=1.0)
    check("Update lambda=1.0 replaces with quality",
          abs(updated.novelty - 1.0) < 1e-10 and abs(updated.boundary_pressure - 0.0) < 1e-10)

    updated_half = hv.update(quality, lambd=0.5)
    check("Update lambda=0.5 blends correctly",
          abs(updated_half.novelty - 0.65) < 1e-6,
          f"expected 0.65, got {updated_half.novelty:.6f}")

    # Zero vector
    zero = HueVector()
    check("Zero HueVector magnitude is 0", zero.magnitude() == 0.0)


# ============================================================================
# 5. INGESTION CHECKS
# ============================================================================

def test_ingestion():
    section("5. Ingestion Checks")

    engine = RosettaEngine(dim=768, k_neighbors=5)
    n = engine.ingest(CORPUS)

    check("Ingestion returns correct count", n == len(CORPUS),
          f"expected={len(CORPUS)}, got={n}")
    check("Engine has correct number of nodes", len(engine.nodes) == len(CORPUS))

    # All nodes have embeddings of correct dimension
    for nid, node in engine.nodes.items():
        if len(node.embedding) != 768:
            check(f"Node {nid} embedding dim", False,
                  f"dim={len(node.embedding)}")
            break
    else:
        check("All nodes have 768-dim embeddings", True)

    # Adjacency graph exists
    check("Adjacency graph populated", len(engine.adjacency) == len(CORPUS))

    # Each node has <= k_neighbors
    max_neighbors = max(len(v) for v in engine.adjacency.values())
    check("Adjacency respects k_neighbors limit", max_neighbors <= 5,
          f"max_neighbors={max_neighbors}")

    # Current node is set
    check("Current node initialized", engine.current_node is not None)
    check("Current node is first chunk", engine.current_node == "chunk_0")

    # Visited path initialized
    check("Visited path initialized", len(engine.visited_path) == 1)

    # Node content stored
    first_node = engine.nodes["chunk_0"]
    check("Node content stored (truncated)",
          len(first_node.content) > 0 and len(first_node.content) <= 200)

    # Re-ingestion clears state
    engine.snaps.append(SnapEvent(
        delta_entropy=0.1, stability_steps=2, eval_gain=0.05,
        structural_delta="test", timestamp="t", fractal_iteration=1,
        folding_depth=0, dragon_sequence=[1]
    ))
    engine.ingest(CORPUS[:5])
    check("Re-ingestion clears snaps", len(engine.snaps) == 0)
    check("Re-ingestion updates node count", len(engine.nodes) == 5)


# ============================================================================
# 6. NAVIGATION CHECKS
# ============================================================================

def test_navigation():
    section("6. Navigation Checks")

    engine = RosettaEngine(dim=768, k_neighbors=5)
    engine.ingest(CORPUS)

    # Single step navigation
    start = engine.current_node
    next_node = engine._select_next_node(temperature=0.8)
    check("Navigation returns a node ID", next_node in engine.nodes)
    check("Navigation returns a neighbor",
          next_node in engine.adjacency.get(start, []) or next_node == start)

    # Entropy computation
    entropy = engine._compute_entropy()
    check("Entropy is non-negative", entropy >= 0.0)
    check("Entropy is finite", math.isfinite(entropy))

    # Step value computation
    value = engine._compute_step_value(next_node)
    check("Step value is positive", value > 0)
    check("Step value is finite", math.isfinite(value))

    # Hue update modifies node
    node = engine.nodes[next_node]
    old_mag = node.hue.magnitude()
    engine._update_hue(next_node)
    new_mag = node.hue.magnitude()
    check("Hue update changes magnitude", new_mag != old_mag or old_mag == 0.0,
          f"old={old_mag:.6f}, new={new_mag:.6f}")

    # Decay reduces all hue magnitudes
    mags_before = {nid: n.hue.magnitude() for nid, n in engine.nodes.items()}
    engine._apply_decay()
    mags_after = {nid: n.hue.magnitude() for nid, n in engine.nodes.items()}
    decayed = all(mags_after[nid] <= mags_before[nid] + 1e-10 for nid in engine.nodes)
    check("Global decay reduces or preserves all hue magnitudes", decayed)


# ============================================================================
# 7. SNAP DETECTION CHECKS
# ============================================================================

def test_snap_detection():
    section("7. Snap Detection Checks")

    engine = RosettaEngine(dim=768, k_neighbors=5)
    engine.ingest(CORPUS)

    # Run enough steps to potentially trigger snaps
    result = engine.run(steps=200, temperature=0.8)

    check("Entropy history populated", len(engine.entropy_history) > 0)
    check("Entropy values are non-negative",
          all(e >= 0 for e in engine.entropy_history))
    check("Entropy values are finite",
          all(math.isfinite(e) for e in engine.entropy_history))

    # Check if snaps were detected (likely with 200 steps on 20 nodes)
    check("At least one snap detected after 200 steps",
          len(engine.snaps) > 0,
          f"snaps={len(engine.snaps)}")

    if engine.snaps:
        snap = engine.snaps[0]
        check("Snap has positive delta_entropy", snap.delta_entropy > 0)
        check("Snap has stability_steps >= 1", snap.stability_steps >= 1)
        check("Snap has positive eval_gain", snap.eval_gain > 0)
        check("Snap has structural_delta string", isinstance(snap.structural_delta, str) and len(snap.structural_delta) > 0)
        check("Snap has ISO timestamp", isinstance(snap.timestamp, str) and 'T' in snap.timestamp)
        check("Snap has fractal_iteration", snap.fractal_iteration > 0)
        check("Snap has folding_depth >= 0", snap.folding_depth >= 0)
        check("Snap has dragon_sequence", isinstance(snap.dragon_sequence, list) and len(snap.dragon_sequence) > 0)
        check("Snap dragon_sequence values are 1/-1",
              all(t in (1, -1) for t in snap.dragon_sequence))

        # Snap to_dict roundtrip
        d = snap.to_dict()
        check("Snap to_dict preserves delta_entropy",
              abs(d['delta_entropy'] - snap.delta_entropy) < 1e-10)


# ============================================================================
# 8. TOKEN MINTING CHECKS
# ============================================================================

def test_token_minting():
    section("8. Token Minting Checks")

    engine = RosettaEngine(dim=768, k_neighbors=5)
    engine.ingest(CORPUS)
    result = engine.run(steps=200, temperature=0.8)

    check("Tokens minted count matches snaps",
          len(engine.tokens) == len(engine.snaps),
          f"tokens={len(engine.tokens)}, snaps={len(engine.snaps)}")

    if engine.tokens:
        token = engine.tokens[0]

        # Hash format checks
        check("operator_hash is 64-char hex",
              len(token.operator_hash) == 64 and all(c in '0123456789abcdef' for c in token.operator_hash))
        check("terrain_hash is 64-char hex",
              len(token.terrain_hash) == 64 and all(c in '0123456789abcdef' for c in token.terrain_hash))

        # Dragon attestation
        da = token.dragon_attestation
        check("dragon_attestation has commitment_hash",
              'commitment_hash' in da and len(da['commitment_hash']) == 64)
        check("dragon_attestation has sequence_length",
              'sequence_length' in da and isinstance(da['sequence_length'], int))
        check("dragon_attestation has verification_seed",
              'verification_seed' in da and len(da['verification_seed']) == 32)

        # Scores
        check("Token has scores list", isinstance(token.scores, list))
        check("Token has >= 3 scores", len(token.scores) >= 3)
        check("All scores are numeric",
              all(isinstance(s, (int, float)) for s in token.scores))

        # Timestamp
        check("Token has ISO timestamp", 'T' in token.timestamp)

        # Snap event is embedded
        check("Token contains snap_event", token.snap_event is not None)
        check("Token snap_event is SnapEvent", isinstance(token.snap_event, SnapEvent))

        # to_dict roundtrip
        td = token.to_dict()
        check("Token to_dict has operator_hash", td['operator_hash'] == token.operator_hash)
        check("Token to_dict has terrain_hash", td['terrain_hash'] == token.terrain_hash)
        check("Token to_dict snap_event is dict", isinstance(td['snap_event'], dict))
        check("Token to_dict dragon_attestation is dict", isinstance(td['dragon_attestation'], dict))

        # JSON serializable
        try:
            json_str = json.dumps(td)
            check("Token to_dict is JSON serializable", True)
            reparsed = json.loads(json_str)
            check("Token JSON roundtrip preserves operator_hash",
                  reparsed['operator_hash'] == token.operator_hash)
        except (TypeError, ValueError) as e:
            check("Token to_dict is JSON serializable", False, str(e))


# ============================================================================
# 9. FULL RUN CHECKS
# ============================================================================

def test_full_run():
    section("9. Full Run (End-to-End)")

    engine = RosettaEngine(dim=768, k_neighbors=5)
    n = engine.ingest(CORPUS)

    result = engine.run(steps=300, temperature=0.8)

    check("ExplorationResult steps_taken == 300", result.steps_taken == 300)
    check("ExplorationResult snaps_detected matches engine",
          result.snaps_detected == len(engine.snaps))
    check("ExplorationResult tokens_minted matches engine",
          result.tokens_minted == len(engine.tokens))
    check("ExplorationResult tokens list matches",
          len(result.tokens) == len(engine.tokens))

    # Visited nodes
    check("Visited nodes > steps (includes initial)",
          len(result.visited_nodes) == 300 + 1,  # initial + 300 steps
          f"visited={len(result.visited_nodes)}")
    check("All visited nodes are valid",
          all(nid in engine.nodes for nid in result.visited_nodes))

    # Entropy history
    check("Entropy history length == steps",
          len(result.entropy_history) == 300,
          f"len={len(result.entropy_history)}")

    # Dragon sequence
    check("Dragon sequence length == steps",
          len(result.dragon_sequence) == 300,
          f"len={len(result.dragon_sequence)}")
    check("Dragon sequence values are 1/-1",
          all(t in (1, -1) for t in result.dragon_sequence))

    # Coverage
    unique = len(set(result.visited_nodes))
    coverage = unique / n
    check(f"Terrain coverage > 50% ({coverage:.0%})", coverage > 0.5,
          f"unique={unique}/{n}")

    # to_dict roundtrip
    rd = result.to_dict()
    check("ExplorationResult to_dict has all fields",
          all(k in rd for k in ['steps_taken', 'snaps_detected', 'tokens_minted',
                                 'tokens', 'visited_nodes', 'entropy_history',
                                 'dragon_sequence']))

    # JSON serializable
    try:
        json_str = json.dumps(rd)
        check("ExplorationResult is JSON serializable", True)
    except (TypeError, ValueError) as e:
        check("ExplorationResult is JSON serializable", False, str(e))

    # Terrain stats
    stats = engine.get_terrain_stats()
    check("get_terrain_stats returns dict", isinstance(stats, dict))
    required_stat_keys = ['total_nodes', 'total_edges', 'steps_taken',
                          'unique_visited', 'coverage', 'snap_count', 'token_count']
    check("Terrain stats has required keys",
          all(k in stats for k in required_stat_keys),
          f"Missing: {[k for k in required_stat_keys if k not in stats]}")

    # Empty engine run
    empty_engine = RosettaEngine()
    empty_result = empty_engine.run(steps=10)
    check("Empty engine run returns 0 steps", empty_result.steps_taken == 0)
    check("Empty engine run returns no snaps", empty_result.snaps_detected == 0)


# ============================================================================
# RUNNER
# ============================================================================

def main():
    print("=" * 60)
    print("  Rosetta Phase 2 Engine — Verification Suite")
    print("=" * 60)

    try:
        test_structural()
        test_embedding()
        test_dragon_curve()
        test_hue_vector()
        test_ingestion()
        test_navigation()
        test_snap_detection()
        test_token_minting()
        test_full_run()
    except Exception as e:
        print(f"\n  [FATAL] Unexpected exception: {e}")
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print(f"{'='*60}")

    if FAIL > 0:
        print(f"\n  STATUS: FAIL ({FAIL} failures)")
        sys.exit(1)
    else:
        print(f"\n  STATUS: ALL PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
