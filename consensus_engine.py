"""
Consensus Engine: Stereogram Token Asset Classification
=======================================================
Evaluates the reproducibility and asset class of Stereogram Tokens
via multi-user simulation in latent space.

A "snap" is the midpoint-plus-synthesis between a User prior cluster
and a Model prior cluster in normalized embedding space. The coherence
factor controls how deterministic the bridge is:
  - coherence ~1.0: all users converge to the same bridge (Latent Truth)
  - coherence ~0.0: users scatter randomly (Subjective Noise)

The ConsensusEngine classifies tokens by measuring mean cosine similarity
of snap vectors to their centroid across N simulated users:
  - >= 0.90: High Fidelity (Latent Truth)
  - >= 0.75: Convergent (Strong Theory)
  - <  0.75: Divergent (Noise)
"""

import numpy as np
from dataclasses import dataclass
from typing import List
import hashlib


# --- CONFIGURATION ---
THRESHOLD_HIGH_FIDELITY = 0.90  # Scientific Fact / Latent Truth
THRESHOLD_CONVERGENT = 0.75     # Strong Theory
# The "Tension" sweet spot: Normalized distance from the center
# of the line connecting User and Model centroids.
IDEAL_TENSION_RATIO = 0.5


@dataclass
class TokenAsset:
    constraint_id: str
    reproducibility_class: str
    mean_similarity: float
    asset_value_score: float  # 0.0 to 1.0
    snap_vectors: List[np.ndarray]


class LatentSpaceSimulator:
    """
    Simulates the vector space interaction between User priors and Model priors.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def generate_prior_cluster(self, center_seed: int, spread: float, n: int) -> List[np.ndarray]:
        """Generates a cloud of vectors around a conceptual centroid."""
        np.random.seed(center_seed)
        centroid = np.random.normal(0, 1, self.dim)
        centroid = centroid / np.linalg.norm(centroid)

        vectors = []
        # Scale spread by 1/sqrt(dim) so cluster tightness is
        # independent of embedding dimensionality
        scaled_spread = spread / np.sqrt(self.dim)
        for _ in range(n):
            noise = np.random.normal(0, scaled_spread, self.dim)
            vec = centroid + noise
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return vectors

    def synthesize_snaps(self,
                        user_vecs: List[np.ndarray],
                        model_vecs: List[np.ndarray],
                        coherence_factor: float) -> List[np.ndarray]:
        """
        Simulates the 'Snap' event.
        coherence_factor:
            1.0 = All users/models find the exact same bridge (Truth).
            0.0 = Users/models find random bridges (Noise).
        """
        snaps = []
        for u, m in zip(user_vecs, model_vecs):
            # The "Bridge" vector is technically the midpoint + synthesis noise
            midpoint = (u + m) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)

            # Apply coherence factor (or lack thereof)
            # High coherence = small random deviation from the "True" bridge
            # Scale by 1/sqrt(dim) so total deviation magnitude is
            # independent of embedding dimensionality
            deviation = np.random.normal(
                0, (1.0 - coherence_factor) / np.sqrt(self.dim), self.dim
            )

            snap = midpoint + deviation
            snap = snap / np.linalg.norm(snap)
            snaps.append(snap)
        return snaps


class ConsensusEngine:
    """
    Evaluates the 'Asset Class' of a Stereogram Token.
    """
    @staticmethod
    def evaluate_asset(constraint_id: str, snap_vectors: List[np.ndarray]) -> TokenAsset:
        # 1. Calculate Mean Cosine Similarity
        # We compute the centroid of all snaps, then avg distance to it
        centroid = np.mean(snap_vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        similarities = []
        for vec in snap_vectors:
            sim = np.dot(vec, centroid)
            similarities.append(sim)

        mean_sim = float(np.mean(similarities))

        # 2. Determine Class
        if mean_sim >= THRESHOLD_HIGH_FIDELITY:
            r_class = "High Fidelity (Latent Truth)"
            value = 1.0
        elif mean_sim >= THRESHOLD_CONVERGENT:
            r_class = "Convergent (Theory)"
            value = 0.6
        else:
            r_class = "Divergent (Noise)"
            value = 0.1

        return TokenAsset(
            constraint_id=constraint_id,
            reproducibility_class=r_class,
            mean_similarity=mean_sim,
            asset_value_score=value,
            snap_vectors=snap_vectors
        )


# --- EXECUTION ---
if __name__ == "__main__":
    sim = LatentSpaceSimulator()
    engine = ConsensusEngine()

    print(">>> ROSETTA CONSENSUS PROTOCOL: 100-USER TEST\n")

    # Scenario A: The "Rosetta" Constraint (High Truth)
    # The constraint is strict, leading to high coherence.
    print("--- Simulation A: Constraint 'Contextual Fur' ---")
    users_a = sim.generate_prior_cluster(center_seed=42, spread=0.2, n=100)
    model_a = sim.generate_prior_cluster(center_seed=99, spread=0.1, n=100)
    snaps_a = sim.synthesize_snaps(users_a, model_a, coherence_factor=0.95)

    asset_a = engine.evaluate_asset("rossetta_v1", snaps_a)
    print(f"Class: {asset_a.reproducibility_class}")
    print(f"Coherence: {asset_a.mean_similarity:.4f}")
    print(f"Asset Value: {asset_a.asset_value_score}")
    print("")

    # Scenario B: A Vague Constraint (Subjective Noise)
    # The constraint is loose, users project their own bias.
    print("--- Simulation B: Constraint 'Make it cool' ---")
    users_b = sim.generate_prior_cluster(center_seed=12, spread=0.8, n=100)
    model_b = sim.generate_prior_cluster(center_seed=55, spread=0.5, n=100)
    snaps_b = sim.synthesize_snaps(users_b, model_b, coherence_factor=0.3)

    asset_b = engine.evaluate_asset("vague_v1", snaps_b)
    print(f"Class: {asset_b.reproducibility_class}")
    print(f"Coherence: {asset_b.mean_similarity:.4f}")
    print(f"Asset Value: {asset_b.asset_value_score}")
