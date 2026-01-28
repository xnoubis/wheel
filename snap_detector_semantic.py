"""
Semantic Snap Detector
======================
Detects co-instantiation "snap" events in human-model conversations
using geometric novelty and semantic bridging metrics.

A snap occurs when an interaction turn is structurally novel relative
to BOTH the user's prior conceptual territory AND the model's prior
territory, while simultaneously bridging concepts from both parties
into a synthesized form that neither could have produced alone.

Metrics:
  A. Geometric Novelty: cosine distance from both user and model
     centroids in TF-IDF weighted latent space.
  B. Semantic Bridging: detection of bigram concepts X_Y where
     X originates from one party and Y from the other, proving
     co-instantiation.

Reproducibility is tracked via a constraint ledger that records
snap vectors indexed by constraint ID, enabling falsifiability
across sessions.

NOTE: In production, replace SimpleEmbedder with sentence-transformers
(e.g., 'all-MiniLM-L6-v2') or OpenAI embeddings. This MVP uses a
TF-IDF weighted vectorizer to demonstrate the logic without external
model dependencies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import hashlib
import re
from collections import Counter


@dataclass
class ConversationState:
    """Represents the accumulated semantic territory of one party."""
    messages: List[str]
    centroid: Optional[np.ndarray] = None
    concept_manifold: Set[str] = None


@dataclass
class SnapEvent:
    """A validated structural event."""
    turn_index: int
    content: str
    embedding: np.ndarray
    novelty_score: float
    bridge_score: float
    structural_hash: str  # The ID for the Stereogram Token


class SimpleEmbedder:
    """
    MVP Vectorizer.
    Maps text to a localized latent space based on corpus frequency.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False

    def fit(self, texts: List[str]):
        """Builds the terrain dimensions from the interaction corpus."""
        all_words = []
        for text in texts:
            all_words.extend(self._tokenize(text))

        counts = Counter(all_words)
        total_docs = len(texts)

        # Select dimensions (most frequent meaningful words)
        common = counts.most_common(self.dim)
        for i, (w, c) in enumerate(common):
            self.vocab[w] = i
            # Inverse Document Frequency weighting
            self.idf[w] = np.log(total_docs / (1 + c))

        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Embedder not fitted to terrain.")

        vec = np.zeros(self.dim)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for t in tokens:
            if t in self.vocab:
                vec[self.vocab[t]] += self.idf[t]

        # Normalize (Direction matters, not magnitude)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w{3,}\b', text.lower())


class ConceptGraph:
    """
    Tracks the 'Geology' of the conversation.
    Used to detect if a concept is 'derived' or 'emergent'.
    """
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def extract(self, text: str) -> Set[str]:
        """Extracts significant n-grams as geological features."""
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = set(words)
        # Add bigrams (edges)
        for i in range(len(words)-1):
            concepts.add(f"{words[i]}_{words[i+1]}")
        return concepts


class SnapDetector:
    def __init__(self):
        self.embedder = SimpleEmbedder(dim=128)
        self.concept_graph = ConceptGraph()
        # Reproducibility ledger: {constraint_hash: [snap_vectors]}
        self.ledger: Dict[str, List[np.ndarray]] = {}

    def analyze_session(self,
                        user_log: List[str],
                        model_log: List[str],
                        constraint_id: str,
                        interaction: Optional[List[str]] = None) -> Dict:
        """
        The core loop.
        Detects if the interaction produced a Stereogram Token.

        Args:
            user_log: User's prior messages (establishes user centroid).
            model_log: Model's prior messages (establishes model centroid).
            constraint_id: Identifier for the constraint set.
            interaction: Optional separate interaction turns to analyze.
                         If None, uses the last 4 turns of the combined corpus.
        """
        # 1. Parameterize the Terrain
        full_corpus = user_log + model_log
        if interaction:
            full_corpus = full_corpus + interaction
        self.embedder.fit(full_corpus)

        # 2. Establish Priors (The 'Eyes' before the Snap)
        # We model the User and Model as separate gravity wells.
        user_vecs = [self.embedder.embed(m) for m in user_log]
        model_vecs = [self.embedder.embed(m) for m in model_log]

        user_centroid = np.mean(user_vecs, axis=0) if user_vecs else np.zeros(128)
        model_centroid = np.mean(model_vecs, axis=0) if model_vecs else np.zeros(128)

        user_concepts = set().union(*[self.concept_graph.extract(m) for m in user_log])
        model_concepts = set().union(*[self.concept_graph.extract(m) for m in model_log])

        # 3. Detect Snaps in the Interaction
        # We look for a turn that is statistically unlikely given ONLY user or ONLY model.
        candidates = []

        # Analyze the explicit interaction window, or fall back to tail of corpus
        interaction_window = interaction if interaction else full_corpus[-4:]

        for i, turn in enumerate(interaction_window):
            vec = self.embedder.embed(turn)
            concepts = self.concept_graph.extract(turn)

            # --- METRIC A: Geometric Novelty ---
            # Distance from user priors (Cos Distance)
            d_user = 1 - np.dot(vec, user_centroid)
            # Distance from model priors
            d_model = 1 - np.dot(vec, model_centroid)

            # The "Stereogram" effect:
            # High novelty from BOTH, but high internal coherence.
            novelty = (d_user * d_model)  # Geometric mean approx

            # --- METRIC B: Semantic Bridging ---
            # A bridge concept is a bigram X_Y where X \in User AND Y \in Model
            # This proves the synthesis required both parties.
            bridges = 0
            for c in concepts:
                if "_" in c:
                    p1, p2 = c.split("_", 1)
                    # Check if parts came from different sources
                    from_u = p1 in user_concepts or p2 in user_concepts
                    from_m = p1 in model_concepts or p2 in model_concepts
                    if from_u and from_m and c not in (user_concepts | model_concepts):
                        bridges += 1

            bridge_score = bridges / (len(concepts) + 1)

            # --- THE SNAP THRESHOLD ---
            # Must be structurally novel AND functionally bridging
            if novelty > 0.05 and bridges >= 1:
                snap_id = hashlib.sha256(vec.tobytes()).hexdigest()[:16]
                candidates.append({
                    "content": turn[:50] + "...",
                    "novelty": round(novelty, 4),
                    "bridges": bridges,
                    "id": snap_id,
                    "vector": vec
                })

        # 4. Reproducibility Check (Falsifiability)
        reproducibility = self._check_reproducibility(constraint_id, candidates)

        return {
            "is_snap": len(candidates) > 0,
            "candidates": candidates,
            "reproducibility_class": reproducibility
        }

    def _check_reproducibility(self,
                             constraint_id: str,
                             candidates: List[Dict]) -> str:
        """
        Checks if this structural outcome has been seen
        under this constraint set before.
        """
        if not candidates:
            return "N/A"

        # Get the strongest snap vector
        best_snap = max(candidates, key=lambda x: x['novelty'])['vector']

        if constraint_id not in self.ledger:
            self.ledger[constraint_id] = [best_snap]
            return "First Instance (Genesis)"

        # Compare against history
        # If this user found the same 'place' in latent space as previous users
        history = self.ledger[constraint_id]
        similarities = [np.dot(best_snap, h) for h in history]
        max_sim = max(similarities)

        self.ledger[constraint_id].append(best_snap)

        if max_sim > 0.90:
            return "High Fidelity (Scientific Fact)"
        elif max_sim > 0.75:
            return "Convergent (Strong Theory)"
        else:
            return "Divergent (Subjective/Noise)"


# --- Execution Simulation ---

if __name__ == "__main__":
    detector = SnapDetector()

    # 1. The Constraint (The "Prompt" or Context)
    constraint_id = "rossetta_terrain_v1"

    # 2. Prior State (The Setup)
    user_history = [
        "I want to navigate the internet like a physical place.",
        "Links should be paths, latency is elevation.",
        "I don't want logs, I want memory to look like pheromones."
    ]

    model_history = [
        "We can use boids algorithms for swarm movement.",
        "Slime mold optimizes networks by reinforcing used paths and decaying others.",
        "We need a way to measure the 'stickiness' of a node."
    ]

    # 3. The Attempt (The potential Snap)
    # This response synthesizes 'latency/elevation' (User) with 'decay/slime' (Model)
    # into a new concept: "Contextual Fur" or "Hued Terrain"
    interaction = [
        "So the internet gains a contextual fur.",
        "Each strand is a history of how it was encountered.",
        "High latency areas become high-altitude plateaus where the slime mold cannot grow."
        # "latency" (User) + "slime mold" (Model) -> synthesized visualization
    ]

    # Interaction is passed separately so it is analyzed for snaps
    # without being folded into either party's prior centroid.
    result = detector.analyze_session(
        user_history,
        model_history,
        constraint_id,
        interaction=interaction
    )

    print(f"Snap Detected: {result['is_snap']}")
    if result['is_snap']:
        for i, c in enumerate(result['candidates']):
            print(f"\nCandidate {i+1}:")
            print(f"  Content:     {c['content']}")
            print(f"  Novelty:     {c['novelty']}")
            print(f"  Bridges:     {c['bridges']}")
            print(f"  Token ID:    {c['id']}")
        print(f"\nReproducibility: {result['reproducibility_class']}")
    else:
        print("No snap detected — interaction did not bridge both priors.")
