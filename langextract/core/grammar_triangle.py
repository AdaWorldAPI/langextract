# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grammar Triangle Field for semantic modulation without collapse.

The Grammar Triangle embeds meaning in a three-way field:
1. NSM (Natural Semantic Metalanguage) - Primitives like FEEL, WANT, KNOW
2. Causality - Flow of agency, temporality, and dependency
3. ICC (Qualia) - Continuous felt-sense dimensions

Instead of classifying meaning into discrete buckets, the triangle creates
a continuous field where meaning flows without collapsing. Templates become
projections, not classifications.

The triangle:
        CAUSALITY (flows, not causes)
              /\\
             /  \\
            /    \\
    NSM <--⊕--> ICC (Qualia field)
  (primitives)    (continuous)
        |
        ↓
   BIPOLAR VSA FIELD
   (holds superposition)
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

from langextract.core import baf_vector

if TYPE_CHECKING:
    from langextract.core import data

__all__ = [
    "GrammarTriangleField",
    "NSMField",
    "CausalityFlow",
    "QualiaField",
    "NSM_PRIMITIVES",
    "QUALIA_DIMENSIONS",
]

# NSM (Natural Semantic Metalanguage) primitives
# Based on Wierzbicka's semantic primes
NSM_PRIMITIVES = [
    # Substantives
    "I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "BODY",
    # Determiners
    "THIS", "THE_SAME", "OTHER",
    # Quantifiers
    "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY",
    # Evaluators
    "GOOD", "BAD",
    # Descriptors
    "BIG", "SMALL",
    # Mental predicates
    "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
    # Speech
    "SAY", "WORDS", "TRUE",
    # Actions/Events
    "DO", "HAPPEN", "MOVE", "TOUCH",
    # Existence/Possession
    "THERE_IS", "HAVE",
    # Life/Death
    "LIVE", "DIE",
    # Time
    "WHEN", "NOW", "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT",
    # Space
    "WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "SIDE", "INSIDE",
    # Logical
    "NOT", "MAYBE", "CAN", "BECAUSE", "IF",
    # Intensifier
    "VERY",
    # Similarity
    "LIKE",
]

# Qualia dimensions (18D phenomenal field)
# Based on experiential qualities relevant to meaning
QUALIA_DIMENSIONS = [
    "valence",           # Positive/negative feeling
    "arousal",           # Activation level
    "dominance",         # Control/power
    "certainty",         # Epistemic confidence
    "agency",            # Self as cause
    "urgency",           # Time pressure
    "intimacy",          # Personal closeness
    "novelty",           # Familiarity/newness
    "complexity",        # Cognitive load
    "concreteness",      # Abstract/concrete
    "temporality",       # Past/present/future orientation
    "sociality",         # Individual/collective
    "formality",         # Casual/formal register
    "intensity",         # Strength of experience
    "scope",             # Local/global relevance
    "stability",         # Transient/enduring
    "salience",          # Attention-grabbing
    "coherence",         # Integration with context
]


@dataclasses.dataclass
class NSMField:
    """NSM primitives as continuous weights, not discrete labels.

    Instead of: ["FEEL", "WANT"]
    We store:   {"FEEL": 0.8, "WANT": 0.6, "KNOW": 0.3, ...}

    Attributes:
        weights: Dictionary mapping NSM primitives to weights [0, 1].
    """

    weights: dict[str, float]

    def __post_init__(self):
        """Validate weights."""
        for prim, weight in self.weights.items():
            if prim not in NSM_PRIMITIVES:
                raise ValueError(f"Unknown NSM primitive: {prim}")
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight must be in [0, 1], got {weight} for {prim}")

    @classmethod
    def from_text(cls, text: str) -> "NSMField":
        """Compute NSM field from text using keyword matching.

        This is a simple heuristic implementation. For production use,
        this should be replaced with a learned model.

        Args:
            text: Input text to analyze.

        Returns:
            NSMField with computed weights.
        """
        text_lower = text.lower()
        weights = {}

        # Simple keyword-based heuristics
        keyword_map = {
            "FEEL": ["feel", "emotion", "sense", "experience"],
            "WANT": ["want", "desire", "wish", "need", "prefer"],
            "KNOW": ["know", "understand", "aware", "realize", "believe"],
            "THINK": ["think", "consider", "suppose", "believe", "opinion"],
            "DO": ["do", "make", "create", "perform", "execute"],
            "SAY": ["say", "tell", "speak", "mention", "discuss"],
            "SEE": ["see", "look", "watch", "observe", "notice"],
            "HEAR": ["hear", "listen", "sound"],
            "GOOD": ["good", "great", "excellent", "positive", "beneficial"],
            "BAD": ["bad", "wrong", "negative", "harmful", "terrible"],
            "NOW": ["now", "current", "present", "today"],
            "BEFORE": ["before", "previous", "earlier", "past", "ago"],
            "AFTER": ["after", "later", "future", "next", "tomorrow"],
            "BECAUSE": ["because", "since", "due", "reason", "cause"],
            "IF": ["if", "when", "unless", "whether", "condition"],
            "CAN": ["can", "could", "able", "possible", "capability"],
            "SOMEONE": ["someone", "person", "people", "anyone", "everybody"],
            "SOMETHING": ["something", "thing", "object", "item"],
        }

        for primitive in NSM_PRIMITIVES:
            keywords = keyword_map.get(primitive, [primitive.lower()])
            count = sum(1 for kw in keywords if kw in text_lower)
            # Normalize to [0, 1] with saturation
            weights[primitive] = min(1.0, count * 0.3)

        return cls(weights=weights)

    def to_vector(self, dimension: int = 1000) -> np.ndarray:
        """Convert NSM field to a float vector.

        Args:
            dimension: Target vector dimension.

        Returns:
            Float vector representation.
        """
        # Create basis vectors for each primitive (deterministic)
        rng = np.random.default_rng(seed=42)
        result = np.zeros(dimension, dtype=np.float32)

        for i, prim in enumerate(NSM_PRIMITIVES):
            weight = self.weights.get(prim, 0.0)
            if weight > 0:
                # Generate deterministic basis vector
                basis = rng.standard_normal(dimension).astype(np.float32)
                basis /= np.linalg.norm(basis)
                result += weight * basis

        return result

    @property
    def dominant_primitives(self) -> list[str]:
        """Return primitives with weight > 0.5."""
        return [p for p, w in self.weights.items() if w > 0.5]


@dataclasses.dataclass
class CausalityFlow:
    """Causality as a 3D flow vector, not discrete edges.

    Dimensions:
    - agency: causing (-1) ← → caused (+1)
    - temporality: before (-1) ← → after (+1)
    - dependency: grounding (-1) ← → dependent (+1)

    Attributes:
        flow: 3D numpy array representing the causal flow.
    """

    flow: np.ndarray  # Shape (3,)

    def __post_init__(self):
        """Validate flow shape."""
        self.flow = np.asarray(self.flow, dtype=np.float32)
        if self.flow.shape != (3,):
            raise ValueError(f"Flow must be 3D, got shape {self.flow.shape}")

    @classmethod
    def from_markers(
        cls,
        caused_by: bool = False,
        causes: bool = False,
        temporal_marker: str | None = None,
        depends_on: bool = False,
        grounds: bool = False,
    ) -> "CausalityFlow":
        """Create causality flow from discrete markers.

        Args:
            caused_by: Whether this is caused by something else.
            causes: Whether this causes something else.
            temporal_marker: "before", "after", or None.
            depends_on: Whether this depends on something else.
            grounds: Whether this grounds something else.

        Returns:
            CausalityFlow representing the causal relationship.
        """
        flow = np.zeros(3, dtype=np.float32)

        # Agency dimension
        if caused_by:
            flow[0] = 0.8
        elif causes:
            flow[0] = -0.8

        # Temporality dimension
        if temporal_marker == "before":
            flow[1] = -0.5
        elif temporal_marker == "after":
            flow[1] = 0.5

        # Dependency dimension
        if depends_on:
            flow[2] = 0.7
        elif grounds:
            flow[2] = -0.7

        return cls(flow=flow)

    @property
    def agency(self) -> float:
        """Return agency dimension value."""
        return float(self.flow[0])

    @property
    def temporality(self) -> float:
        """Return temporality dimension value."""
        return float(self.flow[1])

    @property
    def dependency(self) -> float:
        """Return dependency dimension value."""
        return float(self.flow[2])

    def to_vector(self, dimension: int = 1000) -> np.ndarray:
        """Expand 3D flow to higher-dimensional vector.

        Args:
            dimension: Target vector dimension.

        Returns:
            Float vector representation.
        """
        # Use deterministic projection
        rng = np.random.default_rng(seed=43)
        projection = rng.standard_normal((dimension, 3)).astype(np.float32)
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)
        return projection @ self.flow


@dataclasses.dataclass
class QualiaField:
    """18D phenomenal qualia field.

    Represents continuous felt-sense dimensions relevant to meaning.

    Attributes:
        values: Dictionary mapping qualia dimensions to values [-1, 1].
    """

    values: dict[str, float]

    def __post_init__(self):
        """Validate values."""
        for dim, val in self.values.items():
            if dim not in QUALIA_DIMENSIONS:
                raise ValueError(f"Unknown qualia dimension: {dim}")
            if not -1 <= val <= 1:
                raise ValueError(f"Value must be in [-1, 1], got {val} for {dim}")

    @classmethod
    def from_text(cls, text: str) -> "QualiaField":
        """Compute qualia field from text using heuristics.

        Args:
            text: Input text to analyze.

        Returns:
            QualiaField with computed values.
        """
        text_lower = text.lower()
        values = {}

        # Heuristic mappings
        positive_words = {"good", "great", "excellent", "happy", "love", "wonderful"}
        negative_words = {"bad", "terrible", "awful", "hate", "sad", "horrible"}
        urgent_words = {"urgent", "immediately", "asap", "now", "critical", "emergency"}
        formal_words = {"hereby", "pursuant", "whereas", "therefore", "accordingly"}
        casual_words = {"hey", "yeah", "cool", "awesome", "gonna", "wanna"}

        # Valence
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        values["valence"] = np.clip((pos_count - neg_count) * 0.3, -1, 1)

        # Arousal (based on punctuation and caps)
        exclaim_count = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        values["arousal"] = np.clip(exclaim_count * 0.2 + caps_ratio * 2, -1, 1)

        # Urgency
        urgent_count = sum(1 for w in urgent_words if w in text_lower)
        values["urgency"] = np.clip(urgent_count * 0.4, -1, 1)

        # Formality
        formal_count = sum(1 for w in formal_words if w in text_lower)
        casual_count = sum(1 for w in casual_words if w in text_lower)
        values["formality"] = np.clip((formal_count - casual_count) * 0.3, -1, 1)

        # Default values for other dimensions
        for dim in QUALIA_DIMENSIONS:
            if dim not in values:
                values[dim] = 0.0

        return cls(values=values)

    def to_vector(self) -> np.ndarray:
        """Convert to 18D vector.

        Returns:
            18D float vector.
        """
        result = np.zeros(len(QUALIA_DIMENSIONS), dtype=np.float32)
        for i, dim in enumerate(QUALIA_DIMENSIONS):
            result[i] = self.values.get(dim, 0.0)
        return result

    def to_expanded_vector(self, dimension: int = 1000) -> np.ndarray:
        """Expand 18D qualia to higher dimension.

        Args:
            dimension: Target vector dimension.

        Returns:
            Float vector representation.
        """
        base = self.to_vector()
        rng = np.random.default_rng(seed=44)
        projection = rng.standard_normal((dimension, len(base))).astype(np.float32)
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)
        return projection @ base


@dataclasses.dataclass
class GrammarTriangleField:
    """Grammar that doesn't collapse meaning.

    Instead of: extract → classify → store
    We do:      extract → embed in field → store field

    The field combines NSM primitives, causality flow, and qualia into
    a unified representation that preserves semantic superposition.

    Attributes:
        nsm_field: NSM primitive weights.
        causality_flow: 3D causal flow vector.
        qualia_field: 18D qualia values.
        awareness_vector: Fused bipolar hypervector.
    """

    nsm_field: NSMField
    causality_flow: CausalityFlow
    qualia_field: QualiaField
    awareness_vector: baf_vector.HyperVector | None = None

    @classmethod
    def from_extraction(
        cls,
        extraction: "data.Extraction",
        base_embedding: np.ndarray | None = None,
        dimension: int = baf_vector.DEFAULT_DIM,
    ) -> "GrammarTriangleField":
        """Create a GrammarTriangleField from an extraction.

        Args:
            extraction: The extraction to analyze.
            base_embedding: Optional dense embedding (e.g., from Jina).
            dimension: Target dimension for the awareness vector.

        Returns:
            GrammarTriangleField with all components computed.
        """
        text = extraction.extraction_text

        # Compute triangle components
        nsm = NSMField.from_text(text)
        causality = CausalityFlow.from_markers()  # Default flow
        qualia = QualiaField.from_text(text)

        # Parse causal markers from attributes if available
        if extraction.attributes:
            causality = CausalityFlow.from_markers(
                caused_by="caused_by" in extraction.attributes,
                causes="causes" in extraction.attributes,
                temporal_marker=extraction.attributes.get("temporal_marker"),
                depends_on="depends_on" in extraction.attributes,
                grounds="grounds" in extraction.attributes,
            )

        # Create the field
        field = cls(
            nsm_field=nsm,
            causality_flow=causality,
            qualia_field=qualia,
        )

        # Fuse to bipolar awareness vector
        field.awareness_vector = field._fuse_to_bipolar(
            base_embedding=base_embedding,
            dimension=dimension,
        )

        return field

    def _fuse_to_bipolar(
        self,
        base_embedding: np.ndarray | None = None,
        dimension: int = baf_vector.DEFAULT_DIM,
    ) -> baf_vector.HyperVector:
        """Fuse all triangle components into single bipolar hypervector.

        This is where the magic happens:
        - Base embedding provides semantic foundation
        - NSM/causal/qualia MODULATE but don't COLLAPSE

        Args:
            base_embedding: Optional dense embedding to use as base.
            dimension: Target dimension for the hypervector.

        Returns:
            Fused bipolar hypervector.
        """
        vectors = []
        weights = []

        # Base embedding (highest weight if provided)
        if base_embedding is not None:
            base = baf_vector.float_to_bipolar(base_embedding, dimension)
            base.label = "semantic_base"
            vectors.append(base)
            weights.append(0.5)
        else:
            # Use random base if no embedding provided
            base = baf_vector.random_bipolar(dimension, label="random_base")
            vectors.append(base)
            weights.append(0.3)

        # NSM component
        nsm_vec = self.nsm_field.to_vector(dimension)
        nsm_bipolar = baf_vector.float_to_bipolar(nsm_vec, dimension)
        nsm_bipolar.label = "nsm"
        vectors.append(nsm_bipolar)
        weights.append(0.2)

        # Causality component
        causal_vec = self.causality_flow.to_vector(dimension)
        causal_bipolar = baf_vector.float_to_bipolar(causal_vec, dimension)
        causal_bipolar.label = "causality"
        vectors.append(causal_bipolar)
        weights.append(0.15)

        # Qualia component
        qualia_vec = self.qualia_field.to_expanded_vector(dimension)
        qualia_bipolar = baf_vector.float_to_bipolar(qualia_vec, dimension)
        qualia_bipolar.label = "qualia"
        vectors.append(qualia_bipolar)
        weights.append(0.15)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Bundle (not bind) - preserves superposition
        return baf_vector.bundle(vectors, weights)

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage.

        Returns:
            Dictionary representation.
        """
        return {
            "nsm_field": self.nsm_field.weights,
            "causality_flow": self.causality_flow.flow.tolist(),
            "qualia_field": self.qualia_field.values,
            "awareness_vector_bytes": (
                self.awareness_vector.to_bytes() if self.awareness_vector else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict, dimension: int = baf_vector.DEFAULT_DIM) -> "GrammarTriangleField":
        """Deserialize from dictionary.

        Args:
            d: Dictionary representation.
            dimension: Dimension for the awareness vector.

        Returns:
            GrammarTriangleField instance.
        """
        nsm = NSMField(weights=d["nsm_field"])
        causality = CausalityFlow(flow=np.array(d["causality_flow"]))
        qualia = QualiaField(values=d["qualia_field"])

        awareness_vector = None
        if d.get("awareness_vector_bytes"):
            awareness_vector = baf_vector.HyperVector.from_bytes(
                d["awareness_vector_bytes"], dimension=dimension
            )

        return cls(
            nsm_field=nsm,
            causality_flow=causality,
            qualia_field=qualia,
            awareness_vector=awareness_vector,
        )
