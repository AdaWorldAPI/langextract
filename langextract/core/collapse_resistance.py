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

"""Collapse resistance computation for preserving semantic superposition.

This module provides utilities for computing "collapse resistance" - a measure
of how much an extraction should resist being bound to a fixed semantic role.

The key insight is that:
- High confidence, unambiguous extractions can safely collapse (bind)
- Low confidence or ambiguous extractions should preserve superposition

Collapse resistance ranges from 0.0 (fully bound/collapsed) to 1.0 (fully
superposed/uncollapsed).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from langextract.core import data

__all__ = [
    "CollapseResistanceConfig",
    "UncertaintyTensor",
    "compute_collapse_resistance",
    "compute_uncertainty_tensor",
    "bind_with_awareness",
]


@dataclasses.dataclass
class CollapseResistanceConfig:
    """Configuration for collapse resistance computation.

    Attributes:
        confidence_weight: Weight for extraction confidence (0-1).
        alignment_weight: Weight for alignment quality (0-1).
        context_weight: Weight for contextual ambiguity (0-1).
        base_resistance: Minimum collapse resistance (floor).
        max_resistance: Maximum collapse resistance (ceiling).
    """

    confidence_weight: float = 0.4
    alignment_weight: float = 0.3
    context_weight: float = 0.3
    base_resistance: float = 0.1
    max_resistance: float = 0.95

    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.confidence_weight + self.alignment_weight + self.context_weight
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-3):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.3f}"
            )
        if not 0 <= self.base_resistance < self.max_resistance <= 1:
            raise ValueError(
                "base_resistance must be less than max_resistance, both in [0,1]"
            )


@dataclasses.dataclass
class UncertaintyTensor:
    """3x3 epistemic uncertainty tensor.

    Represents uncertainty across two dimensions:
    - Epistemic state: Known (K), Not-Known (¬K), Unknown-if-Known (?K)
    - Temporal: Past, Present, Future

    Each cell contains a probability [0, 1] representing the confidence
    that the extraction holds the given epistemic-temporal state.

    Layout:
        [[K_past,  K_present,  K_future ],
         [¬K_past, ¬K_present, ¬K_future],
         [?K_past, ?K_present, ?K_future]]
    """

    tensor: np.ndarray  # Shape (3, 3)

    def __post_init__(self):
        """Validate tensor shape and values."""
        if self.tensor.shape != (3, 3):
            raise ValueError(f"Tensor must be 3x3, got {self.tensor.shape}")
        if not np.all((self.tensor >= 0) & (self.tensor <= 1)):
            raise ValueError("All tensor values must be in [0, 1]")

    @classmethod
    def from_confidence(cls, confidence: float) -> "UncertaintyTensor":
        """Create uncertainty tensor from a simple confidence score.

        High confidence -> concentrated on K_present
        Low confidence -> spread across ?K states

        Args:
            confidence: Extraction confidence in [0, 1].

        Returns:
            An UncertaintyTensor reflecting the confidence.
        """
        tensor = np.zeros((3, 3), dtype=np.float32)

        # Known state concentrates on present
        tensor[0, 1] = confidence  # K_present

        # Unknown state spreads uncertainty
        uncertainty = 1.0 - confidence
        tensor[2, 0] = uncertainty * 0.2  # ?K_past
        tensor[2, 1] = uncertainty * 0.6  # ?K_present
        tensor[2, 2] = uncertainty * 0.2  # ?K_future

        return cls(tensor=tensor)

    @classmethod
    def from_dict(cls, d: dict) -> "UncertaintyTensor":
        """Create from a dictionary with named keys.

        Args:
            d: Dictionary with keys like "K_past", "¬K_present", etc.

        Returns:
            An UncertaintyTensor.
        """
        keys = [
            ["K_past", "K_present", "K_future"],
            ["¬K_past", "¬K_present", "¬K_future"],
            ["?K_past", "?K_present", "?K_future"],
        ]
        tensor = np.zeros((3, 3), dtype=np.float32)
        for i, row in enumerate(keys):
            for j, key in enumerate(row):
                tensor[i, j] = d.get(key, 0.0)
        return cls(tensor=tensor)

    def to_dict(self) -> dict:
        """Convert to dictionary with named keys."""
        keys = [
            ["K_past", "K_present", "K_future"],
            ["¬K_past", "¬K_present", "¬K_future"],
            ["?K_past", "?K_present", "?K_future"],
        ]
        result = {}
        for i, row in enumerate(keys):
            for j, key in enumerate(row):
                result[key] = float(self.tensor[i, j])
        return result

    @property
    def total_uncertainty(self) -> float:
        """Compute total uncertainty (sum of ?K row)."""
        return float(np.sum(self.tensor[2, :]))

    @property
    def temporal_spread(self) -> float:
        """Compute how spread the uncertainty is across time."""
        col_sums = np.sum(self.tensor, axis=0)
        if col_sums.sum() == 0:
            return 0.0
        normalized = col_sums / col_sums.sum()
        # Entropy-based spread measure
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        max_entropy = np.log2(3)  # Maximum for 3 states
        return float(entropy / max_entropy)


def compute_collapse_resistance(
    extraction: "data.Extraction",
    config: CollapseResistanceConfig | None = None,
    context_ambiguity: float = 0.5,
) -> float:
    """Compute collapse resistance for an extraction.

    Collapse resistance determines how much the extraction should resist
    being bound to a fixed semantic role. High resistance preserves
    superposition; low resistance allows binding.

    Args:
        extraction: The extraction to evaluate.
        config: Configuration for resistance computation.
        context_ambiguity: External measure of contextual ambiguity [0, 1].

    Returns:
        Collapse resistance in [0, 1].
    """
    if config is None:
        config = CollapseResistanceConfig()

    # Factor 1: Inverse of extraction confidence (if available)
    # Lower confidence -> higher resistance
    confidence_factor = 0.5  # Default neutral
    if hasattr(extraction, "attributes") and extraction.attributes:
        if "confidence" in extraction.attributes:
            try:
                conf = float(extraction.attributes["confidence"])
                confidence_factor = 1.0 - conf
            except (ValueError, TypeError):
                pass

    # Factor 2: Alignment quality
    # Poor alignment -> higher resistance
    alignment_factor = 0.5  # Default neutral
    if extraction.alignment_status is not None:
        from langextract.core.data import AlignmentStatus

        alignment_scores = {
            AlignmentStatus.MATCH_EXACT: 0.1,   # Low resistance for exact
            AlignmentStatus.MATCH_LESSER: 0.4,  # Medium-low
            AlignmentStatus.MATCH_GREATER: 0.5, # Medium
            AlignmentStatus.MATCH_FUZZY: 0.7,   # Higher resistance for fuzzy
        }
        alignment_factor = alignment_scores.get(extraction.alignment_status, 0.5)

    # Factor 3: Context ambiguity (passed in)
    context_factor = context_ambiguity

    # Weighted combination
    raw_resistance = (
        config.confidence_weight * confidence_factor
        + config.alignment_weight * alignment_factor
        + config.context_weight * context_factor
    )

    # Apply floor and ceiling
    resistance = max(config.base_resistance, min(config.max_resistance, raw_resistance))

    return resistance


def compute_uncertainty_tensor(
    extraction: "data.Extraction",
    temporal_markers: dict[str, float] | None = None,
) -> UncertaintyTensor:
    """Compute the uncertainty tensor for an extraction.

    The tensor captures epistemic uncertainty (known/unknown/uncertain)
    across temporal dimensions (past/present/future).

    Args:
        extraction: The extraction to evaluate.
        temporal_markers: Optional dict of temporal signals extracted from
            context, e.g., {"past": 0.3, "present": 0.5, "future": 0.2}.

    Returns:
        A 3x3 UncertaintyTensor.
    """
    # Start with confidence-based default
    confidence = 0.5
    if hasattr(extraction, "attributes") and extraction.attributes:
        if "confidence" in extraction.attributes:
            try:
                confidence = float(extraction.attributes["confidence"])
            except (ValueError, TypeError):
                pass

    # Adjust based on alignment
    if extraction.alignment_status is not None:
        from langextract.core.data import AlignmentStatus

        if extraction.alignment_status == AlignmentStatus.MATCH_EXACT:
            confidence = min(1.0, confidence * 1.2)
        elif extraction.alignment_status == AlignmentStatus.MATCH_FUZZY:
            confidence = confidence * 0.8

    tensor = np.zeros((3, 3), dtype=np.float32)

    # Distribute across epistemic states
    # Cap known_prob to leave room for unknown and uncertain
    known_prob = min(0.85, confidence)
    unknown_prob = 0.1
    uncertain_prob = max(0.0, 1.0 - known_prob - unknown_prob)

    # Normalize to ensure sum equals 1.0
    total = known_prob + unknown_prob + uncertain_prob
    if total > 0:
        known_prob /= total
        unknown_prob /= total
        uncertain_prob /= total

    # Default temporal distribution (concentrated on present)
    if temporal_markers is None:
        temporal_markers = {"past": 0.2, "present": 0.6, "future": 0.2}

    # Normalize temporal markers
    total = sum(temporal_markers.values())
    if total > 0:
        temporal_markers = {k: v / total for k, v in temporal_markers.items()}

    # Fill tensor
    # Row 0: Known (K)
    tensor[0, 0] = known_prob * temporal_markers.get("past", 0.2)
    tensor[0, 1] = known_prob * temporal_markers.get("present", 0.6)
    tensor[0, 2] = known_prob * temporal_markers.get("future", 0.2)

    # Row 1: Not-Known (¬K)
    tensor[1, 0] = unknown_prob * temporal_markers.get("past", 0.2)
    tensor[1, 1] = unknown_prob * temporal_markers.get("present", 0.6)
    tensor[1, 2] = unknown_prob * temporal_markers.get("future", 0.2)

    # Row 2: Uncertain (?K)
    tensor[2, 0] = uncertain_prob * temporal_markers.get("past", 0.2)
    tensor[2, 1] = uncertain_prob * temporal_markers.get("present", 0.6)
    tensor[2, 2] = uncertain_prob * temporal_markers.get("future", 0.2)

    return UncertaintyTensor(tensor=tensor)


def bind_with_awareness(
    vector: "baf_vector.HyperVector",
    role_vector: "baf_vector.HyperVector",
    collapse_resistance: float,
) -> "baf_vector.HyperVector":
    """Bind a vector to a role while preserving awareness based on resistance.

    When collapse_resistance is high, the result preserves more of the original
    semantic superposition. When low, it allows full binding/collapse.

    Args:
        vector: The original semantic vector.
        role_vector: The role to bind to (e.g., AGENT, TOPIC).
        collapse_resistance: How much to resist collapse [0, 1].

    Returns:
        A hypervector blending original and bound representations.
    """
    from langextract.core import baf_vector

    bound = baf_vector.bind(vector, role_vector)

    if collapse_resistance > 0.5:
        # High resistance: bundle original with bound
        # Weight original more heavily
        return baf_vector.bundle(
            [vector, bound],
            weights=[collapse_resistance, 1.0 - collapse_resistance],
        )
    else:
        # Low resistance: mostly collapsed
        return baf_vector.bundle(
            [bound, vector],
            weights=[1.0 - collapse_resistance, collapse_resistance],
        )
