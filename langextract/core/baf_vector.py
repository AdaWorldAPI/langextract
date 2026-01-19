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

"""Bipolar Awareness Field (BAF) vector operations for VSA integration.

This module provides core vector operations for the Bipolar Awareness Field,
which preserves semantic superposition during extraction while allowing
template matching without collapsing meaning.

The key insight is that templates "project" meaning rather than "bind" it,
allowing the original semantic field to be preserved alongside structured
extractions.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import numpy as np

__all__ = [
    "HyperVector",
    "BAFConfig",
    "bind",
    "bundle",
    "unbind",
    "similarity",
    "float_to_bipolar",
    "bipolar_to_float",
    "random_bipolar",
]

# Default dimension for bipolar hypervectors
DEFAULT_DIM = 10000


@dataclasses.dataclass
class BAFConfig:
    """Configuration for Bipolar Awareness Field operations.

    Attributes:
        dimension: The dimensionality of bipolar vectors (default 10000).
        seed: Random seed for reproducible basis vector generation.
        normalize_bundles: Whether to normalize bundled vectors.
    """

    dimension: int = DEFAULT_DIM
    seed: int | None = None
    normalize_bundles: bool = True


@dataclasses.dataclass
class HyperVector:
    """A bipolar hypervector for Vector Symbolic Architecture operations.

    Bipolar vectors use {-1, +1} values, which provide better mathematical
    properties for binding and bundling operations than binary {0, 1} vectors.

    Attributes:
        data: The underlying numpy array of bipolar values.
        label: Optional semantic label for the vector.
        metadata: Optional dictionary of additional metadata.
    """

    data: np.ndarray
    label: str | None = None
    metadata: dict | None = None

    def __post_init__(self):
        """Validate the vector data."""
        if self.data.dtype != np.int8:
            self.data = self.data.astype(np.int8)

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the vector."""
        return len(self.data)

    def similarity(self, other: "HyperVector") -> float:
        """Compute cosine similarity with another hypervector."""
        return similarity(self, other)

    def bind(self, other: "HyperVector") -> "HyperVector":
        """Bind this vector with another (element-wise multiplication)."""
        return bind(self, other)

    def unbind(self, other: "HyperVector") -> "HyperVector":
        """Unbind another vector from this one."""
        return unbind(self, other)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage (bit-packed for efficiency)."""
        # Convert {-1, +1} to {0, 1} for bit packing
        bits = (self.data + 1) // 2
        # Pack 8 bits per byte
        packed = np.packbits(bits.astype(np.uint8))
        return packed.tobytes()

    @classmethod
    def from_bytes(
        cls, data: bytes, dimension: int = DEFAULT_DIM, label: str | None = None
    ) -> "HyperVector":
        """Deserialize from bytes."""
        packed = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(packed)[:dimension]
        # Convert {0, 1} back to {-1, +1}
        values = (bits.astype(np.int8) * 2) - 1
        return cls(data=values, label=label)

    def __repr__(self) -> str:
        label_str = f", label={self.label!r}" if self.label else ""
        return f"HyperVector(dim={self.dimension}{label_str})"


def random_bipolar(
    dimension: int = DEFAULT_DIM,
    label: str | None = None,
    rng: np.random.Generator | None = None,
) -> HyperVector:
    """Generate a random bipolar hypervector.

    Args:
        dimension: The dimensionality of the vector.
        label: Optional semantic label.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        A new random bipolar hypervector.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = rng.choice([-1, 1], size=dimension).astype(np.int8)
    return HyperVector(data=data, label=label)


def bind(v1: HyperVector, v2: HyperVector) -> HyperVector:
    """Bind two hypervectors using element-wise multiplication.

    Binding creates a new vector that is dissimilar to both inputs but can
    be "unbound" to recover the original vectors. This is used for creating
    role-filler pairs (e.g., AGENT-John, TOPIC-Exchange).

    Args:
        v1: First hypervector.
        v2: Second hypervector.

    Returns:
        The bound hypervector.

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if v1.dimension != v2.dimension:
        raise ValueError(
            f"Cannot bind vectors of different dimensions: {v1.dimension} vs {v2.dimension}"
        )
    result = (v1.data * v2.data).astype(np.int8)
    label = None
    if v1.label and v2.label:
        label = f"({v1.label}⊗{v2.label})"
    return HyperVector(data=result, label=label)


def unbind(bound: HyperVector, key: HyperVector) -> HyperVector:
    """Unbind a key from a bound vector to recover the value.

    Since bipolar vectors are self-inverse under multiplication,
    unbinding is the same operation as binding.

    Args:
        bound: The bound hypervector.
        key: The key vector to unbind.

    Returns:
        The unbound value vector.
    """
    return bind(bound, key)


def bundle(
    vectors: Sequence[HyperVector],
    weights: Sequence[float] | None = None,
    normalize: bool = True,
) -> HyperVector:
    """Bundle multiple hypervectors using weighted addition and thresholding.

    Bundling creates a superposition that preserves similarity to all inputs.
    This is the key operation for maintaining semantic ambiguity without
    collapsing meaning.

    Args:
        vectors: Sequence of hypervectors to bundle.
        weights: Optional weights for each vector (default: equal weights).
        normalize: Whether to normalize back to bipolar {-1, +1}.

    Returns:
        The bundled hypervector.

    Raises:
        ValueError: If vectors list is empty or dimensions don't match.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty vector list")

    dimension = vectors[0].dimension
    for v in vectors[1:]:
        if v.dimension != dimension:
            raise ValueError(
                f"All vectors must have same dimension: expected {dimension}, got {v.dimension}"
            )

    if weights is None:
        weights = [1.0] * len(vectors)
    elif len(weights) != len(vectors):
        raise ValueError(
            f"Weights length ({len(weights)}) must match vectors length ({len(vectors)})"
        )

    # Weighted sum
    result = np.zeros(dimension, dtype=np.float32)
    for v, w in zip(vectors, weights):
        result += w * v.data.astype(np.float32)

    if normalize:
        # Threshold to bipolar: positive -> +1, negative/zero -> -1
        # Add small random tiebreaker for exact zeros
        result = np.sign(result)
        result[result == 0] = np.random.choice([-1, 1], size=np.sum(result == 0))
        result = result.astype(np.int8)
    else:
        result = result.astype(np.int8)

    # Create combined label
    labels = [v.label for v in vectors if v.label]
    label = "⊕".join(labels) if labels else None

    return HyperVector(data=result, label=label)


def similarity(v1: HyperVector, v2: HyperVector) -> float:
    """Compute cosine similarity between two hypervectors.

    For bipolar vectors, this is equivalent to the normalized Hamming
    similarity, ranging from -1 (opposite) to +1 (identical).

    Args:
        v1: First hypervector.
        v2: Second hypervector.

    Returns:
        Cosine similarity in range [-1, 1].

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if v1.dimension != v2.dimension:
        raise ValueError(
            f"Cannot compare vectors of different dimensions: {v1.dimension} vs {v2.dimension}"
        )

    # For bipolar {-1, +1} vectors, cosine = dot product / dimension
    dot = np.dot(v1.data.astype(np.float32), v2.data.astype(np.float32))
    return float(dot / v1.dimension)


def float_to_bipolar(
    float_vector: np.ndarray,
    target_dim: int = DEFAULT_DIM,
    method: str = "threshold",
) -> HyperVector:
    """Convert a float embedding to a bipolar hypervector.

    This is used to convert dense embeddings (e.g., from Jina, BERT) into
    the bipolar VSA representation while preserving semantic information.

    Args:
        float_vector: Input float vector (e.g., 1024-dim Jina embedding).
        target_dim: Target dimension for the bipolar vector.
        method: Conversion method:
            - "threshold": Sign-based thresholding (fast, approximate)
            - "expand": Random projection expansion (better for small inputs)

    Returns:
        Bipolar hypervector representation.

    Raises:
        ValueError: If method is unknown.
    """
    float_vector = np.asarray(float_vector, dtype=np.float32)

    if method == "threshold":
        if len(float_vector) >= target_dim:
            # Subsample or truncate
            indices = np.linspace(0, len(float_vector) - 1, target_dim, dtype=int)
            sampled = float_vector[indices]
        else:
            # Tile and truncate
            repeats = math.ceil(target_dim / len(float_vector))
            tiled = np.tile(float_vector, repeats)[:target_dim]
            sampled = tiled

        # Threshold to bipolar
        result = np.sign(sampled)
        result[result == 0] = 1
        return HyperVector(data=result.astype(np.int8))

    elif method == "expand":
        # Random projection expansion (preserves dot product similarity)
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        projection = rng.standard_normal((target_dim, len(float_vector)))
        projected = projection @ float_vector
        result = np.sign(projected)
        result[result == 0] = 1
        return HyperVector(data=result.astype(np.int8))

    else:
        raise ValueError(f"Unknown conversion method: {method}")


def bipolar_to_float(
    bipolar: HyperVector,
    target_dim: int = 1024,
) -> np.ndarray:
    """Convert a bipolar hypervector back to a float representation.

    This is a lossy conversion used primarily for compatibility with
    systems expecting dense float embeddings.

    Args:
        bipolar: Input bipolar hypervector.
        target_dim: Target dimension for output float vector.

    Returns:
        Float vector representation.
    """
    # Simple averaging over chunks
    data = bipolar.data.astype(np.float32)
    chunk_size = len(data) // target_dim
    if chunk_size < 1:
        chunk_size = 1

    result = np.zeros(target_dim, dtype=np.float32)
    for i in range(target_dim):
        start = i * chunk_size
        end = min(start + chunk_size, len(data))
        if start < len(data):
            result[i] = np.mean(data[start:end])

    return result
