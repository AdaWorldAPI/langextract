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

"""Tests for baf_vector module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from langextract.core import baf_vector


class HyperVectorTest(absltest.TestCase):

  def test_random_bipolar_dimension(self):
    """Test that random_bipolar creates vectors with correct dimension."""
    dim = 1000
    vec = baf_vector.random_bipolar(dimension=dim)
    self.assertEqual(vec.dimension, dim)

  def test_random_bipolar_values(self):
    """Test that random_bipolar creates bipolar {-1, +1} values."""
    vec = baf_vector.random_bipolar(dimension=100)
    unique_values = set(vec.data)
    self.assertEqual(unique_values, {-1, 1})

  def test_random_bipolar_label(self):
    """Test that label is preserved."""
    vec = baf_vector.random_bipolar(dimension=100, label="test_label")
    self.assertEqual(vec.label, "test_label")

  def test_random_bipolar_reproducibility(self):
    """Test that same RNG produces same vector."""
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    vec1 = baf_vector.random_bipolar(dimension=100, rng=rng1)
    vec2 = baf_vector.random_bipolar(dimension=100, rng=rng2)
    np.testing.assert_array_equal(vec1.data, vec2.data)

  def test_to_bytes_roundtrip(self):
    """Test serialization/deserialization roundtrip."""
    original = baf_vector.random_bipolar(dimension=1000, label="test")
    data = original.to_bytes()
    restored = baf_vector.HyperVector.from_bytes(data, dimension=1000)
    np.testing.assert_array_equal(original.data, restored.data)

  def test_to_bytes_size(self):
    """Test that bit-packing produces expected size."""
    vec = baf_vector.random_bipolar(dimension=10000)
    data = vec.to_bytes()
    expected_size = 10000 // 8
    self.assertEqual(len(data), expected_size)


class BindTest(absltest.TestCase):

  def test_bind_dimension_match(self):
    """Test that bind requires matching dimensions."""
    v1 = baf_vector.random_bipolar(dimension=100)
    v2 = baf_vector.random_bipolar(dimension=200)
    with self.assertRaises(ValueError) as ctx:
      baf_vector.bind(v1, v2)
    self.assertIn("different dimensions", str(ctx.exception))

  def test_bind_result_is_bipolar(self):
    """Test that binding preserves bipolar values."""
    v1 = baf_vector.random_bipolar(dimension=100)
    v2 = baf_vector.random_bipolar(dimension=100)
    bound = baf_vector.bind(v1, v2)
    unique_values = set(bound.data)
    self.assertEqual(unique_values, {-1, 1})

  def test_bind_is_dissimilar(self):
    """Test that binding produces dissimilar result."""
    v1 = baf_vector.random_bipolar(dimension=1000)
    v2 = baf_vector.random_bipolar(dimension=1000)
    bound = baf_vector.bind(v1, v2)
    # Bound vector should be nearly orthogonal to both inputs
    sim1 = baf_vector.similarity(bound, v1)
    sim2 = baf_vector.similarity(bound, v2)
    self.assertLess(abs(sim1), 0.2)  # Near zero
    self.assertLess(abs(sim2), 0.2)

  def test_bind_unbind_recovery(self):
    """Test that unbinding recovers original vector."""
    v1 = baf_vector.random_bipolar(dimension=1000, label="value")
    v2 = baf_vector.random_bipolar(dimension=1000, label="key")
    bound = baf_vector.bind(v1, v2)
    recovered = baf_vector.unbind(bound, v2)
    # Should be identical since bind is XOR-like for bipolar
    np.testing.assert_array_equal(v1.data, recovered.data)


class BundleTest(absltest.TestCase):

  def test_bundle_empty_list(self):
    """Test that bundling empty list raises error."""
    with self.assertRaises(ValueError) as ctx:
      baf_vector.bundle([])
    self.assertIn("empty", str(ctx.exception).lower())

  def test_bundle_dimension_mismatch(self):
    """Test that bundling mismatched dimensions raises error."""
    v1 = baf_vector.random_bipolar(dimension=100)
    v2 = baf_vector.random_bipolar(dimension=200)
    with self.assertRaises(ValueError) as ctx:
      baf_vector.bundle([v1, v2])
    self.assertIn("same dimension", str(ctx.exception))

  def test_bundle_preserves_similarity(self):
    """Test that bundling preserves similarity to inputs."""
    v1 = baf_vector.random_bipolar(dimension=1000, label="a")
    v2 = baf_vector.random_bipolar(dimension=1000, label="b")
    bundled = baf_vector.bundle([v1, v2])
    # Bundle should be similar to both inputs
    sim1 = baf_vector.similarity(bundled, v1)
    sim2 = baf_vector.similarity(bundled, v2)
    self.assertGreater(sim1, 0.3)  # Positive similarity
    self.assertGreater(sim2, 0.3)

  def test_bundle_weighted(self):
    """Test that weighted bundling respects weights."""
    v1 = baf_vector.random_bipolar(dimension=1000)
    v2 = baf_vector.random_bipolar(dimension=1000)
    # Heavy weight on v1
    bundled = baf_vector.bundle([v1, v2], weights=[0.9, 0.1])
    sim1 = baf_vector.similarity(bundled, v1)
    sim2 = baf_vector.similarity(bundled, v2)
    # Should be more similar to v1
    self.assertGreater(sim1, sim2)

  def test_bundle_label_combination(self):
    """Test that bundling combines labels."""
    v1 = baf_vector.random_bipolar(dimension=100, label="a")
    v2 = baf_vector.random_bipolar(dimension=100, label="b")
    bundled = baf_vector.bundle([v1, v2])
    self.assertEqual(bundled.label, "a⊕b")


class SimilarityTest(absltest.TestCase):

  def test_similarity_identical(self):
    """Test that identical vectors have similarity 1.0."""
    v = baf_vector.random_bipolar(dimension=100)
    sim = baf_vector.similarity(v, v)
    self.assertAlmostEqual(sim, 1.0)

  def test_similarity_opposite(self):
    """Test that opposite vectors have similarity -1.0."""
    v1 = baf_vector.random_bipolar(dimension=100)
    v2 = baf_vector.HyperVector(data=-v1.data)
    sim = baf_vector.similarity(v1, v2)
    self.assertAlmostEqual(sim, -1.0)

  def test_similarity_range(self):
    """Test that similarity is in [-1, 1]."""
    v1 = baf_vector.random_bipolar(dimension=100)
    v2 = baf_vector.random_bipolar(dimension=100)
    sim = baf_vector.similarity(v1, v2)
    self.assertGreaterEqual(sim, -1.0)
    self.assertLessEqual(sim, 1.0)

  def test_similarity_dimension_mismatch(self):
    """Test that similarity requires matching dimensions."""
    v1 = baf_vector.random_bipolar(dimension=100)
    v2 = baf_vector.random_bipolar(dimension=200)
    with self.assertRaises(ValueError):
      baf_vector.similarity(v1, v2)


class FloatConversionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="threshold", method="threshold"),
      dict(testcase_name="expand", method="expand"),
  )
  def test_float_to_bipolar_dimension(self, method):
    """Test that conversion produces correct dimension."""
    float_vec = np.random.randn(1024).astype(np.float32)
    bipolar = baf_vector.float_to_bipolar(float_vec, target_dim=10000, method=method)
    self.assertEqual(bipolar.dimension, 10000)

  @parameterized.named_parameters(
      dict(testcase_name="threshold", method="threshold"),
      dict(testcase_name="expand", method="expand"),
  )
  def test_float_to_bipolar_values(self, method):
    """Test that conversion produces bipolar values."""
    float_vec = np.random.randn(1024).astype(np.float32)
    bipolar = baf_vector.float_to_bipolar(float_vec, target_dim=10000, method=method)
    unique_values = set(bipolar.data)
    self.assertEqual(unique_values, {-1, 1})

  def test_float_to_bipolar_similarity_preservation(self):
    """Test that similar float vectors produce similar bipolar vectors."""
    v1 = np.random.randn(1024).astype(np.float32)
    v2 = v1 + 0.1 * np.random.randn(1024).astype(np.float32)  # Small perturbation

    b1 = baf_vector.float_to_bipolar(v1, target_dim=10000, method="expand")
    b2 = baf_vector.float_to_bipolar(v2, target_dim=10000, method="expand")

    sim = baf_vector.similarity(b1, b2)
    self.assertGreater(sim, 0.5)  # Should preserve similarity

  def test_bipolar_to_float_dimension(self):
    """Test that bipolar_to_float produces correct dimension."""
    bipolar = baf_vector.random_bipolar(dimension=10000)
    float_vec = baf_vector.bipolar_to_float(bipolar, target_dim=1024)
    self.assertEqual(len(float_vec), 1024)

  def test_unknown_method_raises(self):
    """Test that unknown conversion method raises error."""
    float_vec = np.random.randn(1024).astype(np.float32)
    with self.assertRaises(ValueError) as ctx:
      baf_vector.float_to_bipolar(float_vec, method="unknown")
    self.assertIn("unknown", str(ctx.exception).lower())


class BAFConfigTest(absltest.TestCase):

  def test_default_config(self):
    """Test default configuration values."""
    config = baf_vector.BAFConfig()
    self.assertEqual(config.dimension, 10000)
    self.assertIsNone(config.seed)
    self.assertTrue(config.normalize_bundles)

  def test_custom_config(self):
    """Test custom configuration."""
    config = baf_vector.BAFConfig(dimension=5000, seed=42, normalize_bundles=False)
    self.assertEqual(config.dimension, 5000)
    self.assertEqual(config.seed, 42)
    self.assertFalse(config.normalize_bundles)


if __name__ == "__main__":
  absltest.main()
