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

"""Tests for collapse_resistance module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from langextract.core import collapse_resistance as cr
from langextract.core import data


class CollapseResistanceConfigTest(absltest.TestCase):

  def test_default_config(self):
    """Test default configuration."""
    config = cr.CollapseResistanceConfig()
    self.assertAlmostEqual(config.confidence_weight, 0.4)
    self.assertAlmostEqual(config.alignment_weight, 0.3)
    self.assertAlmostEqual(config.context_weight, 0.3)
    self.assertAlmostEqual(config.base_resistance, 0.1)
    self.assertAlmostEqual(config.max_resistance, 0.95)

  def test_weights_must_sum_to_one(self):
    """Test that weights must sum to 1.0."""
    with self.assertRaises(ValueError) as ctx:
      cr.CollapseResistanceConfig(
          confidence_weight=0.5,
          alignment_weight=0.5,
          context_weight=0.5,  # Total = 1.5
      )
    self.assertIn("sum to 1.0", str(ctx.exception))

  def test_resistance_bounds_validation(self):
    """Test that resistance bounds are validated."""
    with self.assertRaises(ValueError) as ctx:
      cr.CollapseResistanceConfig(
          base_resistance=0.9,
          max_resistance=0.5,  # max < base
      )
    self.assertIn("less than max_resistance", str(ctx.exception))


class UncertaintyTensorTest(absltest.TestCase):

  def test_tensor_shape_validation(self):
    """Test that tensor must be 3x3."""
    with self.assertRaises(ValueError) as ctx:
      cr.UncertaintyTensor(tensor=np.zeros((2, 2)))
    self.assertIn("3x3", str(ctx.exception))

  def test_tensor_value_validation(self):
    """Test that tensor values must be in [0, 1]."""
    tensor = np.ones((3, 3)) * 2  # Values > 1
    with self.assertRaises(ValueError) as ctx:
      cr.UncertaintyTensor(tensor=tensor)
    self.assertIn("[0, 1]", str(ctx.exception))

  def test_from_confidence_high(self):
    """Test tensor generation from high confidence."""
    tensor = cr.UncertaintyTensor.from_confidence(0.9)
    # High confidence should concentrate on K_present
    self.assertGreater(tensor.tensor[0, 1], 0.5)  # K_present
    self.assertLess(tensor.total_uncertainty, 0.2)

  def test_from_confidence_low(self):
    """Test tensor generation from low confidence."""
    tensor = cr.UncertaintyTensor.from_confidence(0.2)
    # Low confidence should spread to ?K states
    self.assertGreater(tensor.total_uncertainty, 0.4)

  def test_from_dict_roundtrip(self):
    """Test dict serialization roundtrip."""
    original = cr.UncertaintyTensor.from_confidence(0.7)
    d = original.to_dict()
    restored = cr.UncertaintyTensor.from_dict(d)
    np.testing.assert_array_almost_equal(original.tensor, restored.tensor)

  def test_total_uncertainty(self):
    """Test total uncertainty calculation."""
    tensor = np.zeros((3, 3))
    tensor[2, :] = [0.1, 0.2, 0.3]  # ?K row
    ut = cr.UncertaintyTensor(tensor=tensor)
    self.assertAlmostEqual(ut.total_uncertainty, 0.6)

  def test_temporal_spread_uniform(self):
    """Test temporal spread for uniform distribution."""
    tensor = np.ones((3, 3)) / 3
    ut = cr.UncertaintyTensor(tensor=tensor)
    # Uniform distribution should have maximum spread
    self.assertGreater(ut.temporal_spread, 0.9)


class ComputeCollapseResistanceTest(parameterized.TestCase):

  def test_default_extraction(self):
    """Test collapse resistance for basic extraction."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test entity",
    )
    resistance = cr.compute_collapse_resistance(ext)
    # Should be within bounds
    self.assertGreaterEqual(resistance, 0.1)
    self.assertLessEqual(resistance, 0.95)

  def test_high_confidence_low_resistance(self):
    """Test that high confidence leads to lower resistance."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
        attributes={"confidence": "0.95"},
    )
    resistance = cr.compute_collapse_resistance(ext)
    # High confidence should allow more collapse (lower resistance)
    self.assertLess(resistance, 0.5)

  def test_low_confidence_high_resistance(self):
    """Test that low confidence leads to higher resistance."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
        attributes={"confidence": "0.1"},
    )
    resistance = cr.compute_collapse_resistance(ext)
    # Low confidence should resist collapse (higher resistance)
    self.assertGreater(resistance, 0.5)

  @parameterized.named_parameters(
      dict(
          testcase_name="exact_match",
          alignment_status=data.AlignmentStatus.MATCH_EXACT,
          expected_range=(0.1, 0.5),
      ),
      dict(
          testcase_name="fuzzy_match",
          alignment_status=data.AlignmentStatus.MATCH_FUZZY,
          expected_range=(0.5, 0.95),
      ),
  )
  def test_alignment_status_affects_resistance(self, alignment_status, expected_range):
    """Test that alignment status affects resistance."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
        alignment_status=alignment_status,
    )
    resistance = cr.compute_collapse_resistance(ext)
    self.assertGreaterEqual(resistance, expected_range[0])
    self.assertLessEqual(resistance, expected_range[1])

  def test_context_ambiguity_affects_resistance(self):
    """Test that context ambiguity parameter affects resistance."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    low_ambiguity = cr.compute_collapse_resistance(ext, context_ambiguity=0.1)
    high_ambiguity = cr.compute_collapse_resistance(ext, context_ambiguity=0.9)
    self.assertGreater(high_ambiguity, low_ambiguity)

  def test_custom_config(self):
    """Test with custom configuration."""
    config = cr.CollapseResistanceConfig(
        confidence_weight=0.6,
        alignment_weight=0.2,
        context_weight=0.2,
        base_resistance=0.2,
        max_resistance=0.8,
    )
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    resistance = cr.compute_collapse_resistance(ext, config=config)
    self.assertGreaterEqual(resistance, 0.2)
    self.assertLessEqual(resistance, 0.8)


class ComputeUncertaintyTensorTest(absltest.TestCase):

  def test_basic_extraction(self):
    """Test uncertainty tensor for basic extraction."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    tensor = cr.compute_uncertainty_tensor(ext)
    self.assertIsInstance(tensor, cr.UncertaintyTensor)
    self.assertEqual(tensor.tensor.shape, (3, 3))

  def test_temporal_markers(self):
    """Test uncertainty tensor with temporal markers."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    tensor = cr.compute_uncertainty_tensor(
        ext,
        temporal_markers={"past": 0.8, "present": 0.1, "future": 0.1},
    )
    # Past should dominate
    past_sum = tensor.tensor[:, 0].sum()
    present_sum = tensor.tensor[:, 1].sum()
    self.assertGreater(past_sum, present_sum)

  def test_high_confidence_extraction(self):
    """Test tensor for high confidence extraction."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
        attributes={"confidence": "0.9"},
    )
    tensor = cr.compute_uncertainty_tensor(ext)
    # Should concentrate on Known states
    known_sum = tensor.tensor[0, :].sum()
    uncertain_sum = tensor.tensor[2, :].sum()
    self.assertGreater(known_sum, uncertain_sum)


class BindWithAwarenessTest(absltest.TestCase):

  def test_low_resistance_favors_bound(self):
    """Test that low resistance favors the bound result."""
    from langextract.core import baf_vector

    vector = baf_vector.random_bipolar(dimension=1000)
    role = baf_vector.random_bipolar(dimension=1000)

    result = cr.bind_with_awareness(vector, role, collapse_resistance=0.1)

    # Low resistance should be more similar to pure bind
    pure_bound = baf_vector.bind(vector, role)
    sim_to_bound = baf_vector.similarity(result, pure_bound)
    sim_to_original = baf_vector.similarity(result, vector)

    self.assertGreater(sim_to_bound, sim_to_original)

  def test_high_resistance_preserves_original(self):
    """Test that high resistance preserves original vector."""
    from langextract.core import baf_vector

    vector = baf_vector.random_bipolar(dimension=1000)
    role = baf_vector.random_bipolar(dimension=1000)

    result = cr.bind_with_awareness(vector, role, collapse_resistance=0.9)

    # High resistance should be more similar to original
    pure_bound = baf_vector.bind(vector, role)
    sim_to_bound = baf_vector.similarity(result, pure_bound)
    sim_to_original = baf_vector.similarity(result, vector)

    self.assertGreater(sim_to_original, sim_to_bound)

  def test_medium_resistance_blends(self):
    """Test that medium resistance produces blend."""
    from langextract.core import baf_vector

    vector = baf_vector.random_bipolar(dimension=1000)
    role = baf_vector.random_bipolar(dimension=1000)

    result = cr.bind_with_awareness(vector, role, collapse_resistance=0.5)

    # Should have some similarity to both
    pure_bound = baf_vector.bind(vector, role)
    sim_to_bound = baf_vector.similarity(result, pure_bound)
    sim_to_original = baf_vector.similarity(result, vector)

    self.assertGreater(sim_to_bound, 0.1)
    self.assertGreater(sim_to_original, 0.1)


if __name__ == "__main__":
  absltest.main()
