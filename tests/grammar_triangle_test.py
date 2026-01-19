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

"""Tests for grammar_triangle module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from langextract.core import baf_vector
from langextract.core import data
from langextract.core import grammar_triangle as gt


class NSMFieldTest(absltest.TestCase):

  def test_from_text_basic(self):
    """Test NSM field extraction from basic text."""
    nsm = gt.NSMField.from_text("I feel happy and want to know more")
    # Should detect mental predicates
    self.assertGreater(nsm.weights.get("FEEL", 0), 0)
    self.assertGreater(nsm.weights.get("WANT", 0), 0)
    self.assertGreater(nsm.weights.get("KNOW", 0), 0)

  def test_from_text_action(self):
    """Test NSM field for action-oriented text."""
    nsm = gt.NSMField.from_text("I will do this and make it happen")
    self.assertGreater(nsm.weights.get("DO", 0), 0)

  def test_from_text_temporal(self):
    """Test NSM field for temporal text."""
    # Note: Using base form "say" since simple keyword matching doesn't handle verb tenses
    nsm = gt.NSMField.from_text("Before that, I say something now")
    self.assertGreater(nsm.weights.get("BEFORE", 0), 0)
    self.assertGreater(nsm.weights.get("SAY", 0), 0)

  def test_invalid_primitive(self):
    """Test that invalid primitive raises error."""
    with self.assertRaises(ValueError) as ctx:
      gt.NSMField(weights={"INVALID_PRIMITIVE": 0.5})
    self.assertIn("Unknown NSM primitive", str(ctx.exception))

  def test_invalid_weight_range(self):
    """Test that weight outside [0,1] raises error."""
    with self.assertRaises(ValueError) as ctx:
      gt.NSMField(weights={"FEEL": 1.5})
    self.assertIn("[0, 1]", str(ctx.exception))

  def test_to_vector(self):
    """Test conversion to vector."""
    nsm = gt.NSMField.from_text("I feel good")
    vec = nsm.to_vector(dimension=1000)
    self.assertEqual(len(vec), 1000)
    self.assertEqual(vec.dtype, np.float32)

  def test_dominant_primitives(self):
    """Test extraction of dominant primitives."""
    nsm = gt.NSMField(weights={"FEEL": 0.8, "WANT": 0.6, "KNOW": 0.3})
    dominant = nsm.dominant_primitives
    self.assertIn("FEEL", dominant)
    self.assertIn("WANT", dominant)
    self.assertNotIn("KNOW", dominant)


class CausalityFlowTest(absltest.TestCase):

  def test_from_markers_caused_by(self):
    """Test causality flow for 'caused_by' marker."""
    flow = gt.CausalityFlow.from_markers(caused_by=True)
    self.assertGreater(flow.agency, 0.5)  # Positive = caused

  def test_from_markers_causes(self):
    """Test causality flow for 'causes' marker."""
    flow = gt.CausalityFlow.from_markers(causes=True)
    self.assertLess(flow.agency, -0.5)  # Negative = causing

  def test_from_markers_temporal(self):
    """Test causality flow for temporal markers."""
    before = gt.CausalityFlow.from_markers(temporal_marker="before")
    after = gt.CausalityFlow.from_markers(temporal_marker="after")
    self.assertLess(before.temporality, 0)
    self.assertGreater(after.temporality, 0)

  def test_from_markers_dependency(self):
    """Test causality flow for dependency markers."""
    depends = gt.CausalityFlow.from_markers(depends_on=True)
    grounds = gt.CausalityFlow.from_markers(grounds=True)
    self.assertGreater(depends.dependency, 0.5)
    self.assertLess(grounds.dependency, -0.5)

  def test_flow_shape(self):
    """Test that flow is 3D."""
    flow = gt.CausalityFlow.from_markers()
    self.assertEqual(flow.flow.shape, (3,))

  def test_invalid_flow_shape(self):
    """Test that invalid flow shape raises error."""
    with self.assertRaises(ValueError) as ctx:
      gt.CausalityFlow(flow=np.zeros(5))
    self.assertIn("3D", str(ctx.exception))

  def test_to_vector(self):
    """Test expansion to higher dimension."""
    flow = gt.CausalityFlow.from_markers(causes=True, temporal_marker="after")
    vec = flow.to_vector(dimension=1000)
    self.assertEqual(len(vec), 1000)


class QualiaFieldTest(absltest.TestCase):

  def test_from_text_positive(self):
    """Test qualia extraction for positive text."""
    qualia = gt.QualiaField.from_text("This is great and wonderful!")
    self.assertGreater(qualia.values.get("valence", 0), 0)

  def test_from_text_negative(self):
    """Test qualia extraction for negative text."""
    qualia = gt.QualiaField.from_text("This is terrible and awful")
    self.assertLess(qualia.values.get("valence", 0), 0)

  def test_from_text_urgent(self):
    """Test qualia extraction for urgent text."""
    qualia = gt.QualiaField.from_text("URGENT! Do this immediately!")
    self.assertGreater(qualia.values.get("urgency", 0), 0)
    self.assertGreater(qualia.values.get("arousal", 0), 0)

  def test_from_text_formal(self):
    """Test qualia extraction for formal text."""
    qualia = gt.QualiaField.from_text("Hereby pursuant to the agreement")
    self.assertGreater(qualia.values.get("formality", 0), 0)

  def test_from_text_casual(self):
    """Test qualia extraction for casual text."""
    qualia = gt.QualiaField.from_text("Hey yeah that's cool!")
    self.assertLess(qualia.values.get("formality", 0), 0)

  def test_invalid_dimension(self):
    """Test that invalid dimension raises error."""
    with self.assertRaises(ValueError) as ctx:
      gt.QualiaField(values={"invalid_dimension": 0.5})
    self.assertIn("Unknown qualia dimension", str(ctx.exception))

  def test_invalid_value_range(self):
    """Test that value outside [-1,1] raises error."""
    with self.assertRaises(ValueError) as ctx:
      gt.QualiaField(values={"valence": 2.0})
    self.assertIn("[-1, 1]", str(ctx.exception))

  def test_to_vector(self):
    """Test conversion to 18D vector."""
    qualia = gt.QualiaField.from_text("test")
    vec = qualia.to_vector()
    self.assertEqual(len(vec), len(gt.QUALIA_DIMENSIONS))

  def test_to_expanded_vector(self):
    """Test expansion to higher dimension."""
    qualia = gt.QualiaField.from_text("test")
    vec = qualia.to_expanded_vector(dimension=1000)
    self.assertEqual(len(vec), 1000)


class GrammarTriangleFieldTest(absltest.TestCase):

  def test_from_extraction_basic(self):
    """Test creating grammar triangle from basic extraction."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="I feel happy about this",
    )
    triangle = gt.GrammarTriangleField.from_extraction(ext)

    self.assertIsInstance(triangle.nsm_field, gt.NSMField)
    self.assertIsInstance(triangle.causality_flow, gt.CausalityFlow)
    self.assertIsInstance(triangle.qualia_field, gt.QualiaField)
    self.assertIsInstance(triangle.awareness_vector, baf_vector.HyperVector)

  def test_from_extraction_with_attributes(self):
    """Test grammar triangle uses extraction attributes."""
    ext = data.Extraction(
        extraction_class="action",
        extraction_text="caused the change",
        attributes={
            "caused_by": "true",
            "temporal_marker": "before",
        },
    )
    triangle = gt.GrammarTriangleField.from_extraction(ext)

    # Should detect causal markers from attributes
    self.assertGreater(triangle.causality_flow.agency, 0)

  def test_from_extraction_with_base_embedding(self):
    """Test grammar triangle with base embedding."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    base_embedding = np.random.randn(1024).astype(np.float32)
    triangle = gt.GrammarTriangleField.from_extraction(
        ext, base_embedding=base_embedding
    )

    self.assertIsNotNone(triangle.awareness_vector)
    self.assertEqual(triangle.awareness_vector.dimension, baf_vector.DEFAULT_DIM)

  def test_awareness_vector_dimension(self):
    """Test custom dimension for awareness vector."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    triangle = gt.GrammarTriangleField.from_extraction(ext, dimension=5000)

    self.assertEqual(triangle.awareness_vector.dimension, 5000)

  def test_to_dict_roundtrip(self):
    """Test serialization roundtrip."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="I want to know",
    )
    original = gt.GrammarTriangleField.from_extraction(ext, dimension=1000)
    d = original.to_dict()
    restored = gt.GrammarTriangleField.from_dict(d, dimension=1000)

    # Check NSM weights preserved
    for prim, weight in original.nsm_field.weights.items():
      self.assertAlmostEqual(weight, restored.nsm_field.weights[prim], places=5)

    # Check causality flow preserved
    np.testing.assert_array_almost_equal(
        original.causality_flow.flow, restored.causality_flow.flow
    )

    # Check qualia preserved
    for dim, val in original.qualia_field.values.items():
      self.assertAlmostEqual(val, restored.qualia_field.values[dim], places=5)


class ExtractionBAFIntegrationTest(absltest.TestCase):
  """Test BAF integration with Extraction dataclass."""

  def test_compute_baf_components(self):
    """Test computing BAF components for an extraction."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="I feel excited about this project",
        alignment_status=data.AlignmentStatus.MATCH_EXACT,
    )
    ext.compute_baf_components()

    # Check all components computed
    self.assertIsNotNone(ext.grammar_triangle)
    self.assertIsNotNone(ext.baf_vector)
    self.assertIsNotNone(ext.collapse_resistance)
    self.assertIsNotNone(ext.uncertainty_tensor)

  def test_collapse_resistance_range(self):
    """Test that collapse resistance is in valid range."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    ext.compute_baf_components()

    self.assertGreaterEqual(ext.collapse_resistance, 0.0)
    self.assertLessEqual(ext.collapse_resistance, 1.0)

  def test_compute_baf_with_base_embedding(self):
    """Test computing BAF with external embedding."""
    ext = data.Extraction(
        extraction_class="entity",
        extraction_text="test",
    )
    base_embedding = np.random.randn(1024).astype(np.float32)
    ext.compute_baf_components(base_embedding=base_embedding)

    self.assertIsNotNone(ext.baf_vector)

  def test_compute_baf_with_context_ambiguity(self):
    """Test that context_ambiguity affects collapse resistance."""
    ext1 = data.Extraction(extraction_class="entity", extraction_text="test")
    ext2 = data.Extraction(extraction_class="entity", extraction_text="test")

    ext1.compute_baf_components(context_ambiguity=0.1)
    ext2.compute_baf_components(context_ambiguity=0.9)

    # Higher ambiguity should lead to higher resistance
    self.assertGreater(ext2.collapse_resistance, ext1.collapse_resistance)


if __name__ == "__main__":
  absltest.main()
