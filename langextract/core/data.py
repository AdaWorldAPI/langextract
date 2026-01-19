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

"""Classes used to represent core data types of annotation pipeline."""
from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING
import uuid

from langextract.core import tokenizer
from langextract.core import types

if TYPE_CHECKING:
    from langextract.core import baf_vector
    from langextract.core import collapse_resistance
    from langextract.core import grammar_triangle

FormatType = types.FormatType  # Backward compat

EXTRACTIONS_KEY = "extractions"
ATTRIBUTE_SUFFIX = "_attributes"

__all__ = [
    "AlignmentStatus",
    "CharInterval",
    "Extraction",
    "Document",
    "AnnotatedDocument",
    "ExampleData",
    "FormatType",
    "EXTRACTIONS_KEY",
    "ATTRIBUTE_SUFFIX",
]


class AlignmentStatus(enum.Enum):
  MATCH_EXACT = "match_exact"
  MATCH_GREATER = "match_greater"
  MATCH_LESSER = "match_lesser"
  MATCH_FUZZY = "match_fuzzy"


@dataclasses.dataclass
class CharInterval:
  """Class for representing a character interval.

  Attributes:
    start_pos: The starting position of the interval (inclusive).
    end_pos: The ending position of the interval (exclusive).
  """

  start_pos: int | None = None
  end_pos: int | None = None


@dataclasses.dataclass(init=False)
class Extraction:
  """Represents an extraction extracted from text.

  This class encapsulates an extraction's characteristics and its position
  within the source text. It can represent a diverse range of information for
  NLP information extraction tasks.

  Attributes:
    extraction_class: The class of the extraction.
    extraction_text: The text of the extraction.
    char_interval: The character interval of the extraction in the original
      text.
    alignment_status: The alignment status of the extraction.
    extraction_index: The index of the extraction in the list of extractions.
    group_index: The index of the group the extraction belongs to.
    description: A description of the extraction.
    attributes: A list of attributes of the extraction.
    token_interval: The token interval of the extraction.
  """

  extraction_class: str
  extraction_text: str
  char_interval: CharInterval | None = None
  alignment_status: AlignmentStatus | None = None
  extraction_index: int | None = None
  group_index: int | None = None
  description: str | None = None
  attributes: dict[str, str | list[str]] | None = None
  _token_interval: tokenizer.TokenInterval | None = dataclasses.field(
      default=None, repr=False, compare=False
  )
  # BAF (Bipolar Awareness Field) components for VSA integration
  _baf_vector: "baf_vector.HyperVector | None" = dataclasses.field(
      default=None, repr=False, compare=False
  )
  _collapse_resistance: float | None = dataclasses.field(
      default=None, repr=False, compare=False
  )
  _uncertainty_tensor: "collapse_resistance.UncertaintyTensor | None" = dataclasses.field(
      default=None, repr=False, compare=False
  )
  _grammar_triangle: "grammar_triangle.GrammarTriangleField | None" = dataclasses.field(
      default=None, repr=False, compare=False
  )

  def __init__(
      self,
      extraction_class: str,
      extraction_text: str,
      *,
      token_interval: tokenizer.TokenInterval | None = None,
      char_interval: CharInterval | None = None,
      alignment_status: AlignmentStatus | None = None,
      extraction_index: int | None = None,
      group_index: int | None = None,
      description: str | None = None,
      attributes: dict[str, str | list[str]] | None = None,
      baf_vector: "baf_vector.HyperVector | None" = None,
      collapse_resistance: float | None = None,
      uncertainty_tensor: "collapse_resistance.UncertaintyTensor | None" = None,
      grammar_triangle: "grammar_triangle.GrammarTriangleField | None" = None,
  ):
    self.extraction_class = extraction_class
    self.extraction_text = extraction_text
    self.char_interval = char_interval
    self._token_interval = token_interval
    self.alignment_status = alignment_status
    self.extraction_index = extraction_index
    self.group_index = group_index
    self.description = description
    self.attributes = attributes
    # BAF components
    self._baf_vector = baf_vector
    self._collapse_resistance = collapse_resistance
    self._uncertainty_tensor = uncertainty_tensor
    self._grammar_triangle = grammar_triangle

  @property
  def token_interval(self) -> tokenizer.TokenInterval | None:
    return self._token_interval

  @token_interval.setter
  def token_interval(self, value: tokenizer.TokenInterval | None) -> None:
    self._token_interval = value

  @property
  def baf_vector(self) -> "baf_vector.HyperVector | None":
    """Return the BAF (Bipolar Awareness Field) hypervector."""
    return self._baf_vector

  @baf_vector.setter
  def baf_vector(self, value: "baf_vector.HyperVector | None") -> None:
    """Set the BAF hypervector."""
    self._baf_vector = value

  @property
  def collapse_resistance(self) -> float | None:
    """Return the collapse resistance value (0-1).

    Collapse resistance determines how much the extraction should resist
    being bound to a fixed semantic role. High resistance (close to 1.0)
    preserves superposition; low resistance (close to 0.0) allows binding.
    """
    return self._collapse_resistance

  @collapse_resistance.setter
  def collapse_resistance(self, value: float | None) -> None:
    """Set the collapse resistance value."""
    if value is not None and not 0 <= value <= 1:
      raise ValueError(f"collapse_resistance must be in [0, 1], got {value}")
    self._collapse_resistance = value

  @property
  def uncertainty_tensor(self) -> "collapse_resistance.UncertaintyTensor | None":
    """Return the 3x3 epistemic uncertainty tensor."""
    return self._uncertainty_tensor

  @uncertainty_tensor.setter
  def uncertainty_tensor(self, value: "collapse_resistance.UncertaintyTensor | None") -> None:
    """Set the uncertainty tensor."""
    self._uncertainty_tensor = value

  @property
  def grammar_triangle(self) -> "grammar_triangle.GrammarTriangleField | None":
    """Return the Grammar Triangle Field (NSM, causality, qualia)."""
    return self._grammar_triangle

  @grammar_triangle.setter
  def grammar_triangle(self, value: "grammar_triangle.GrammarTriangleField | None") -> None:
    """Set the Grammar Triangle Field."""
    self._grammar_triangle = value

  def compute_baf_components(
      self,
      base_embedding: "baf_vector.np.ndarray | None" = None,
      context_ambiguity: float = 0.5,
  ) -> None:
    """Compute all BAF components for this extraction.

    This computes:
    - Grammar Triangle Field (NSM, causality, qualia)
    - Collapse resistance
    - Uncertainty tensor
    - BAF hypervector

    Args:
      base_embedding: Optional dense embedding vector (e.g., from Jina).
      context_ambiguity: External measure of contextual ambiguity [0, 1].
    """
    from langextract.core import collapse_resistance as cr
    from langextract.core import grammar_triangle as gt

    # Compute Grammar Triangle Field
    self._grammar_triangle = gt.GrammarTriangleField.from_extraction(
        self, base_embedding=base_embedding
    )

    # Extract awareness vector from triangle
    if self._grammar_triangle.awareness_vector is not None:
      self._baf_vector = self._grammar_triangle.awareness_vector

    # Compute collapse resistance
    self._collapse_resistance = cr.compute_collapse_resistance(
        self, context_ambiguity=context_ambiguity
    )

    # Compute uncertainty tensor
    self._uncertainty_tensor = cr.compute_uncertainty_tensor(self)


@dataclasses.dataclass
class Document:
  """Document class for annotating documents.

  Attributes:
    text: Raw text representation for the document.
    document_id: Unique identifier for each document and is auto-generated if
      not set.
    additional_context: Additional context to supplement prompt instructions.
    tokenized_text: Tokenized text for the document, computed from `text`.
  """

  text: str
  additional_context: str | None = None
  _document_id: str | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _tokenized_text: tokenizer.TokenizedText | None = dataclasses.field(
      init=False, default=None, repr=False, compare=False
  )

  def __init__(
      self,
      text: str,
      *,
      document_id: str | None = None,
      additional_context: str | None = None,
  ):
    self.text = text
    self.additional_context = additional_context
    self._document_id = document_id

  @property
  def document_id(self) -> str:
    """Returns the document ID, generating a unique one if not set."""
    if self._document_id is None:
      self._document_id = f"doc_{uuid.uuid4().hex[:8]}"
    return self._document_id

  @document_id.setter
  def document_id(self, value: str | None) -> None:
    """Sets the document ID."""
    self._document_id = value

  @property
  def tokenized_text(self) -> tokenizer.TokenizedText:
    if self._tokenized_text is None:
      self._tokenized_text = tokenizer.tokenize(self.text)
    return self._tokenized_text

  @tokenized_text.setter
  def tokenized_text(self, value: tokenizer.TokenizedText) -> None:
    self._tokenized_text = value


@dataclasses.dataclass
class AnnotatedDocument:
  """Class for representing annotated documents.

  Attributes:
    document_id: Unique identifier for each document - autogenerated if not
      set.
    extractions: List of extractions in the document.
    text: Raw text representation of the document.
    tokenized_text: Tokenized text of the document, computed from `text`.
  """

  extractions: list[Extraction] | None = None
  text: str | None = None
  _document_id: str | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _tokenized_text: tokenizer.TokenizedText | None = dataclasses.field(
      init=False, default=None, repr=False, compare=False
  )

  def __init__(
      self,
      *,
      document_id: str | None = None,
      extractions: list[Extraction] | None = None,
      text: str | None = None,
  ):
    self.extractions = extractions
    self.text = text
    self._document_id = document_id

  @property
  def document_id(self) -> str:
    """Returns the document ID, generating a unique one if not set."""
    if self._document_id is None:
      self._document_id = f"doc_{uuid.uuid4().hex[:8]}"
    return self._document_id

  @document_id.setter
  def document_id(self, value: str | None) -> None:
    """Sets the document ID."""
    self._document_id = value

  @property
  def tokenized_text(self) -> tokenizer.TokenizedText | None:
    if self._tokenized_text is None and self.text is not None:
      self._tokenized_text = tokenizer.tokenize(self.text)
    return self._tokenized_text

  @tokenized_text.setter
  def tokenized_text(self, value: tokenizer.TokenizedText) -> None:
    self._tokenized_text = value


@dataclasses.dataclass
class ExampleData:
  """A single training/example data instance for a structured prompting.

  Attributes:
    text: The raw input text (sentence, paragraph, etc.).
    extractions: A list of Extraction objects extracted from the text.
  """

  text: str
  extractions: list[Extraction] = dataclasses.field(default_factory=list)
