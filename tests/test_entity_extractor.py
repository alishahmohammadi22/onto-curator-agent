"""Tests for EntityExtractorAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from onto_curator.agents.entity_extractor import EntityExtractorAgent
from onto_curator.models.schemas import CandidateTerm, EntityType


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Viable cell density was determined by trypan blue exclusion using a Vi-CELL XR. "
    "CD3+ T cells were transduced with a lentiviral CAR construct at MOI 5. "
    "Cytotoxicity against Raji cells was assessed at an effector-to-target ratio of 5:1. "
    "Endotoxin levels were measured by LAL assay and confirmed < 0.5 EU/mL."
)

MOCK_RESPONSE_JSON = json.dumps({
    "entities": [
        {
            "text": "viable cell density",
            "normalized_text": "viable cell density",
            "context": "Viable cell density was determined by trypan blue exclusion.",
            "entity_type": "measurement",
            "confidence": 0.95,
        },
        {
            "text": "trypan blue exclusion",
            "normalized_text": "trypan blue exclusion",
            "context": "determined by trypan blue exclusion using a Vi-CELL XR",
            "entity_type": "assay",
            "confidence": 0.92,
        },
        {
            "text": "CD3+ T cells",
            "normalized_text": "cd3+ t cells",
            "context": "CD3+ T cells were transduced with a lentiviral CAR construct",
            "entity_type": "cell_type",
            "confidence": 0.97,
        },
        {
            "text": "CAR construct",
            "normalized_text": "car construct",
            "context": "transduced with a lentiviral CAR construct at MOI 5",
            "entity_type": "molecule",
            "confidence": 0.78,
        },
        {
            "text": "LAL assay",
            "normalized_text": "lal assay",
            "context": "Endotoxin levels were measured by LAL assay",
            "entity_type": "assay",
            "confidence": 0.91,
        },
    ]
})


def _make_mock_client(response_content: str) -> MagicMock:
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = response_content
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEntityExtractorAgent:

    def test_extracts_expected_number_of_terms(self):
        client = _make_mock_client(MOCK_RESPONSE_JSON)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2")
        terms = agent.extract(SAMPLE_TEXT)
        assert len(terms) == 5

    def test_terms_sorted_by_confidence_descending(self):
        client = _make_mock_client(MOCK_RESPONSE_JSON)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2")
        terms = agent.extract(SAMPLE_TEXT)
        confidences = [t.confidence for t in terms]
        assert confidences == sorted(confidences, reverse=True)

    def test_returns_candidate_term_objects(self):
        client = _make_mock_client(MOCK_RESPONSE_JSON)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2")
        terms = agent.extract(SAMPLE_TEXT)
        assert all(isinstance(t, CandidateTerm) for t in terms)

    def test_min_confidence_filter(self):
        client = _make_mock_client(MOCK_RESPONSE_JSON)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2", min_confidence=0.93)
        terms = agent.extract(SAMPLE_TEXT)
        # Only CD3+ T cells (0.97) and viable cell density (0.95) pass
        assert all(t.confidence >= 0.93 for t in terms)
        assert len(terms) == 2

    def test_entity_types_valid(self):
        client = _make_mock_client(MOCK_RESPONSE_JSON)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2")
        terms = agent.extract(SAMPLE_TEXT)
        valid_types = {e.value for e in EntityType}
        for t in terms:
            assert t.entity_type in valid_types

    def test_unknown_entity_type_defaults_to_other(self):
        bad_json = json.dumps({
            "entities": [{
                "text": "some term",
                "normalized_text": "some term",
                "context": "context",
                "entity_type": "COMPLETELY_INVALID_TYPE",
                "confidence": 0.8,
            }]
        })
        client = _make_mock_client(bad_json)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2")
        terms = agent.extract("some term")
        assert terms[0].entity_type == EntityType.OTHER

    def test_source_document_propagated(self):
        client = _make_mock_client(MOCK_RESPONSE_JSON)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2")
        terms = agent.extract(SAMPLE_TEXT, source_document="DOC-001")
        assert all(t.source_document == "DOC-001" for t in terms)

    def test_handles_plain_list_response(self):
        """API response might be a plain JSON array instead of wrapped object."""
        plain_list = json.dumps([{
            "text": "trypan blue",
            "normalized_text": "trypan blue",
            "context": "trypan blue exclusion assay",
            "entity_type": "assay",
            "confidence": 0.9,
        }])
        client = _make_mock_client(plain_list)
        agent = EntityExtractorAgent(client=client, model="gpt-5.2")
        terms = agent.extract("trypan blue exclusion assay")
        assert len(terms) == 1
        assert terms[0].text == "trypan blue"
