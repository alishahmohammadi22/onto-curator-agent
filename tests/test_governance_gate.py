"""Tests for GovernanceGateAgent."""

from __future__ import annotations

import pytest

from onto_curator.agents.governance_gate import GovernanceGateAgent
from onto_curator.models.schemas import (
    CandidateTerm,
    ConflictItem,
    ConflictReport,
    ConflictType,
    EntityType,
    GovernanceAction,
    OntologyMatch,
    TermMapping,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mapping(
    text: str,
    confidence: float,
    entity_type: EntityType = EntityType.ASSAY,
    needs_new_term: bool = False,
) -> TermMapping:
    candidate = CandidateTerm(
        text=text,
        normalized_text=text.lower(),
        context=f"Context sentence containing {text}.",
        entity_type=entity_type,
        confidence=0.9,
    )
    top_match = None if needs_new_term else OntologyMatch(
        ontology_id=f"OBI:000{hash(text) % 9000 + 1000}",
        ontology_name="OBI",
        matched_label=text,
        match_score=confidence,
        match_source="OLS",
    )
    return TermMapping(
        candidate=candidate,
        top_match=top_match,
        mapping_confidence=confidence,
        needs_new_term=needs_new_term,
    )


def _empty_conflict_report() -> ConflictReport:
    return ConflictReport(has_conflicts=False)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGovernanceGateAgent:

    def setup_method(self):
        self.gate = GovernanceGateAgent(high_threshold=0.88, low_threshold=0.60)

    def test_auto_approve_high_confidence(self):
        mapping = _make_mapping("trypan blue exclusion", confidence=0.95)
        decisions = self.gate.decide_batch([mapping], _empty_conflict_report())
        assert decisions[0].action == GovernanceAction.AUTO_APPROVE

    def test_human_review_medium_confidence(self):
        mapping = _make_mapping("flow cytometry panel", confidence=0.75)
        decisions = self.gate.decide_batch([mapping], _empty_conflict_report())
        assert decisions[0].action == GovernanceAction.HUMAN_REVIEW

    def test_escalate_low_confidence(self):
        mapping = _make_mapping("some vague term", confidence=0.45)
        decisions = self.gate.decide_batch([mapping], _empty_conflict_report())
        assert decisions[0].action == GovernanceAction.ESCALATE

    def test_escalate_new_term_needed(self):
        mapping = _make_mapping("novel CAR construct", confidence=0.92, needs_new_term=True)
        decisions = self.gate.decide_batch([mapping], _empty_conflict_report())
        assert decisions[0].action == GovernanceAction.ESCALATE
        assert "new term" in decisions[0].reason.lower()

    def test_escalate_conflicted_term(self):
        mapping = _make_mapping("viability assay", confidence=0.93)
        conflict_report = ConflictReport(
            has_conflicts=True,
            conflict_count=1,
            conflicts=[
                ConflictItem(
                    conflict_type=ConflictType.NEAR_SYNONYM,
                    term_a="viability assay",
                    term_b="cell viability assay",
                    description="Near synonyms mapped to different ontology IDs",
                )
            ],
        )
        decisions = self.gate.decide_batch([mapping], conflict_report)
        assert decisions[0].action == GovernanceAction.ESCALATE

    def test_batch_returns_decision_per_mapping(self):
        mappings = [
            _make_mapping("CD3+ T cells", 0.97, EntityType.CELL_TYPE),
            _make_mapping("kLa measurement", 0.72, EntityType.MEASUREMENT),
            _make_mapping("new biomarker X", 0.91, needs_new_term=True),
        ]
        decisions = self.gate.decide_batch(mappings, _empty_conflict_report())
        assert len(decisions) == 3
        assert decisions[0].action == GovernanceAction.AUTO_APPROVE
        assert decisions[1].action == GovernanceAction.HUMAN_REVIEW
        assert decisions[2].action == GovernanceAction.ESCALATE

    def test_steward_assigned_for_human_review(self):
        mapping = _make_mapping("KRAS mutation", 0.75, EntityType.GENE)
        decisions = self.gate.decide_batch([mapping], _empty_conflict_report())
        assert decisions[0].assigned_steward == "genomics_ontology_steward"

    def test_no_steward_for_auto_approve(self):
        mapping = _make_mapping("LAL assay", 0.95, EntityType.ASSAY)
        decisions = self.gate.decide_batch([mapping], _empty_conflict_report())
        assert decisions[0].assigned_steward is None

    def test_empty_mappings_returns_empty(self):
        decisions = self.gate.decide_batch([], _empty_conflict_report())
        assert decisions == []

    def test_critical_priority_very_low_confidence(self):
        mapping = _make_mapping("unclear term", 0.20)
        decisions = self.gate.decide_batch([mapping], _empty_conflict_report())
        assert decisions[0].priority == "critical"
