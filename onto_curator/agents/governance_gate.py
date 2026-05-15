"""
GovernanceGate Agent — enforces approval workflows by classifying
each term mapping into auto_approve / human_review / escalate
based on confidence score, entity type, and conflict status.
"""

from __future__ import annotations

import logging
from typing import Optional

from onto_curator.models.schemas import (
    ConflictReport,
    GovernanceAction,
    GovernanceDecision,
    TermMapping,
)

logger = logging.getLogger(__name__)

# Domain steward routing table  {entity_type → steward role}
_STEWARD_MAP: dict[str, str] = {
    "assay": "assay_ontology_steward",
    "molecule": "chemistry_ontology_steward",
    "cell_type": "biology_ontology_steward",
    "organism": "biology_ontology_steward",
    "process": "process_ontology_steward",
    "gene": "genomics_ontology_steward",
    "protein": "genomics_ontology_steward",
    "disease": "clinical_ontology_steward",
    "instrument": "laboratory_operations_steward",
    "measurement": "assay_ontology_steward",
    "other": "general_ontology_steward",
}


class GovernanceGateAgent:
    """
    Routes each TermMapping to the appropriate governance action.

    Decision logic
    --------------
    confidence ≥ high_threshold AND no conflicts  → auto_approve
    confidence ≥ low_threshold                    → human_review
    confidence < low_threshold OR new term needed
      OR critical conflict present                → escalate

    Parameters
    ----------
    high_threshold : float
        Minimum confidence for auto-approval (default: 0.88).
    low_threshold : float
        Minimum confidence for human review (default: 0.60).
    """

    def __init__(
        self,
        high_threshold: float = 0.88,
        low_threshold: float = 0.60,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def decide_batch(
        self,
        mappings: list[TermMapping],
        conflict_report: ConflictReport,
    ) -> list[GovernanceDecision]:
        """
        Produce a GovernanceDecision for each mapping in ``mappings``.

        Parameters
        ----------
        mappings : list[TermMapping]
        conflict_report : ConflictReport
            Result from ConflictResolverAgent for the same batch.

        Returns
        -------
        list[GovernanceDecision]
        """
        # Index terms that are part of a conflict
        conflicted_terms: set[str] = set()
        for item in conflict_report.conflicts:
            conflicted_terms.add(item.term_a.lower())
            conflicted_terms.add(item.term_b.lower())

        decisions: list[GovernanceDecision] = []
        for mapping in mappings:
            decision = self._decide_single(mapping, conflicted_terms)
            decisions.append(decision)

        auto = sum(1 for d in decisions if d.action == GovernanceAction.AUTO_APPROVE)
        review = sum(1 for d in decisions if d.action == GovernanceAction.HUMAN_REVIEW)
        escalate = sum(1 for d in decisions if d.action == GovernanceAction.ESCALATE)
        logger.info(
            "GovernanceGate: %d auto-approved | %d human-review | %d escalated",
            auto, review, escalate,
        )
        return decisions

    def _decide_single(
        self,
        mapping: TermMapping,
        conflicted_terms: set[str],
    ) -> GovernanceDecision:
        conf = mapping.mapping_confidence
        term_lower = mapping.candidate.text.lower()
        entity_type = mapping.candidate.entity_type
        steward = _STEWARD_MAP.get(str(entity_type), "general_ontology_steward")

        # --- Escalation conditions ---
        if mapping.needs_new_term:
            return GovernanceDecision(
                term_text=mapping.candidate.text,
                action=GovernanceAction.ESCALATE,
                reason="No suitable existing ontology term found; new term proposal required.",
                confidence_used=conf,
                assigned_steward=steward,
                priority="high",
            )

        if term_lower in conflicted_terms:
            return GovernanceDecision(
                term_text=mapping.candidate.text,
                action=GovernanceAction.ESCALATE,
                reason="Term is involved in a detected conflict; requires expert resolution.",
                confidence_used=conf,
                assigned_steward=steward,
                priority="high",
            )

        if conf < self.low_threshold:
            return GovernanceDecision(
                term_text=mapping.candidate.text,
                action=GovernanceAction.ESCALATE,
                reason=f"Mapping confidence {conf:.2f} is below minimum threshold {self.low_threshold:.2f}.",
                confidence_used=conf,
                assigned_steward=steward,
                priority="critical" if conf < 0.30 else "high",
            )

        # --- Human review ---
        if conf < self.high_threshold:
            return GovernanceDecision(
                term_text=mapping.candidate.text,
                action=GovernanceAction.HUMAN_REVIEW,
                reason=(
                    f"Mapping confidence {conf:.2f} is between thresholds "
                    f"({self.low_threshold:.2f}–{self.high_threshold:.2f}); "
                    "routed for expert validation."
                ),
                confidence_used=conf,
                assigned_steward=steward,
                priority="medium",
            )

        # --- Auto-approve ---
        return GovernanceDecision(
            term_text=mapping.candidate.text,
            action=GovernanceAction.AUTO_APPROVE,
            reason=(
                f"Mapping confidence {conf:.2f} exceeds auto-approval threshold "
                f"{self.high_threshold:.2f} with no active conflicts."
            ),
            confidence_used=conf,
            assigned_steward=None,
            priority="low",
        )
