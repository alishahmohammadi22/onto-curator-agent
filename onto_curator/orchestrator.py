"""
LangGraph-based orchestrator for the OntoCurator multi-agent pipeline.

Graph topology (linear with error edge):

  extract → map → resolve → govern → END
     ↓ (error)          ↓ (error)
    END                 END
"""

from __future__ import annotations

import logging
import operator
import uuid
from typing import Annotated, Any, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from onto_curator.agents.conflict_resolver import ConflictResolverAgent
from onto_curator.agents.entity_extractor import EntityExtractorAgent
from onto_curator.agents.governance_gate import GovernanceGateAgent
from onto_curator.agents.term_mapper import TermMapperAgent
from onto_curator.models.schemas import (
    ConflictReport,
    CurationResult,
    GovernanceAction,
    TermMapping,
)

logger = logging.getLogger(__name__)


# ─── LangGraph State ──────────────────────────────────────────────────────────

class CurationState(TypedDict):
    run_id: str
    input_text: str
    source_document: Optional[str]
    candidate_terms: list
    term_mappings: list
    conflict_report: Optional[dict]
    curation_results: list
    errors: Annotated[list[str], operator.add]


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class OntoCuratorOrchestrator:
    """
    Wires the four agents into a LangGraph state machine.

    Parameters
    ----------
    extractor : EntityExtractorAgent
    mapper : TermMapperAgent
    resolver : ConflictResolverAgent
    gate : GovernanceGateAgent
    """

    def __init__(
        self,
        extractor: EntityExtractorAgent,
        mapper: TermMapperAgent,
        resolver: ConflictResolverAgent,
        gate: GovernanceGateAgent,
    ) -> None:
        self.extractor = extractor
        self.mapper = mapper
        self.resolver = resolver
        self.gate = gate
        self._graph = self._build_graph()

    # ── Node functions ────────────────────────────────────────────────────────

    def _node_extract(self, state: CurationState) -> dict[str, Any]:
        """EntityExtractor node."""
        try:
            terms = self.extractor.extract(
                state["input_text"],
                source_document=state.get("source_document"),
            )
            logger.info("[extract] %d candidate terms", len(terms))
            return {"candidate_terms": [t.model_dump() for t in terms]}
        except Exception as exc:
            logger.error("[extract] failed: %s", exc)
            return {"candidate_terms": [], "errors": [f"extract: {exc}"]}

    def _node_map(self, state: CurationState) -> dict[str, Any]:
        """TermMapper node."""
        from onto_curator.models.schemas import CandidateTerm

        raw_terms = state.get("candidate_terms", [])
        if not raw_terms:
            return {"term_mappings": []}

        terms = [CandidateTerm(**t) for t in raw_terms]
        try:
            mappings = self.mapper.map_terms(terms)
            logger.info("[map] %d mappings produced", len(mappings))
            return {"term_mappings": [m.model_dump() for m in mappings]}
        except Exception as exc:
            logger.error("[map] failed: %s", exc)
            return {"term_mappings": [], "errors": [f"map: {exc}"]}

    def _node_resolve(self, state: CurationState) -> dict[str, Any]:
        """ConflictResolver node."""
        raw_mappings = state.get("term_mappings", [])
        if not raw_mappings:
            empty = ConflictReport(has_conflicts=False)
            return {"conflict_report": empty.model_dump()}

        mappings = [TermMapping(**m) for m in raw_mappings]
        try:
            report = self.resolver.resolve(mappings)
            logger.info("[resolve] conflicts found: %s", report.has_conflicts)
            return {"conflict_report": report.model_dump()}
        except Exception as exc:
            logger.error("[resolve] failed: %s", exc)
            empty = ConflictReport(has_conflicts=False)
            return {"conflict_report": empty.model_dump(), "errors": [f"resolve: {exc}"]}

    def _node_govern(self, state: CurationState) -> dict[str, Any]:
        """GovernanceGate node — produces final CurationResult list."""
        raw_mappings = state.get("term_mappings", [])
        raw_report = state.get("conflict_report") or {}

        mappings = [TermMapping(**m) for m in raw_mappings]
        conflict_report = ConflictReport(**raw_report) if raw_report else ConflictReport(has_conflicts=False)

        decisions = self.gate.decide_batch(mappings, conflict_report)

        results: list[dict] = []
        for mapping, decision in zip(mappings, decisions):
            status_map = {
                GovernanceAction.AUTO_APPROVE: "approved",
                GovernanceAction.HUMAN_REVIEW: "pending_review",
                GovernanceAction.ESCALATE: "escalated",
            }
            result = CurationResult(
                mapping=mapping,
                governance=decision,
                audit_trail=[
                    {"step": "extract", "agent": "EntityExtractorAgent"},
                    {"step": "map", "agent": "TermMapperAgent", "top_match": (
                        mapping.top_match.ontology_id if mapping.top_match else None
                    )},
                    {"step": "resolve", "agent": "ConflictResolverAgent",
                     "conflicts_detected": conflict_report.has_conflicts},
                    {"step": "govern", "agent": "GovernanceGateAgent",
                     "action": decision.action},
                ],
                final_ontology_id=(
                    mapping.top_match.ontology_id if mapping.top_match and
                    decision.action == GovernanceAction.AUTO_APPROVE else None
                ),
                status=status_map.get(decision.action, "pending_review"),
            )
            results.append(result.model_dump())

        logger.info("[govern] %d curation results assembled", len(results))
        return {"curation_results": results}

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(CurationState)

        g.add_node("extract", self._node_extract)
        g.add_node("map", self._node_map)
        g.add_node("resolve", self._node_resolve)
        g.add_node("govern", self._node_govern)

        g.set_entry_point("extract")
        g.add_edge("extract", "map")
        g.add_edge("map", "resolve")
        g.add_edge("resolve", "govern")
        g.add_edge("govern", END)

        return g.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, input_text: str, source_document: Optional[str] = None) -> CurationState:
        """
        Run the full curation pipeline on ``input_text``.

        Parameters
        ----------
        input_text : str
            Raw scientific text to curate.
        source_document : str, optional
            Source document ID for provenance.

        Returns
        -------
        CurationState
            Final state dict with all intermediate and final results.
        """
        run_id = str(uuid.uuid4())[:8]
        logger.info("OntoCurator run %s starting", run_id)

        initial_state: CurationState = {
            "run_id": run_id,
            "input_text": input_text,
            "source_document": source_document,
            "candidate_terms": [],
            "term_mappings": [],
            "conflict_report": None,
            "curation_results": [],
            "errors": [],
        }

        final_state = self._graph.invoke(initial_state)
        logger.info(
            "OntoCurator run %s complete: %d results, %d errors",
            run_id, len(final_state.get("curation_results", [])), len(final_state.get("errors", []))
        )
        return final_state
