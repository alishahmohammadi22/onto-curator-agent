"""
OntoCuratorPipeline — the high-level entry point that wires together
all agents and exposes a clean, one-call API.

Usage
-----
from onto_curator import OntoCuratorPipeline

pipeline = OntoCuratorPipeline.from_env()   # reads OPENAI_API_KEY etc. from env
summary  = pipeline.curate(text)
print(summary.auto_curation_rate)
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from onto_curator.agents.conflict_resolver import ConflictResolverAgent
from onto_curator.agents.entity_extractor import EntityExtractorAgent
from onto_curator.agents.governance_gate import GovernanceGateAgent
from onto_curator.agents.term_mapper import TermMapperAgent
from onto_curator.models.schemas import (
    ConflictReport,
    CurationResult,
    CurationSummary,
    GovernanceAction,
    TermMapping,
)
from onto_curator.orchestrator import OntoCuratorOrchestrator
from onto_curator.tools.bioportal import BioPortalClient
from onto_curator.tools.ols import OLSClient


class OntoCuratorPipeline:
    """
    High-level facade for the OntoCurator agent pipeline.

    Parameters
    ----------
    openai_client : OpenAI
        Authenticated OpenAI client.
    model : str
        OpenAI model name (default: "gpt-5.2").
    bioportal_api_key : str, optional
        BioPortal API key. Falls back to ``BIOPORTAL_API_KEY`` env var.
    min_extraction_confidence : float
        Minimum extraction confidence to keep a candidate term (default: 0.5).
    auto_approve_threshold : float
        Mapping confidence above which terms are auto-approved (default: 0.88).
    human_review_threshold : float
        Minimum mapping confidence for human review routing (default: 0.60).
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str = "gpt-5.2",
        bioportal_api_key: Optional[str] = None,
        min_extraction_confidence: float = 0.5,
        auto_approve_threshold: float = 0.88,
        human_review_threshold: float = 0.60,
    ) -> None:
        self.model = model

        bioportal = BioPortalClient(api_key=bioportal_api_key)
        ols = OLSClient()

        extractor = EntityExtractorAgent(
            client=openai_client,
            model=model,
            min_confidence=min_extraction_confidence,
        )
        mapper = TermMapperAgent(
            client=openai_client,
            bioportal=bioportal,
            ols=ols,
            model=model,
        )
        resolver = ConflictResolverAgent(client=openai_client, model=model)
        gate = GovernanceGateAgent(
            high_threshold=auto_approve_threshold,
            low_threshold=human_review_threshold,
        )

        self._orchestrator = OntoCuratorOrchestrator(
            extractor=extractor,
            mapper=mapper,
            resolver=resolver,
            gate=gate,
        )

    @classmethod
    def from_env(cls, model: str = "gpt-5.2", **kwargs) -> "OntoCuratorPipeline":
        """
        Instantiate pipeline from environment variables.

        Required env vars
        -----------------
        OPENAI_API_KEY     : OpenAI API key
        BIOPORTAL_API_KEY  : (optional) BioPortal API key
        """
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Set it in your environment or in a .env file."
            )

        client = OpenAI(api_key=api_key)
        bp_key = os.environ.get("BIOPORTAL_API_KEY")
        return cls(openai_client=client, model=model, bioportal_api_key=bp_key, **kwargs)

    def curate(
        self,
        text: str,
        source_document: Optional[str] = None,
    ) -> CurationSummary:
        """
        Run the full curation pipeline on ``text``.

        Parameters
        ----------
        text : str
            Free-text scientific content (lab report, protocol, assay description, etc.)
        source_document : str, optional
            Document identifier for provenance tracking.

        Returns
        -------
        CurationSummary
            Aggregated statistics + full list of CurationResult objects.
        """
        import uuid

        state = self._orchestrator.run(text, source_document=source_document)

        raw_results: list[dict] = state.get("curation_results", [])
        results = [CurationResult(**r) for r in raw_results]

        raw_conflict: dict = state.get("conflict_report") or {}
        conflict_report = ConflictReport(**raw_conflict) if raw_conflict else ConflictReport(has_conflicts=False)

        auto = sum(1 for r in results if r.status == "approved")
        review = sum(1 for r in results if r.status == "pending_review")
        escalated = sum(1 for r in results if r.status == "escalated")
        new_terms = sum(1 for r in results if r.mapping.needs_new_term)
        total = len(results)
        rate = round((auto / total * 100) if total > 0 else 0.0, 1)

        return CurationSummary(
            total_terms=total,
            auto_approved=auto,
            pending_human_review=review,
            escalated=escalated,
            new_terms_proposed=new_terms,
            conflict_count=conflict_report.conflict_count,
            auto_curation_rate=rate,
            run_id=state.get("run_id", str(uuid.uuid4())[:8]),
            results=results,
        )
