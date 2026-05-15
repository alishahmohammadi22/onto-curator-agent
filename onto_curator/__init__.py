"""
onto_curator — Agentic AI for Autonomous Ontology Term Curation
"""

from onto_curator.pipeline import OntoCuratorPipeline
from onto_curator.models.schemas import (
    CandidateTerm,
    TermMapping,
    ConflictReport,
    CurationResult,
    GovernanceDecision,
    CurationSummary,
)

__version__ = "0.1.0"
__author__ = "Ali Shahmohammadi"

__all__ = [
    "OntoCuratorPipeline",
    "CandidateTerm",
    "TermMapping",
    "ConflictReport",
    "CurationResult",
    "GovernanceDecision",
    "CurationSummary",
]
