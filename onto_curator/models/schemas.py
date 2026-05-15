"""
Pydantic schemas for all OntoCurator data models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────


class EntityType(str, Enum):
    ASSAY = "assay"
    MOLECULE = "molecule"
    CELL_TYPE = "cell_type"
    ORGANISM = "organism"
    PROCESS = "process"
    INSTRUMENT = "instrument"
    MEASUREMENT = "measurement"
    GENE = "gene"
    PROTEIN = "protein"
    DISEASE = "disease"
    OTHER = "other"


class GovernanceAction(str, Enum):
    AUTO_APPROVE = "auto_approve"
    HUMAN_REVIEW = "human_review"
    ESCALATE = "escalate"


class ConflictType(str, Enum):
    DUPLICATE = "duplicate"
    NEAR_SYNONYM = "near_synonym"
    HIERARCHY_VIOLATION = "hierarchy_violation"
    SCOPE_MISMATCH = "scope_mismatch"
    NONE = "none"


# ─── Core Term Models ─────────────────────────────────────────────────────────


class CandidateTerm(BaseModel):
    """A biomedical entity extracted from source text that may need ontology registration."""

    text: str = Field(..., description="The surface form of the term as found in text")
    normalized_text: str = Field(..., description="Lowercase, whitespace-normalised form")
    context: str = Field(..., description="Surrounding sentence for disambiguation")
    entity_type: EntityType = Field(..., description="Semantic category of the term")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence [0,1]")
    source_document: Optional[str] = Field(None, description="Source document identifier")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"use_enum_values": True}


class OntologyMatch(BaseModel):
    """A single candidate match from an ontology lookup."""

    ontology_id: str = Field(..., description="Curie-style ID e.g. OBI:0001924")
    ontology_name: str = Field(..., description="Short ontology name e.g. OBI")
    matched_label: str = Field(..., description="Preferred label in the ontology")
    synonyms: list[str] = Field(default_factory=list)
    definition: Optional[str] = None
    iri: Optional[str] = None
    match_score: float = Field(..., ge=0.0, le=1.0)
    match_source: Literal["BioPortal", "OLS", "local_cache"] = "BioPortal"


class TermMapping(BaseModel):
    """Full mapping result for one candidate term, with ranked ontology candidates."""

    candidate: CandidateTerm
    top_match: Optional[OntologyMatch] = None
    alternative_matches: list[OntologyMatch] = Field(default_factory=list)
    mapping_confidence: float = Field(0.0, ge=0.0, le=1.0)
    needs_new_term: bool = Field(False, description="True if no suitable match found")
    mapper_reasoning: Optional[str] = Field(None, description="LLM explanation of mapping choice")
    mapped_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Conflict & Governance ────────────────────────────────────────────────────


class ConflictItem(BaseModel):
    """One detected conflict between term mappings."""

    conflict_type: ConflictType
    term_a: str
    term_b: str
    description: str
    suggested_resolution: Optional[str] = None


class ConflictReport(BaseModel):
    """Conflict analysis result for a batch of term mappings."""

    has_conflicts: bool
    conflict_count: int = 0
    conflicts: list[ConflictItem] = Field(default_factory=list)
    resolver_reasoning: Optional[str] = None
    resolved_at: datetime = Field(default_factory=datetime.utcnow)


class GovernanceDecision(BaseModel):
    """Routing decision for a single term mapping."""

    term_text: str
    action: GovernanceAction
    reason: str
    confidence_used: float
    assigned_steward: Optional[str] = Field(
        None, description="Domain steward if routed for human review"
    )
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    decided_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"use_enum_values": True}


# ─── Final Result ─────────────────────────────────────────────────────────────


class CurationResult(BaseModel):
    """Complete curation record for one term — mapping + governance + audit."""

    mapping: TermMapping
    governance: GovernanceDecision
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    final_ontology_id: Optional[str] = None
    status: Literal["approved", "pending_review", "escalated", "rejected"] = "pending_review"


class CurationSummary(BaseModel):
    """Aggregated statistics for a full curation run."""

    total_terms: int
    auto_approved: int
    pending_human_review: int
    escalated: int
    new_terms_proposed: int
    conflict_count: int
    auto_curation_rate: float  # percentage
    run_id: str
    run_at: datetime = Field(default_factory=datetime.utcnow)
    results: list[CurationResult] = Field(default_factory=list)
