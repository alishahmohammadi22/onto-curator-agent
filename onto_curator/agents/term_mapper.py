"""
TermMapper Agent — searches BioPortal and OLS for ontology matches
and uses GPT-5.2 to select the best mapping from candidates.
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from onto_curator.models.schemas import CandidateTerm, OntologyMatch, TermMapping
from onto_curator.tools.bioportal import BioPortalClient
from onto_curator.tools.ols import OLSClient

logger = logging.getLogger(__name__)

_MAPPING_SYSTEM_PROMPT = """You are an expert biomedical ontologist. Given a candidate term
and a list of ontology search results, select the BEST matching ontology term.

Consider:
1. Semantic equivalence — the ontology term must mean the same thing, not just share words
2. Ontology authority — prefer OBI, ChEBI, GO, CL, DOID over less authoritative ontologies
3. Specificity — prefer the most specific accurate match over a vague parent term
4. Context — use the surrounding sentence to disambiguate

Return a JSON object with:
  "selected_id":   the ontology ID of the best match (or null if none are suitable)
  "confidence":    float 0.0–1.0 (your confidence in this mapping)
  "needs_new_term": true if no existing term covers this concept
  "reasoning":     1-2 sentence explanation of your choice
"""


class TermMapperAgent:
    """
    Maps candidate terms to ontology entries using BioPortal/OLS search
    followed by GPT-5.2 ranking.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI client.
    bioportal : BioPortalClient
        BioPortal REST API wrapper.
    ols : OLSClient
        OLS REST API wrapper.
    model : str
        LLM model for ranking (default: "gpt-5.2").
    top_k : int
        Maximum candidates to pass to the LLM for ranking (default: 5).
    """

    def __init__(
        self,
        client: OpenAI,
        bioportal: BioPortalClient,
        ols: OLSClient,
        model: str = "gpt-5.2",
        top_k: int = 5,
    ) -> None:
        self.client = client
        self.bioportal = bioportal
        self.ols = ols
        self.model = model
        self.top_k = top_k

    def map_terms(self, terms: list[CandidateTerm]) -> list[TermMapping]:
        """Map a list of extracted candidate terms to ontology entries."""
        mappings: list[TermMapping] = []
        for term in terms:
            mapping = self._map_single(term)
            mappings.append(mapping)
        return mappings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _map_single(self, term: CandidateTerm) -> TermMapping:
        """Run BioPortal + OLS lookup then GPT-5.2 ranking for one term."""
        logger.info("TermMapper: mapping '%s'", term.text)

        candidates: list[OntologyMatch] = []

        # 1. BioPortal search
        try:
            bp_results = self.bioportal.search(term.normalized_text, max_results=self.top_k)
            candidates.extend(bp_results)
        except Exception as exc:
            logger.warning("BioPortal search failed for '%s': %s", term.text, exc)

        # 2. OLS search (supplementary)
        try:
            ols_results = self.ols.search(term.normalized_text, max_results=self.top_k)
            # Deduplicate by ontology_id
            existing_ids = {c.ontology_id for c in candidates}
            for r in ols_results:
                if r.ontology_id not in existing_ids:
                    candidates.append(r)
                    existing_ids.add(r.ontology_id)
        except Exception as exc:
            logger.warning("OLS search failed for '%s': %s", term.text, exc)

        # If no candidates found, return a mapping flagged for new term proposal
        if not candidates:
            logger.info("TermMapper: no candidates found for '%s' — flagging as new term", term.text)
            return TermMapping(
                candidate=term,
                top_match=None,
                alternative_matches=[],
                mapping_confidence=0.0,
                needs_new_term=True,
                mapper_reasoning="No matches found in BioPortal or OLS.",
            )

        # 3. GPT-5.2 selects the best match
        best_id, llm_confidence, needs_new, reasoning = self._llm_rank(term, candidates)

        top_match: Optional[OntologyMatch] = None
        alternatives: list[OntologyMatch] = []
        for c in candidates:
            if c.ontology_id == best_id:
                top_match = c
            else:
                alternatives.append(c)

        # Blend API match score with LLM confidence
        blended_confidence = llm_confidence
        if top_match:
            blended_confidence = 0.6 * llm_confidence + 0.4 * top_match.match_score

        return TermMapping(
            candidate=term,
            top_match=top_match,
            alternative_matches=alternatives[: self.top_k - 1],
            mapping_confidence=round(blended_confidence, 3),
            needs_new_term=needs_new,
            mapper_reasoning=reasoning,
        )

    def _llm_rank(
        self, term: CandidateTerm, candidates: list[OntologyMatch]
    ) -> tuple[Optional[str], float, bool, str]:
        """Ask GPT-5.2 to choose the best ontology match from candidates."""
        candidate_text = "\n".join(
            f"- ID: {c.ontology_id} | Label: {c.matched_label} | "
            f"Ontology: {c.ontology_name} | Score: {c.match_score:.2f} | "
            f"Definition: {(c.definition or 'n/a')[:120]}"
            for c in candidates
        )

        user_msg = (
            f"Term to map: \"{term.text}\"\n"
            f"Entity type: {term.entity_type}\n"
            f"Context sentence: \"{term.context}\"\n\n"
            f"Candidate ontology matches:\n{candidate_text}\n\n"
            "Return a JSON object with keys: selected_id, confidence, needs_new_term, reasoning"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _MAPPING_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        import json
        data = json.loads(resp.choices[0].message.content)
        return (
            data.get("selected_id"),
            float(data.get("confidence", 0.5)),
            bool(data.get("needs_new_term", False)),
            data.get("reasoning", ""),
        )
