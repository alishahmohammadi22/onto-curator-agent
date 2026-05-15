"""
ConflictResolver Agent — uses GPT-5.2 to detect and resolve
conflicts among a batch of term mappings (duplicates, near-synonyms,
hierarchy violations, scope mismatches).
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from onto_curator.models.schemas import ConflictItem, ConflictReport, ConflictType, TermMapping

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert biomedical ontology curator specialising in conflict
detection. Given a list of proposed ontology term mappings, identify any conflicts:

1. DUPLICATE — two different surface forms map to exactly the same ontology ID
2. NEAR_SYNONYM — two terms are semantically near-identical but mapped to different ontology IDs
3. HIERARCHY_VIOLATION — a child term is mapped to a concept that is not a subtype of its parent
4. SCOPE_MISMATCH — the entity type (assay, molecule, etc.) doesn't match the ontology class

For each conflict, suggest a resolution (e.g., "merge into OBI:0001924", "keep both as
distinct terms", "change mapping for term X to Y").

Return a JSON object:
{
  "has_conflicts": bool,
  "conflicts": [
    {
      "conflict_type": "duplicate|near_synonym|hierarchy_violation|scope_mismatch",
      "term_a": "<surface form>",
      "term_b": "<surface form or ontology ID>",
      "description": "<plain English description>",
      "suggested_resolution": "<what to do>"
    }
  ],
  "reasoning": "<overall summary of findings>"
}
"""


class ConflictResolverAgent:
    """
    Analyses a batch of TermMappings for inter-term conflicts.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI client.
    model : str
        LLM model to use (default: "gpt-5.2").
    """

    def __init__(self, client: OpenAI, model: str = "gpt-5.2") -> None:
        self.client = client
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def resolve(self, mappings: list[TermMapping]) -> ConflictReport:
        """
        Detect and propose resolutions for conflicts in ``mappings``.

        Parameters
        ----------
        mappings : list[TermMapping]
            All term mappings produced by TermMapperAgent for one batch.

        Returns
        -------
        ConflictReport
            Structured conflict analysis with per-conflict items.
        """
        if not mappings:
            return ConflictReport(has_conflicts=False)

        logger.info("ConflictResolver: analysing %d term mappings", len(mappings))

        # Build compact summary for LLM (avoid token waste)
        mapping_lines = []
        for m in mappings:
            oid = m.top_match.ontology_id if m.top_match else "UNMAPPED"
            label = m.top_match.matched_label if m.top_match else "—"
            mapping_lines.append(
                f"- Term: \"{m.candidate.text}\" | Type: {m.candidate.entity_type} "
                f"| MappedTo: {oid} ({label}) | Confidence: {m.mapping_confidence:.2f}"
            )

        user_msg = (
            "Analyse the following term mappings for conflicts:\n\n"
            + "\n".join(mapping_lines)
            + "\n\nReturn the JSON conflict report as instructed."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        data = json.loads(resp.choices[0].message.content)

        conflict_items: list[ConflictItem] = []
        for raw in data.get("conflicts", []):
            try:
                raw_type = raw.get("conflict_type", "none").lower()
                ctype = ConflictType(raw_type) if raw_type in {e.value for e in ConflictType} else ConflictType.NONE
                conflict_items.append(
                    ConflictItem(
                        conflict_type=ctype,
                        term_a=raw.get("term_a", ""),
                        term_b=raw.get("term_b", ""),
                        description=raw.get("description", ""),
                        suggested_resolution=raw.get("suggested_resolution"),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed conflict item %s: %s", raw, exc)

        report = ConflictReport(
            has_conflicts=bool(data.get("has_conflicts", len(conflict_items) > 0)),
            conflict_count=len(conflict_items),
            conflicts=conflict_items,
            resolver_reasoning=data.get("reasoning"),
        )

        logger.info(
            "ConflictResolver: found %d conflict(s)", report.conflict_count
        )
        return report
