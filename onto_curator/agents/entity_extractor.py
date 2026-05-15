"""
EntityExtractor Agent — uses GPT-5.2 to identify biomedical entities
in free-text that are candidates for ontology registration.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from onto_curator.models.schemas import CandidateTerm, EntityType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a biomedical named entity recognition expert specialising in
pharmaceutical R&D and ontology curation. Your task is to extract entities from text that
may need to be registered in or mapped to a controlled biomedical ontology.

Focus on:
- Assays and measurement methods
- Cell types and biological materials
- Molecules, compounds, and reagents
- Biological processes and mechanisms
- Instruments and equipment
- Genes and proteins
- Diseases and phenotypes

Return a JSON array of objects. Each object must have:
  "text":           exact surface form from the input text
  "normalized_text": lowercase, whitespace-normalised version
  "context":        the full sentence where the term appears
  "entity_type":    one of [assay, molecule, cell_type, organism, process,
                             instrument, measurement, gene, protein, disease, other]
  "confidence":     float 0.0–1.0 (your confidence this needs ontology registration)

Only include terms that are meaningful domain concepts — skip generic words like
"study", "data", "result", "value".
"""


class EntityExtractorAgent:
    """
    Calls GPT-5.2 to extract ontology-candidate terms from source text.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI client.
    model : str
        Model to use (default: "gpt-5.2").
    min_confidence : float
        Discard extracted terms below this threshold (default: 0.5).
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5.2",
        min_confidence: float = 0.5,
    ) -> None:
        self.client = client
        self.model = model
        self.min_confidence = min_confidence

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract(
        self,
        text: str,
        source_document: Optional[str] = None,
    ) -> list[CandidateTerm]:
        """
        Extract candidate ontology terms from ``text``.

        Parameters
        ----------
        text : str
            Raw scientific text (lab report, assay description, protocol, etc.)
        source_document : str, optional
            Document identifier for provenance.

        Returns
        -------
        list[CandidateTerm]
            Extracted candidate terms sorted by confidence (descending).
        """
        logger.info("EntityExtractor: sending %d chars to %s", len(text), self.model)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Extract all ontology-candidate biomedical entities from the "
                        f"following text:\n\n---\n{text}\n---\n\n"
                        "Return only valid JSON — an array of objects with the fields "
                        "described in your instructions."
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw = response.choices[0].message.content
        logger.debug("EntityExtractor raw response: %s", raw[:500])

        parsed = json.loads(raw)
        # Handle both {"entities": [...]} and plain [...]
        items: list[dict] = parsed if isinstance(parsed, list) else parsed.get("entities", [])

        terms: list[CandidateTerm] = []
        for item in items:
            try:
                # Normalise entity_type value to a valid enum member
                raw_type = item.get("entity_type", "other").lower().replace(" ", "_")
                valid_types = {e.value for e in EntityType}
                entity_type = raw_type if raw_type in valid_types else "other"

                term = CandidateTerm(
                    text=item["text"],
                    normalized_text=item.get("normalized_text", item["text"].lower().strip()),
                    context=item.get("context", text[:200]),
                    entity_type=EntityType(entity_type),
                    confidence=float(item.get("confidence", 0.75)),
                    source_document=source_document,
                )
                if term.confidence >= self.min_confidence:
                    terms.append(term)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed term %s: %s", item, exc)

        terms.sort(key=lambda t: t.confidence, reverse=True)
        logger.info("EntityExtractor: extracted %d terms (min_confidence=%.2f)", len(terms), self.min_confidence)
        return terms
