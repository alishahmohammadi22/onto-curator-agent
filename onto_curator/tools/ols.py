"""
OLS (Ontology Lookup Service) REST API client.

EBI's OLS4 API — publicly available, no API key required.
https://www.ebi.ac.uk/ols4/api
"""

from __future__ import annotations

import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from onto_curator.models.schemas import OntologyMatch

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.ebi.ac.uk/ols4/api"


class OLSClient:
    """
    Wrapper around the EBI Ontology Lookup Service (OLS4) search API.

    No authentication required. Public API with rate limits.

    Parameters
    ----------
    timeout : int
        HTTP request timeout in seconds (default: 10).
    """

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def search(self, query: str, max_results: int = 5) -> list[OntologyMatch]:
        """
        Search OLS for terms matching ``query``.

        Parameters
        ----------
        query : str
            Search string.
        max_results : int
            Maximum number of results.

        Returns
        -------
        list[OntologyMatch]
        """
        params = {
            "q": query,
            "rows": max_results,
            "fieldList": "id,label,short_form,obo_id,ontology_name,description,synonym",
            "type": "class",
        }

        resp = requests.get(
            f"{_BASE_URL}/search",
            params=params,
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        docs = data.get("response", {}).get("docs", [])
        matches: list[OntologyMatch] = []

        for doc in docs[:max_results]:
            try:
                obo_id = doc.get("obo_id") or doc.get("short_form") or doc.get("id", "")
                label = doc.get("label", query)
                ontology_name = doc.get("ontology_name", "OLS").upper()
                descriptions: list[str] = doc.get("description") or []
                synonyms: list[str] = doc.get("synonym") or []

                # Calculate a basic match score: exact label match = 1.0,
                # substring = 0.7, else 0.5
                score = self._score(query, label, synonyms)

                matches.append(
                    OntologyMatch(
                        ontology_id=obo_id,
                        ontology_name=ontology_name,
                        matched_label=label,
                        synonyms=synonyms[:5],
                        definition=descriptions[0] if descriptions else None,
                        iri=doc.get("iri"),
                        match_score=score,
                        match_source="OLS",
                    )
                )
            except Exception as exc:
                logger.debug("Skipping OLS result %s: %s", doc.get("id"), exc)

        logger.debug("OLS: %d results for '%s'", len(matches), query)
        return matches

    @staticmethod
    def _score(query: str, label: str, synonyms: list[str]) -> float:
        """Simple heuristic match score."""
        q = query.lower()
        if q == label.lower():
            return 1.0
        if q in label.lower() or label.lower() in q:
            return 0.75
        for syn in synonyms:
            if q == syn.lower():
                return 0.85
            if q in syn.lower():
                return 0.65
        return 0.50
