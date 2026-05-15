"""
BioPortal REST API client.

Searches the NCBO BioPortal ontology repository for term matches.
Requires a BioPortal API key (free registration at bioportal.bioontology.org).

If BIOPORTAL_API_KEY is not set the client operates in mock mode and returns
an empty list, allowing the pipeline to gracefully fall back to OLS.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from onto_curator.models.schemas import OntologyMatch

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.bioontology.org"

# Ontologies to search (high-value for pharma/biomedical)
_DEFAULT_ONTOLOGIES = ["OBI", "CHEBI", "GO", "CL", "DOID", "EFO", "BAO", "HP", "NCIT", "MESH"]


class BioPortalClient:
    """
    Thin wrapper around the BioPortal REST API v1.

    Parameters
    ----------
    api_key : str, optional
        BioPortal API key. Falls back to ``BIOPORTAL_API_KEY`` env var.
        If neither is set, operates in mock mode (always returns empty list).
    ontologies : list[str], optional
        Subset of ontology abbreviations to search. Defaults to ``_DEFAULT_ONTOLOGIES``.
    timeout : int
        HTTP request timeout in seconds (default: 10).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ontologies: Optional[list[str]] = None,
        timeout: int = 10,
    ) -> None:
        self.api_key = api_key or os.getenv("BIOPORTAL_API_KEY", "")
        self.ontologies = ontologies or _DEFAULT_ONTOLOGIES
        self.timeout = timeout
        self._mock = not bool(self.api_key)
        if self._mock:
            logger.warning(
                "BioPortalClient: no API key found — running in mock mode (returns empty results). "
                "Set BIOPORTAL_API_KEY environment variable for real lookups."
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def search(self, query: str, max_results: int = 5) -> list[OntologyMatch]:
        """
        Search BioPortal for terms matching ``query``.

        Parameters
        ----------
        query : str
            Search string (normalised term text).
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        list[OntologyMatch]
            Ranked by relevance score (descending).
        """
        if self._mock:
            return []

        params = {
            "q": query,
            "ontologies": ",".join(self.ontologies),
            "pagesize": max_results,
            "include": "prefLabel,synonym,definition,notation",
            "apikey": self.api_key,
        }

        resp = requests.get(
            f"{_BASE_URL}/search",
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        matches: list[OntologyMatch] = []
        for item in data.get("collection", [])[:max_results]:
            try:
                # Extract ontology acronym from the links
                ontology_link: str = (
                    item.get("links", {}).get("ontology", "")
                    or item.get("@id", "")
                )
                ontology_name = self._parse_ontology_name(ontology_link)

                # BioPortal relevance score is not normalised; we cap at 1.0
                score = min(float(item.get("score", 1.0)) / 10.0, 1.0)

                matches.append(
                    OntologyMatch(
                        ontology_id=item.get("@id", "").split("/")[-1],
                        ontology_name=ontology_name,
                        matched_label=item.get("prefLabel", query),
                        synonyms=item.get("synonym", []),
                        definition=(item.get("definition") or [None])[0],
                        iri=item.get("@id"),
                        match_score=score,
                        match_source="BioPortal",
                    )
                )
            except Exception as exc:
                logger.debug("Skipping BioPortal result %s: %s", item.get("@id"), exc)

        logger.debug("BioPortal: %d results for '%s'", len(matches), query)
        return matches

    @staticmethod
    def _parse_ontology_name(link: str) -> str:
        """Extract ontology acronym from a BioPortal link URL."""
        # e.g. https://data.bioontology.org/ontologies/OBI
        parts = link.rstrip("/").split("/")
        if "ontologies" in parts:
            idx = parts.index("ontologies")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "UNKNOWN"
