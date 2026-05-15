# OntoCurator Agent

**Agentic AI for Autonomous Ontology Term Curation in Pharmaceutical R&D**

[![CI](https://github.com/alishahmohammadi22/onto-curator-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/alishahmohammadi22/onto-curator-agent/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

OntoCurator is a **multi-agent pipeline** that automates first-pass ontology curation:

```
Free-text input
     │
     ▼
EntityExtractor  →  TermMapper  →  ConflictResolver  →  GovernanceGate
(GPT-5.2 NER)      (BioPortal        (GPT-5.2               (rule-based
                    + OLS + GPT)      conflict detection)      routing)
     │                  │                  │                     │
     ▼                  ▼                  ▼                     ▼
Candidate          Ontology           Conflict             auto_approve /
  terms            mappings           report               human_review /
                  (OBI, ChEBI,                              escalate
                   GO, CL, ...)
```

**Key capabilities**
- Extracts biomedical entities from any free-text (lab reports, protocols, assay descriptions)
- Maps terms to OBO Foundry ontologies via BioPortal and EBI OLS
- Detects duplicates, near-synonyms, and hierarchy violations
- Routes each term to auto-approval, human review, or escalation based on confidence
- Full W3C PROV-O-style audit trail per term
- Confidence-calibrated — never silently auto-approves low-quality mappings

---

## Quick install

```bash
# Core package
pip install onto-curator

# With Jupyter notebook extras
pip install "onto-curator[notebooks]"

# Editable install from source
git clone https://github.com/alishahmohammadi22/onto-curator-agent.git
cd onto-curator-agent
pip install -e ".[notebooks]"
```

---

## Usage

```python
from onto_curator import OntoCuratorPipeline

# Reads OPENAI_API_KEY (and optionally BIOPORTAL_API_KEY) from environment / .env
pipeline = OntoCuratorPipeline.from_env(model="gpt-5.2")

text = """
    Viable cell density was 3.2 × 10^6 cells/mL by trypan blue exclusion.
    CD3+ T cells were transduced with an anti-CD19 CAR lentiviral vector at MOI 5.
    Endotoxin was < 0.5 EU/mL by LAL assay.
"""

summary = pipeline.curate(text, source_document="DOC-001")

print(f"Terms extracted  : {summary.total_terms}")
print(f"Auto-approved    : {summary.auto_approved}")
print(f"Pending review   : {summary.pending_human_review}")
print(f"Escalated        : {summary.escalated}")
print(f"Auto-curation %  : {summary.auto_curation_rate}%")

# Inspect individual term results
for result in summary.results:
    m = result.mapping
    print(f"  {m.candidate.text:30s} → {m.top_match.ontology_id if m.top_match else 'NEW'}"
          f"  [{result.status}]")
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key (for GPT-5.2) |
| `BIOPORTAL_API_KEY` | Optional | BioPortal API key — free at [bioportal.bioontology.org](https://bioportal.bioontology.org). Without it the pipeline uses EBI OLS only. |

Create a `.env` file at the project root (git-ignored):
```
OPENAI_API_KEY=sk-...
BIOPORTAL_API_KEY=...
```

---

## Architecture

### Agents

| Agent | Role | Model |
|---|---|---|
| **EntityExtractorAgent** | NER on free text — returns `CandidateTerm` list | GPT-5.2 |
| **TermMapperAgent** | BioPortal + OLS search → GPT-5.2 ranking | GPT-5.2 |
| **ConflictResolverAgent** | Detects duplicates, near-synonyms, hierarchy violations | GPT-5.2 |
| **GovernanceGateAgent** | Routes terms: auto-approve / human-review / escalate | Rule-based |

### Governance thresholds (configurable)

| Confidence | Action |
|---|---|
| ≥ 0.88 | `auto_approve` — term written directly to ontology |
| 0.60 – 0.88 | `human_review` — routed to domain steward |
| < 0.60 | `escalate` — flagged as critical, blocked |
| `needs_new_term=True` | `escalate` — new term proposal workflow |

### Ontologies searched

OBI · ChEBI · GO · CL · DOID · EFO · BAO · HP · NCIT · MeSH

---

## Running the notebook

```bash
pip install -e ".[notebooks]"
# Set OPENAI_API_KEY in .env
jupyter lab notebooks/01_quickstart_demo.ipynb
```

The notebook includes three realistic cell therapy manufacturing lab reports and walks through the full pipeline with visualisations of governance routing and confidence distributions.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Project structure

```
onto_curator/
├── pipeline.py          # OntoCuratorPipeline — top-level API
├── orchestrator.py      # LangGraph state machine
├── agents/
│   ├── entity_extractor.py
│   ├── term_mapper.py
│   ├── conflict_resolver.py
│   └── governance_gate.py
├── tools/
│   ├── bioportal.py     # BioPortal REST API client
│   └── ols.py           # EBI OLS4 API client
└── models/
    └── schemas.py       # Pydantic data models
notebooks/
└── 01_quickstart_demo.ipynb
tests/
├── test_entity_extractor.py
└── test_governance_gate.py
```

---

## Roadmap

- [ ] Neo4j knowledge graph export
- [ ] Active learning on curator corrections
- [ ] SciSpacy / BioBERT local NER (offline mode, no API call)
- [ ] Batch processing CLI (`onto-curator curate --file report.txt`)
- [ ] FAIR provenance export (W3C PROV-O RDF)
- [ ] Streamlit governance dashboard

---

## Author

**Ali Shahmohammadi, Ph.D.**  
Associate Director, FAIR Data Strategy & Digital Connectivity — Takeda Pharmaceutical  
[github.com/alishahmohammadi22](https://github.com/alishahmohammadi22)

## License

MIT — see [LICENSE](LICENSE)
