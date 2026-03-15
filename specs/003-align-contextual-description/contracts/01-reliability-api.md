# Reliability API Contract (003-align-contextual-description)

## Overview

This contract defines the public API for the reliability pipeline: provenance access, ontology validation (ValidityScore), factuality evaluation (FActScore), KG cleaning, and iterative seed re-use. These interfaces are exposed to users of the GraphMERT library.

**Status**: Design complete, ready for implementation  
**Last Updated**: 2026-03-15  
**Feature**: [spec.md](../spec.md)

---

## 1. Provenance access

### 1.1 Triple provenance on extraction output

**Requirement**: Every triple in the `KnowledgeGraph` returned by `extract_knowledge_graph` MUST carry provenance data (document and segment/sentence reference) when provenance tracking is enabled (FR-001).

**Contract**: Relations (triples) have an associated provenance record with at least:
- document identifier
- segment or sentence identifier (or character span)

**Access**: Callers can obtain provenance for a given triple (e.g., by relation id or index) and receive the source reference so they can open or display the source text (User Story 1).

**Example (illustrative)**:
- Input: `kg` (KnowledgeGraph), relation index or id
- Output: ProvenanceRecord or equivalent (document_id, segment_id, optional span/context)

---

### 1.2 Configuration

**Requirement**: Extraction options MUST allow enabling provenance tracking (e.g., `enable_provenance_tracking` or equivalent in ProcessingOptions). When enabled, extraction populates provenance for each triple.

**Contract**: ProcessingOptions (or equivalent) includes a flag; default may be true for this feature’s scope so that extraction always produces provenance when the pipeline is used.

---

## 2. Ontology validation and ValidityScore

### 2.1 Validate KG against ontology

**Requirement**: System MUST support validation of extracted triples against a seed ontology and MUST produce a validity indicator or score (FR-002).

**Signature (illustrative)**:
- Input: `kg::KnowledgeGraph`, ontology or domain (e.g., DomainProvider or ontology config)
- Output: ValidityReport (score, total_triples, valid_count, optional per-triple details)

**Contract**:
- When ontology is present: run schema checks (relation types, entity types, domain/range); return ValidityReport with score in [0, 1].
- When ontology is missing or incomplete: degrade gracefully (skip or relax validation) and indicate in report or options (FR-008).

**Example (illustrative)**:
- `validate_kg(kg, domain)` or `evaluate_validity(kg; ontology=...)` returning a struct with `.score`, `.valid_count`, `.total_triples`.

---

### 2.2 ValidityScore naming

**Contract**: The validity metric is exposed under the name **ValidityScore** (or equivalent) in documentation and API (e.g., `validity_score` field or function name) to align with Contextual_information.md and the paper.

---

## 3. Factuality evaluation (FActScore)

### 3.1 Compute factuality score

**Requirement**: System MUST support a factuality evaluation that produces a score when reference or ground-truth data is available (FR-003).

**Signature (illustrative)**:
- Input: `kg::KnowledgeGraph`, reference data (e.g., gold triples or reference KG)
- Output: FactualityScore (score, total_triples, correct_count) or equivalent

**Contract**:
- When reference data is provided: compare triples to reference and return proportion correct (FActScore).
- When reference data is not provided: function may be inapplicable; no score produced (documented behavior).

**Example (illustrative)**:
- `evaluate_factscore(kg, reference_triples)` or `evaluate_factscore(kg; reference_path=...)` returning a struct with `.score`, `.correct_count`, `.total_triples`.

---

### 3.2 FActScore naming

**Contract**: The factuality metric is exposed under the name **FActScore** (or equivalent) in documentation and API to align with the paper and Contextual_information.md.

---

## 4. KG cleaning

### 4.1 Clean KG

**Requirement**: System MUST support a KG cleaning step that can remove or rectify triples that are unsupported, low-confidence, or contradicted, according to configurable policy (FR-004).

**Signature (illustrative)**:
- Input: `kg::KnowledgeGraph`, cleaning policy (thresholds, rules)
- Output: `KnowledgeGraph` (cleaned); optional report (e.g., counts removed)

**Contract**:
- Policy specifies at least: minimum confidence threshold, whether to require provenance, how to handle contradictions (drop/flag).
- Output is a new KnowledgeGraph; original is not mutated. Cleaned KG is suitable for downstream use or as augmented seed (FR-006).

**Example (illustrative)**:
- `clean_kg(kg; min_confidence=0.5, require_provenance=true)` returning KnowledgeGraph.

---

## 5. Encoder in extraction path

### 5.1 Model loading

**Requirement**: When a saved model that includes the encoder is loaded via `load_model(path)`, the returned model MUST include the RoBERTa encoder (and H-GAT) so that extraction uses it (FR-005).

**Contract**:
- Persistence layer (e.g., `models/persistence.jl`) loads full model (RoBERTa + H-GAT) when loading from a full checkpoint.
- Extraction pipeline in `api/extraction.jl` uses the model’s encoder when present (no stub or bypass at inference). Behavior is verifiable by design or test (SC-005).

---

## 6. Iterative seed re-use

### 6.1 Use cleaned/curated KG as seed

**Requirement**: System MUST support using a cleaned or curated KG as augmented seed for a subsequent extraction or training run (FR-006).

**Contract**:
- At least one documented path exists: e.g., export cleaned KG to file → configure as seed path → run training or extraction with that seed.
- Implementation may extend existing seed injection (e.g., `training/seed_injection.jl`) or domain seed loaders to accept a KG (in memory or from file). Exact function signatures are implementation detail; the contract is that such a path exists and is documented (SC-006).

---

## 7. Error and edge behavior

- **Empty corpus / no triples**: Extraction returns empty KG; no phantom provenance (edge case in spec).
- **Missing ontology**: Validation skips or returns a report indicating ontology absent; no hard failure (FR-008).
- **Missing reference for factuality**: Factuality evaluation not run; no FActScore; documented (assumptions in spec).
- **Very large corpora/KGs**: Batch processing or configurable limits; acceptable behavior documented so users can reason about scale (edge case in spec).

---

## 8. Documentation contract

**Requirement**: User- and agent-facing documentation MUST describe the reliability narrative (provenance, FActScore, ValidityScore, KG cleaning, ontology, human-in-the-loop) and map Contextual_information.md to project capabilities and gaps (FR-007, SC-007).

**Contract**: A single place or linked set of docs (e.g., REFERENCE_SOURCES_AND_ENCODER, PROJECT_STATUS, README or docs) covers:
- How reliability and provenance are achieved
- How to use validation, factuality, and cleaning
- Mapping from Contextual_information.md to implemented capabilities and remaining gaps
