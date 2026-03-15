# Feature Specification: Align Project with Contextual Description and Reliability Narrative

**Feature Branch**: `003-align-contextual-description`  
**Created**: 2026-03-15  
**Status**: Draft  
**Input**: User description: "Substantially improve our project to reflect the interesting description in Contextual_information.md. Ask questions."

## Summary

The project shall be improved so that its capabilities, documentation, and outcomes align with the narrative in Contextual_information.md: a compact encoder that produces **reliable** domain-specific knowledge graphs with **provenance**, **factuality** (FActScore), **validity** (ValidityScore), **ontology-consistent** construction, **KG cleaning**, and support for **human-in-the-loop** auditing and iterative improvement. The improvement scope is **full implementation of the reliability pipeline**: documentation updates plus implementation of provenance storage per triple, ontology validation and ValidityScore, factuality evaluation (FActScore), KG cleaning step, encoder in the extraction path, and iterative seed re-use.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extract KG with traceable provenance (Priority: P1)

As a researcher or domain expert, I want every triple in the extracted knowledge graph to be traceable to a specific source span (e.g., sentence or segment) in the input corpus, so that I can verify and audit why an edge exists.

**Why this priority**: Provenance is central to the "reliable KG" narrative and enables human-in-the-loop auditing; without it, the project cannot claim attributable outputs.

**Independent Test**: Run extraction on a short corpus; inspect the output to confirm each triple is linked to a document and segment/sentence identifier. Delivers value by making the KG auditable.

**Acceptance Scenarios**:

1. **Given** a corpus of domain text and a configured extraction run, **When** extraction completes, **Then** each output triple includes a reference to its source (document and segment/sentence).
2. **Given** a triple in the extracted KG, **When** a user requests its provenance, **Then** the system returns the exact source span (or reference) so the user can open the source text.

---

### User Story 2 - Validate triples against ontology and report validity (Priority: P2)

As a domain expert or operator, I want extracted triples to be checked against the seed ontology (relation types, entity types, domain/range constraints) and want a validity metric (ValidityScore-style) so that I know how well the KG conforms to the schema.

**Why this priority**: Validity is a stated differentiator in the contextual description (e.g., ~68.7% ValidityScore vs. ~43% for an LLM baseline); the system must support measuring and enforcing ontology consistency.

**Independent Test**: Run extraction with a domain that has an ontology; run validation and obtain a validity metric. Compare before/after cleaning if cleaning is implemented.

**Acceptance Scenarios**:

1. **Given** an extracted KG and a seed ontology, **When** validation runs, **Then** triples are checked for schema compatibility (allowed relation usage, entity types, domain/range) and a validity score or report is produced.
2. **Given** triples that violate the ontology, **When** validation runs, **Then** they are either rejected or flagged (and optionally corrected if the configured policy supports it).

---

### User Story 3 - Measure factuality and support KG cleaning (Priority: P2)

As a researcher, I want to measure how many extracted triples are factually correct (FActScore-style) and want a KG cleaning step that removes or rectifies unsupported, low-confidence, or contradicted triples so that the final KG is more reliable.

**Why this priority**: The contextual description emphasizes KG cleaning as a step that significantly boosts FActScore; supporting this aligns the project with the described pipeline.

**Independent Test**: Run extraction; run factuality evaluation (where ground truth or references exist) and optionally run a cleaning step; compare metrics before and after.

**Acceptance Scenarios**:

1. **Given** an extracted KG and (where available) ground truth or reference data, **When** factuality evaluation runs, **Then** a factuality score (e.g., proportion of correct triples) is produced.
2. **Given** an extracted KG, **When** a KG cleaning stage runs, **Then** unsupported, low-confidence, or contradicted triples are removed or rectified according to configured rules, and the resulting KG is available for downstream use or re-fed as augmented seed.

---

### User Story 4 - Use encoder in extraction path and support iterative improvement (Priority: P3)

As a developer or operator, I want the extraction pipeline to use the compact encoder (RoBERTa-class) end-to-end so that inference matches the paper’s architecture, and I want the ability to feed an improved or cleaned KG back as seed for further refinement.

**Why this priority**: Encoder-in-path and iterative improvement are explicit in the contextual pipeline description; they close the gap between "implemented in training" and "used at extraction."

**Independent Test**: Load a model and run extraction; confirm the encoder is used. Optionally run a cycle: extract → clean/curate → feed back as seed → re-run; confirm the pipeline supports this loop.

**Acceptance Scenarios**:

1. **Given** a saved model that includes the encoder, **When** extraction is run, **Then** the encoder is invoked as part of the extraction path (no stub or bypass).
2. **Given** a cleaned or curated KG, **When** the user configures it as augmented seed, **Then** the system can use it for a subsequent extraction or training run to support iterative improvement.

---

### User Story 5 - Documentation and narrative alignment (Priority: P3)

As a stakeholder or new contributor, I want project documentation to clearly describe the reliability narrative (provenance, FActScore, ValidityScore, KG cleaning, ontology, human-in-the-loop) and how the project’s behavior maps to the description in Contextual_information.md.

**Why this priority**: Ensures that "reflect the interesting description" is satisfied at least at the documentation level and guides future implementation.

**Independent Test**: Read the main user-facing and agent-facing docs; confirm that reliability, provenance, factuality, validity, and iterative improvement are described and linked to capabilities or roadmap.

**Acceptance Scenarios**:

1. **Given** a reader of the project docs, **When** they look for how reliability and provenance are achieved, **Then** they find a clear description (and, where implemented, how to use it).
2. **Given** Contextual_information.md, **When** a reader checks the project’s reference/spec docs, **Then** they find an explicit mapping (e.g., in REFERENCE_SOURCES_AND_ENCODER or equivalent) from that narrative to project capabilities and gaps.

---

### Edge Cases

- What happens when the corpus is empty or no triples are extracted? The system should report this clearly and not produce a KG with phantom provenance.
- What happens when the seed ontology is missing or incomplete for a domain? The system should degrade gracefully (e.g., skip or relax validation) and document the limitation.
- How does the system handle very large corpora or KGs for provenance storage and cleaning? Acceptable behavior is defined (e.g., batch processing, configurable limits) so that users can reason about scale.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST associate each extracted triple with provenance data (document and segment/sentence or equivalent source reference) so that users can trace the triple to source text.
- **FR-002**: The system MUST support validation of extracted triples against a seed ontology (relation types, entity types, domain/range constraints) and MUST produce a validity indicator or score when such validation is run.
- **FR-003**: The system MUST support a factuality evaluation that produces a score (e.g., proportion of triples deemed correct) when reference or ground-truth data is available.
- **FR-004**: The system MUST support a KG cleaning step that can remove or rectify triples that are unsupported, low-confidence, or contradicted, according to configurable policy.
- **FR-005**: The system MUST use the compact encoder in the extraction path when a model that includes the encoder is loaded (no stub or bypass that skips the encoder at inference).
- **FR-006**: The system MUST support using a cleaned or curated KG as augmented seed for a subsequent extraction or training run (iterative improvement).
- **FR-007**: The system MUST provide user- and agent-facing documentation that describes the reliability narrative (provenance, factuality, validity, KG cleaning, ontology, human-in-the-loop) and how it maps to Contextual_information.md and to current capabilities or roadmap.
- **FR-008**: The system MUST handle missing or incomplete ontology gracefully (e.g., skip or relax validation with clear indication) so that operation is possible in constrained environments.

### Key Entities

- **Triple**: A (head, relation, tail) fact; must carry provenance (document, segment/sentence reference).
- **Seed ontology / seed KG**: Domain schema and optional known triples; defines relation types, entity types, and domain/range constraints used for validation and training.
- **Provenance record**: Document identifier, segment/sentence identifier (or span), and optional context; attached to each triple.
- **Validity score/report**: Result of checking triples against the ontology; may be a single metric or a per-triple/aggregate report.
- **Factuality score**: Result of comparing triples to reference/ground truth; proportion or count of correct triples.
- **Cleaned KG**: KG produced after applying removal/rectification rules (unsupported, low-confidence, contradicted triples).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can trace every extracted triple to a specific source span (document + segment/sentence) in the corpus.
- **SC-002**: Operators can obtain a validity score or report after running extraction when a seed ontology is configured.
- **SC-003**: Operators can obtain a factuality score when reference or ground-truth data is available for evaluation.
- **SC-004**: A KG cleaning step is available and reduces the set of triples to those that meet configurable support/confidence/contradiction rules.
- **SC-005**: Extraction runs use the compact encoder when a full model is loaded (verified by design or test).
- **SC-006**: At least one documented path exists to use a cleaned or curated KG as augmented seed for a subsequent run.
- **SC-007**: Documentation clearly describes the reliability narrative and maps Contextual_information.md to project capabilities and gaps; new contributors can find this within a single place or linked set of docs.

## Assumptions

- The project continues to target domain-specific KGs (e.g., biomedical, Wikipedia) with a pluggable domain system.
- "Compact encoder" is implemented as a RoBERTa-class architecture; alignment means using it in the extraction path and optionally supporting pretrained weights, not replacing it with a different encoder from the notebook or Python clone.
- FActScore and ValidityScore are used as the canonical names for factuality and validity metrics; exact formulas may match the paper or be documented approximations.
- Reference/ground-truth data for factuality evaluation may be optional (e.g., only in evaluation harness or specific domains); the system can operate without it but cannot produce a factuality score in that case.
- Scope is full implementation: both documentation and implementation of all pipeline stages (provenance, validation/ValidityScore, factuality/FActScore, KG cleaning, encoder-in-path, iterative seed re-use) are in scope.
