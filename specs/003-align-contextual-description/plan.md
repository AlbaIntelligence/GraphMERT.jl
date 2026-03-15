# Implementation Plan: Align Project with Contextual Description and Reliability Narrative

**Branch**: `003-align-contextual-description` | **Date**: 2026-03-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/003-align-contextual-description/spec.md`

## Summary

Implement the full reliability pipeline so the project reflects Contextual_information.md: (1) **provenance** — every triple carries document and segment/sentence reference; (2) **ontology validation** and **ValidityScore**; (3) **factuality evaluation** (**FActScore**) when reference data exists; (4) **KG cleaning** (remove/rectify unsupported, low-confidence, contradicted triples); (5) **encoder in extraction path** (wire persistence so extraction uses RoBERTa+H-GAT); (6) **iterative seed re-use** (cleaned/curated KG as augmented seed); (7) **documentation** mapping the reliability narrative to capabilities and roadmap. Technical approach: extend existing types and evaluation modules, add provenance record shape, wire `load_model` to full model, and document in REFERENCE_SOURCES_AND_ENCODER and project docs.

## Technical Context

**Language/Version**: Julia 1.10+ (Project.toml, LTS compatibility per existing project)
**Primary Dependencies**: GraphMERT package (RoBERTa, H-GAT, leafy chain, domains); existing evaluation modules (evaluation/factscore.jl, evaluation/validity.jl); persistence layer (models/persistence.jl)
**Storage**: In-memory KGs; provenance and metadata on triples; optional export (CSV/JSON); no new external DB
**Testing**: Test.jl; unit tests in GraphMERT/test/unit/, integration in test/integration/; constitution requires >80% coverage for public APIs
**Target Platform**: Laptop/server (research and scientific computing); same as existing GraphMERT.jl
**Project Type**: Library (Julia package)
**Performance Goals**: Align with existing goals (e.g., process tokens efficiently); validation and cleaning scale with KG size—MVP uses in-memory processing with configurable limits; batch/streaming can be added post-MVP if needed
**Constraints**: Reproducibility (seeds, versioned deps); no hardcoded secrets; ontology optional (graceful degradation)
**Scale/Scope**: Single-document and multi-document extraction; KGs sized for in-memory processing with configurable limits for very large corpora; acceptable behavior for scale is documented (limits, in-memory design) so users can reason about it—benchmarks for critical algorithms (validation, cleaning, factuality) are added in Polish phase or documented as post-MVP

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre–Phase 0

- [x] **Scientific Accuracy**: Algorithms (FActScore, ValidityScore, cleaning rules) backed by paper/spec and documented; provenance and validation are mathematically well-defined.
- [x] **Performance Excellence**: Provenance and validation add minimal overhead; cleaning and evaluation designed for batch processing; complexity documented.
- [x] **Reproducible Research**: Random seeds and env specs unchanged; new behavior does not introduce non-determinism; provenance supports audit.
- [x] **Comprehensive Testing**: New/updated public APIs (provenance access, validate, factuality, clean, augmented seed) require unit and integration tests; coverage target >80% per constitution.
- [x] **Clear Documentation**: Docstrings for new/exposed functions; docs describe reliability narrative and mapping to Contextual_information.md; quickstart updated.
- [x] **Code Quality**: Julia style; types for provenance/validity/cleaning; no new unnecessary dependencies.
- [x] **Package Management**: No new external dependencies unless justified (e.g., optional reference-data loaders).

**Constitution Compliance**: ✅ **PASS** (no violations)

### Post–Phase 1 (design complete)

- [x] **Scientific Accuracy**: research.md and data-model.md define FActScore, ValidityScore, provenance, and cleaning with clear semantics.
- [x] **Performance Excellence**: Contracts and data model assume batch processing and configurable limits; no new blocking I/O.
- [x] **Reproducible Research**: No new non-determinism; provenance supports audit.
- [x] **Comprehensive Testing**: Contracts require testable APIs; plan references unit/integration tests.
- [x] **Clear Documentation**: quickstart.md and contracts/01-reliability-api.md added; FR-007 covered by docs mapping.
- [x] **Code Quality**: Data model and contracts stay technology-agnostic where appropriate; implementation will follow Julia style.

**Constitution Compliance**: ✅ **PASS** (no violations after design)

## Project Structure

### Documentation (this feature)

```text
specs/003-align-contextual-description/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (reliability API)
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created by /speckit.plan)
```

### Source Code (repository root)

Existing GraphMERT package layout; changes are additive and wiring:

```text
GraphMERT/
├── src/
│   ├── api/
│   │   ├── extraction.jl     # Provenance attachment; encoder-in-path via model
│   │   └── (new or extended) reliability.jl  # validate_kg, clean_kg, factuality_score, etc.
│   ├── evaluation/
│   │   ├── factscore.jl      # Wire FActScore; optional reference data
│   │   └── validity.jl       # Wire ValidityScore; ontology checks
│   ├── models/
│   │   └── persistence.jl    # Wire load_model to full RoBERTa+H-GAT model
│   ├── types.jl              # ProvenanceRecord or extend Relation/Entity
│   ├── domains/              # Ontology/validation hooks per domain
│   └── ...
└── test/
    ├── unit/                 # Provenance, validation, cleaning, factuality
    └── integration/          # End-to-end extraction with provenance; iterative seed
```

**Structure Decision**: Single Julia package (GraphMERT). No new top-level apps. Reliability pipeline is implemented inside existing `GraphMERT/src` (api, evaluation, types, models) and documented under `reports/` and `specs/003-align-contextual-description/`.

## Phase 2: Task Planning

*Phase 2 is executed by the `/speckit.tasks` command, which produces `tasks.md` and optional task breakdown files. This section plans what Phase 2 should cover so that task creation has clear input.*

### 2.1 Objective of Phase 2

Break the implementation into ordered, testable tasks that implement the reliability pipeline (provenance, validation, factuality, cleaning, encoder-in-path, iterative seed, documentation) without violating the constitution and following the contracts and data model.

### 2.2 Task categories (for tasks.md)

1. **Provenance**
   - Add or extend `ProvenanceRecord` (or equivalent) in types; attach to each triple at extraction.
   - Ensure extraction pipeline populates document_id and segment_id (or span); expose `get_provenance(kg, relation)` (or equivalent).

2. **Validation**
   - Wire ontology validation in evaluation/validity.jl; accept domain/ontology; return ValidityReport (ValidityScore).
   - Graceful degradation when ontology missing (FR-008).

3. **Factuality**
   - Wire FActScore in evaluation/factscore.jl; optional reference data; return FactualityScore when reference present.

4. **KG cleaning**
   - Implement or wire clean_kg with CleaningPolicy (min_confidence, require_provenance, contradiction handling); output cleaned KG.

5. **Encoder-in-path**
   - Ensure load_model returns full GraphMERTModel; extraction uses encoder when model is full model (no bypass).

6. **Iterative seed**
   - Document and implement at least one path to use cleaned/curated KG as augmented seed (e.g., export → seed config → next run).

7. **Documentation**
   - Map Contextual_information.md to capabilities and roadmap; update quickstart, REFERENCE_SOURCES_AND_ENCODER (or equivalent); docstrings for new APIs.

8. **Testing**
   - Unit tests for provenance, validate_kg, clean_kg, evaluate_factscore; integration tests for full pipeline; coverage ≥80% for new public APIs.

## Complexity Tracking

No constitution violations requiring justification.
