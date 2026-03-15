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
   - Config: `enable_provenance_tracking` in extraction options.
   - Tests: unit (provenance on relations), integration (extract → inspect provenance).

2. **Ontology validation and ValidityScore**
   - Extend or use `evaluation/validity.jl` to accept domain/ontology; compute score and optional per-triple report.
   - Graceful degradation when ontology missing (FR-008).
   - Tests: unit (validation logic), integration (validate_kg with domain).

3. **Factuality (FActScore)**
   - Wire or extend `evaluation/factscore.jl` to accept reference data and return FactualityScore; document “no reference ⇒ no score.”
   - Tests: unit (score computation), integration (evaluate_factscore with reference).

4. **KG cleaning**
   - Implement `clean_kg(kg; policy)` (or equivalent) per contract; configurable min_confidence, require_provenance, contradiction handling.
   - Output new KnowledgeGraph; optional removal report.
   - Tests: unit (cleaning rules), integration (clean_kg → verify counts and use as seed).

5. **Encoder in extraction path**
   - Wire `models/persistence.jl` so `load_model(path)` returns full model (RoBERTa + H-GAT) when loading a full checkpoint.
   - Ensure `api/extraction.jl` uses the model’s encoder when present (no bypass).
   - Tests: integration (load model → run extraction → verify encoder used, e.g., by design or lightweight test).

6. **Iterative seed re-use**
   - Document and implement at least one path: cleaned/curated KG → export or in-memory → use as augmented seed in training or extraction (extend seed_injection or domain seed loaders as needed).
   - Tests: integration (clean → configure seed → run extraction or training).

7. **Documentation**
   - Update REFERENCE_SOURCES_AND_ENCODER (and related reports) with reliability narrative and mapping from Contextual_information.md to capabilities/gaps.
   - Update PROJECT_STATUS, AGENTS.md, and user-facing docs (quickstart, API docs) for provenance, validation, factuality, cleaning, iterative seed.
   - Ensure SC-007: single place or linked set of docs for the narrative.

### 2.3 Dependencies between task groups

- **Provenance** is required before **cleaning** (require_provenance) and before **iterative seed** (cleaned KG carries provenance).
- **Encoder-in-path** is independent but should be early so extraction tests use the real model.
- **Validation** and **factuality** can proceed in parallel after types/contracts are clear.
- **Documentation** can be incremental (per capability) with a final pass for the full narrative.

### 2.4 Deliverables of Phase 2

- **tasks.md**: Ordered list or breakdown of tasks with acceptance criteria, linked to spec (FR-xxx, SC-xxx) and contracts.
- Optional: per-category task files (e.g., `tasks/01-provenance.md`, `tasks/02-validation.md`, …) if the workflow supports it.
- Each task should be testable and sized so that implementation and tests can be completed without blocking unrelated work where possible.

### 2.5 Handoff to /speckit.tasks

When running **Create Tasks** (`/speckit.tasks`), use this plan (especially § Phase 2), the [spec](spec.md), [contracts](contracts/01-reliability-api.md), and [data-model](data-model.md) to generate `tasks.md` and any task breakdown files. Preserve the task categories and dependency order above unless the tool produces a more detailed structure.

## Complexity Tracking

No constitution violations; this section is empty.
