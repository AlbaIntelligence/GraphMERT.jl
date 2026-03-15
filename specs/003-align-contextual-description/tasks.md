# Tasks: Align Project with Contextual Description and Reliability Narrative

**Input**: Design documents from `specs/003-align-contextual-description/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing. Tests are included per plan Phase 2 (unit/integration per category).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US5)
- Include exact file paths in descriptions

## Path Conventions

- **GraphMERT package**: `GraphMERT/src/`, `GraphMERT/test/` at repository root
- **Specs**: `specs/003-align-contextual-description/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure feature branch and design artifacts are in place; add reliability API placeholder.

- [x] T001 Verify design artifacts (spec.md, plan.md, data-model.md, contracts/, research.md, quickstart.md) and GraphMERT package structure per specs/003-align-contextual-description/plan.md
- [x] T002 [P] Create GraphMERT/src/api/reliability.jl with module stub and docstring referencing contracts/01-reliability-api.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Types and config that ALL user stories depend on. No user story work can begin until this phase is complete.

- [x] T003 Add ProvenanceRecord struct (document_id, segment_id, optional span_start, span_end, context) to GraphMERT/src/types.jl per data-model.md
- [x] T004 Add enable_provenance_tracking field to ProcessingOptions in GraphMERT/src/types.jl and set default in default_processing_options in GraphMERT/src/config.jl (contract §1.2)

**Checkpoint**: Foundation ready — user story implementation can begin

---

## Phase 3: User Story 1 — Extract KG with traceable provenance (Priority: P1) — MVP

**Goal**: Every triple in the extracted KG is traceable to a source span (document + segment/sentence); users can request provenance for any triple.

**Independent Test**: Run extraction on a short corpus with enable_provenance_tracking=true; inspect output so each triple has document_id and segment_id; call get_provenance(kg, relation) and receive source reference.

### Implementation for User Story 1

- [x] T005 [US1] Populate ProvenanceRecord for each relation in extract_knowledge_graph in GraphMERT/src/api/extraction.jl when enable_provenance_tracking is true; when corpus is empty or no triples extracted return empty KG with no phantom provenance (FR-001, edge case)
- [x] T006 [US1] Implement get_provenance(kg, relation_or_index) returning ProvenanceRecord in GraphMERT/src/api/reliability.jl (or extraction.jl) per contracts/01-reliability-api.md §1
- [x] T007 [US1] Add unit test for provenance attachment and get_provenance in GraphMERT/test/unit/test_provenance.jl
- [x] T008 [US1] Add integration test: extract_knowledge_graph then assert each relation has provenance and get_provenance returns valid record; include empty corpus / zero triples case (no phantom provenance) in GraphMERT/test/integration/test_extraction_provenance.jl

**Checkpoint**: User Story 1 is independently testable; extraction returns provenance when enabled

---

## Phase 4: User Story 2 — Validate triples against ontology and report validity (Priority: P2)

**Goal**: Extracted triples are checked against the seed ontology; operators get a ValidityScore (or ValidityReport) when validation runs; graceful degradation when ontology is missing.

**Independent Test**: Run extraction with a domain that has an ontology; run validate_kg(kg, domain) (or evaluate_validity); obtain validity score/report. Run with missing ontology and confirm graceful behavior.

### Implementation for User Story 2

- [x] T009 [P] [US2] Add ValidityReport type (score, total_triples, valid_count, optional per_triple, ontology_id) to GraphMERT/src/types.jl or GraphMERT/src/evaluation/validity.jl per data-model.md
- [x] T010 [US2] Extend evaluate_validity in GraphMERT/src/evaluation/validity.jl to accept domain or ontology and return ValidityReport with score in [0,1]; support reject/flag (and optionally correct) per policy (FR-002)
- [x] T011 [US2] Add validate_kg(kg, domain) wrapper (or export) in GraphMERT/src/api/reliability.jl per contracts/01-reliability-api.md §2
- [x] T012 [US2] Implement graceful degradation when ontology is missing or incomplete in GraphMERT/src/evaluation/validity.jl (FR-008)
- [x] T013 [US2] Add integration test for validate_kg with domain and with missing ontology in GraphMERT/test/integration/test_validity.jl

**Checkpoint**: User Story 2 is independently testable; ValidityScore available when ontology present

---

## Phase 5: User Story 3 — Measure factuality and support KG cleaning (Priority: P2)

**Goal**: FActScore when reference data is available; KG cleaning step that removes/rectifies unsupported, low-confidence, or contradicted triples per configurable policy; cleaned KG usable as augmented seed.

**Independent Test**: Run evaluate_factscore(kg, reference) when reference exists and get score; run clean_kg(kg; policy) and verify reduced triple set and policy adherence.

### Implementation for User Story 3

- [x] T014 [P] [US3] Add FactualityScore type (score, total_triples, correct_count, reference_id) and CleaningPolicy type (min_confidence, require_provenance, contradiction_handling) to GraphMERT/src/types.jl or GraphMERT/src/api/reliability.jl per data-model.md
- [x] T015 [US3] Extend evaluate_factscore in GraphMERT/src/evaluation/factscore.jl to accept reference data and return FactualityScore; document no-score when reference absent (FR-003)
- [x] T016 [US3] Implement clean_kg(kg; policy) returning new KnowledgeGraph in GraphMERT/src/api/reliability.jl per contracts/01-reliability-api.md §4; document in-memory design and configurable limits for scale (FR-004, edge case)
- [x] T017 [US3] Add unit test for clean_kg rules (min_confidence, require_provenance) in GraphMERT/test/unit/test_cleaning.jl
- [x] T018 [US3] Add integration test for evaluate_factscore with reference and for clean_kg in GraphMERT/test/integration/test_factscore_cleaning.jl

**Checkpoint**: User Story 3 is independently testable; FActScore and clean_kg available

---

## Phase 6: User Story 4 — Use encoder in extraction path and support iterative improvement (Priority: P3)

**Goal**: load_model returns full model (RoBERTa + H-GAT) when loading a full checkpoint; extraction uses the encoder (no stub bypass); at least one documented path to use cleaned/curated KG as augmented seed.

**Independent Test**: Load model via load_model(path); run extraction and verify encoder is used (by design or lightweight test). Run clean → export/configure as seed → re-run extraction or training.

### Implementation for User Story 4

- [x] T019 [US4] Wire load_model so it returns full model (RoBERTa + H-GAT) when loading a full checkpoint: implement or extend in GraphMERT/src/models/persistence.jl and call from GraphMERT/src/GraphMERT.jl (FR-005)
- [x] T020 [US4] Ensure extraction path in GraphMERT/src/api/extraction.jl uses the model encoder when present (no stub or bypass)
- [x] T021 [US4] Document and implement at least one path for augmented seed (cleaned/curated KG as seed for training or extraction) in GraphMERT/src/training/seed_injection.jl or domain loaders and GraphMERT/docs or reports (FR-006, SC-006)
- [x] T022 [US4] Add integration test: load full model, run extraction, verify encoder used; optional test clean → seed → re-run in GraphMERT/test/integration/test_encoder_in_path.jl

**Checkpoint**: User Story 4 is independently testable; encoder-in-path and iterative seed path available

---

## Phase 7: User Story 5 — Documentation and narrative alignment (Priority: P3)

**Goal**: User- and agent-facing docs describe the reliability narrative (provenance, FActScore, ValidityScore, KG cleaning, ontology, human-in-the-loop) and map Contextual_information.md to capabilities and gaps (FR-007, SC-007).

**Independent Test**: Read REFERENCE_SOURCES_AND_ENCODER, PROJECT_STATUS, and user docs; confirm reliability narrative and mapping are present and findable.

### Implementation for User Story 5

- [x] T023 [P] [US5] Update reports/REFERENCE_SOURCES_AND_ENCODER.md with reliability narrative and mapping from Contextual_information.md to project capabilities and gaps
- [x] T024 [P] [US5] Update reports/PROJECT_STATUS.md to reflect reliability pipeline (provenance, validation, factuality, cleaning, encoder-in-path, iterative seed) and encoder-in-path status
- [x] T025 [US5] Update AGENTS.md and user-facing quickstart (e.g. GraphMERT/docs or README) per specs/003-align-contextual-description/quickstart.md for provenance, validate_kg, clean_kg, factuality, iterative seed
- [x] T026 [US5] Add or update API documentation for get_provenance, validate_kg, clean_kg, evaluate_factscore: complete docstrings with examples for each export in GraphMERT/src and ensure generated docs in GraphMERT/docs/src (constitution: docstrings with examples)

**Checkpoint**: User Story 5 complete; single place or linked set of docs for the narrative

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Validation, exports, and cross-cutting checks.

- [x] T027 [P] Run quickstart.md validation (specs/003-align-contextual-description/quickstart.md) against implemented API
- [x] T028 Export new types and functions (ProvenanceRecord, ValidityReport, FactualityScore, CleaningPolicy, get_provenance, validate_kg, clean_kg) from GraphMERT/src/GraphMERT.jl
- [x] T029 Cross-check implementation against contracts/01-reliability-api.md and fix any gaps
- [x] T030 Verify test coverage ≥80% for new public APIs (get_provenance, validate_kg, clean_kg, evaluate_factscore) per constitution; add unit tests that validate performance where critical (e.g. clean_kg, evaluate_validity) or document waiver in plan
- [x] T031 Add performance benchmarks or regression tests for clean_kg, evaluate_validity, evaluate_factscore in GraphMERT/test/performance/ or document in plan that benchmarks are post-MVP

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 — MVP
- **Phase 4 (US2)**: Depends on Phase 2 — can run in parallel with US1 after Phase 2 if staffed
- **Phase 5 (US3)**: Depends on Phase 2; cleaning benefits from US1 (provenance) but can be implemented with placeholder provenance
- **Phase 6 (US4)**: Depends on Phase 2; iterative seed uses clean_kg (US3)
- **Phase 7 (US5)**: Can start after any story; best after US1–US4 so docs reflect implemented capabilities
- **Phase 8 (Polish)**: Depends on completion of desired user stories; T030 (coverage), T031 (benchmarks) may run after T028/T029

### User Story Dependencies

- **US1 (P1)**: No dependency on other stories — start after Phase 2
- **US2 (P2)**: No dependency on US1 — start after Phase 2
- **US3 (P2)**: Prefer US1 done so require_provenance in cleaning is meaningful; not strictly required
- **US4 (P3)**: Encoder-in-path independent; iterative seed path benefits from US3 (clean_kg)
- **US5 (P3)**: Independent; most valuable after US1–US4 implemented

### Within Each User Story

- Types/structs before functions that use them
- Core implementation before tests
- Integration tests after unit tests or in same phase

### Parallel Opportunities

- Phase 1: T002 [P]
- Phase 2: T003 and T004 are same file — sequential
- After Phase 2: US1, US2 can proceed in parallel; US3 can start in parallel with US1/US2
- Within US2: T009 [P]; within US3: T014 [P]; within US5: T023 [P], T024 [P]
- Phase 8: T027 [P]; T030, T031 (coverage and benchmarks) can follow T029

---

## Parallel Example: User Story 1

```text
# After Phase 2 complete, US1 only:
T005 → T006 → T007, T008 (T007 and T008 can be written in parallel)
```

## Parallel Example: User Stories 2 and 3

```text
# After Phase 2, US2 and US3 in parallel:
US2: T009 [P] then T010 → T011 → T012 → T013
US3: T014 [P] then T015, T016 in parallel → T017, T018
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (provenance)
4. **STOP and VALIDATE**: Run extraction with provenance; call get_provenance; run tests
5. Demo: extract KG and show traceable triples

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. Add US1 → test independently → MVP (provenance)
3. Add US2 → validate_kg, ValidityScore → test independently
4. Add US3 → FActScore, clean_kg → test independently
5. Add US4 → encoder-in-path, iterative seed → test independently
6. Add US5 → documentation → full narrative alignment
7. Polish → exports and contract check

### Parallel Team Strategy

- After Phase 2: Developer A — US1; Developer B — US2; Developer C — US3
- Then US4 (encoder + seed), then US5 (docs), then Polish

---

## Notes

- [P] tasks use different files or have no dependency on in-progress tasks
- [USn] maps task to spec user story for traceability
- Each user story is independently completable and testable per spec
- Commit after each task or logical group
- File paths use GraphMERT/ prefix (package at repo root)
- No after_tasks hooks required (extensions.yml not present)
