---

description: Task list for Wikipedia KG Testing feature
---

# Tasks: Wikipedia Knowledge Graph Testing

**Input**: Design documents from `specs/001-wikipedia-kg-testing/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `GraphMERT/test/` at repository root
- Paths shown below assume GraphMERT package path

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and test data collection

- [X] T001 [P] Create test data directory structure in GraphMERT/test/wikipedia/
- [X] T002 [P] Collect Wikipedia article texts for French monarchy testing in test/wikipedia/data/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Test utilities and reference data that MUST be complete before ANY user story can be tested

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Create test utilities module in GraphMERT/test/wikipedia/test_utils.jl
- [X] T004 [P] Create reference facts dataset in GraphMERT/test/wikipedia/reference_facts.jl
- [X] T005 [P] Create quality metrics computation helper in GraphMERT/test/wikipedia/metrics.jl
- [X] T006 Setup test configuration and fixtures in GraphMERT/test/wikipedia/fixtures.jl

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - French Monarchy Entity Extraction (Priority: P1) 🎯 MVP

**Goal**: Validate that Wikipedia domain correctly identifies royal titles, dynastic relationships, and historical figures from French monarchy Wikipedia articles

**Independent Test**: Run extraction on French monarchy articles and verify entities like "Louis XIV", "Henry IV" are identified with correct entity types

### Tests for User Story 1

- [X] T007 [P] [US1] Create entity extraction unit tests in GraphMERT/test/wikipedia/test_entity_extraction.jl
- [X] T008 [P] [US1] Create entity type classification tests in GraphMERT/test/wikipedia/test_entity_types.jl

### Implementation for User Story 1

- [X] T009 [US1] Create entity extraction test runner in GraphMERT/test/wikipedia/run_entity_extraction.jl
- [X] T010 [US1] Run entity extraction on Louis XIV article and validate results
- [X] T011 [US1] Run entity extraction on Henry IV article and validate results
- [X] T012 [US1] Verify entity precision meets 80% threshold (SC-001)

**Checkpoint**: At this point, entity extraction should be validated against French monarchy text

---

## Phase 4: User Story 2 - Dynastic Relation Extraction (Priority: P1)

**Goal**: Validate that relation extraction correctly identifies dynastic relationships between French monarchs

**Independent Test**: Extract relations between king entities and verify PARENT_OF, SPOUSE_OF, REIGNED_AFTER relationships

### Tests for User Story 2

- [X] T013 [P] [US2] Create relation extraction unit tests in GraphMERT/test/wikipedia/test_relation_extraction.jl

### Implementation for User Story 2

- [X] T014 [US2] Create relation extraction test runner in GraphMERT/test/wikipedia/run_relation_extraction.jl
- [X] T015 [US2] Run relation extraction between Louis XIV and Louis XV entities
- [X] T016 [US2] Verify relation precision meets 70% threshold (SC-002)

**Checkpoint**: At this point, relation extraction should be validated

---

## Phase 5: User Story 3 - Knowledge Graph Quality Assessment (Priority: P2)

**Goal**: Assess overall quality of extracted knowledge graphs against known French monarchy facts

**Independent Test**: Construct KG from French monarchy articles and evaluate precision, recall, F1 against reference facts

### Tests for User Story 3

- [X] T017 [P] [US3] Create quality assessment tests in GraphMERT/test/wikipedia/test_quality.jl

### Implementation for User Story 3

- [X] T018 [US3] Create quality assessment test runner in GraphMERT/test/wikipedia/run_quality_assessment.jl
- [X] T019 [US3] Compute quality metrics (entity precision, relation precision, recall)
- [X] T020 [US3] Verify relation precision exceeds 70% threshold (SC-002)
- [X] T021 [US3] Verify 75% of known facts captured (SC-004)
- [X] T022 [US3] Verify confidence scoring AUC > 0.7 (SC-005)
- [X] T023 [US3] Test batch processing of 20 articles (SC-006)

### Polish Phase

- [X] T024 [P] Update examples/wikipedia/01_wikipedia_entity_extraction.jl with test results
- [X] T025 [P] Create test summary report in GraphMERT/test/wikipedia/TEST_REPORT.md
- [X] T026 Document test findings and recommendations in reports/
- [X] T027 [P] Verify test coverage meets 80% threshold (constitution requirement)
- [X] T028 [P] Implement and test KG export to JSON format (FR-008)
- [X] T029 [P] Implement and test KG export to CSV format (FR-008)
- [X] T030 [P] Document random seeds for reproducibility (constitution requirement)
- [X] T031 Run full test suite and verify no regressions

**Checkpoint**: All user stories should now be independently functional

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Can run in parallel with US1
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 and US2 for full KG

### Within Each User Story

- Tests should run before or alongside implementation
- Entity extraction before relation extraction
- Full pipeline after component validation

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, US1 and US2 can start in parallel
- Tests for a user story marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Entity extraction unit tests in test/wikipedia/test_entity_extraction.jl"
Task: "Entity type classification tests in test/wikipedia/test_entity_types.jl"

# Run entity extraction validations:
Task: "Extract entities from Louis XIV article"
Task: "Extract entities from Henry IV article"
Task: "Extract entities from Marie Antoinette article"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test entity extraction independently

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Document results
3. Add User Story 2 → Test independently → Document results
4. Add User Story 3 → Test independently → Full quality assessment

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Entity Extraction)
   - Developer B: User Story 2 (Relation Extraction)
3. Both complete → User Story 3 (Quality Assessment)
4. Document and polish

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests pass before documenting results
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- This is a TESTING feature - primary goal is validation, not implementation
