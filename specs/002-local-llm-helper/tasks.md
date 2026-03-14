# Tasks: Local LLM Helper for GraphMERT

**Feature**: Local LLM Helper for GraphMERT  
**Date**: 2026-03-13  
**Spec**: specs/002-local-llm-helper/spec.md

---

## Implementation Strategy

**MVP Scope**: User Story 1 (Offline Extraction) - This is the core feature.  
**Delivery**: Incremental - each user story adds value independently.

### Phase Overview

| Phase | Focus | Tasks |
|-------|-------|-------|
| 1 | Setup | 2 tasks |
| 2 | Foundational | 5 tasks |
| 3 | US1 - Offline Extraction | 8 tasks |
| 4 | US2 - Cost Reduction | 3 tasks |
| 5 | US3 - Quality | 3 tasks |
| 6 | Polish | 2 tasks |
| 7 | Integration - Hardcoded vs LLM | 5 tasks |
| 8 | Performance & Memory | 4 tasks |

**Total**: 32 tasks

---

## Phase 1: Setup

Project initialization and dependency setup.

- [X] T001 Add LlamaCpp.jl dependency to GraphMERT/Project.toml
- [X] T002 Create GraphMERT/src/llm/local.jl module file structure

---

## Phase 2: Foundational

Core types and integration points that all user stories depend on.

- [X] T003 Define LocalLLMConfig struct in GraphMERT/src/llm/local.jl
- [X] T004 Define LocalLLMClient struct in GraphMERT/src/llm/local.jl
- [X] T005 Define ModelMetadata struct in GraphMERT/src/llm/local.jl
- [X] T006 Add use_local, local_config, use_ollama, and ollama_config fields to ProcessingOptions in GraphMERT/src/types.jl
- [X] T007 Implement model loading function (load_model) in GraphMERT/src/llm/local.jl

---

## Phase 3: User Story 1 - Offline Knowledge Graph Extraction (Priority: P1)

**Goal**: Enable offline entity and relation extraction using local LLM

**Independent Test**: Disconnect network, run extraction on Wikipedia article, verify entities extracted

**Acceptance Criteria**:
- Entity extraction works without network
- Relation extraction works without network
- Tail formation works without network

### Implementation Tasks

- [X] T008 [P] [US1] Implement discover_entities for LocalLLMClient in GraphMERT/src/llm/local.jl
- [X] T009 [P] [US1] Implement match_relations for LocalLLMClient in GraphMERT/src/llm/local.jl
- [X] T010 [P] [US1] Implement form_tail_from_tokens for LocalLLMClient in GraphMERT/src/llm/local.jl
- [X] T011 [US1] Add backend selection logic to api/extraction.jl (use LocalLLMClient when use_local=true)
- [X] T012 [US1] Add LocalLLMClient export to GraphMERT/src/GraphMERT.jl
- [X] T013 [US1] Add model validation (file exists, valid GGUF) with error messages in GraphMERT/src/llm/local.jl
- [X] T014 [US1] Handle empty text/entities/tokens edge cases per contracts in GraphMERT/src/llm/local.jl
- [X] T015 [US1] Test offline extraction with sample Wikipedia article

---

## Phase 4: User Story 2 - Reduce Extraction Costs (Priority: P2)

**Goal**: Zero API costs through local inference

**Independent Test**: Process 100 articles, verify no external API calls made

**Acceptance Criteria**:
- No HTTP requests to external LLM APIs
- All inference runs locally on CPU

### Implementation Tasks

- [X] T016 [P] [US2] Add network call detection/verification utility in GraphMERT/src/llm/local.jl
- [X] T017 [US2] Verify batch processing (10 articles) completes without network in test environment
- [X] T018 [US2] Document verification procedure for offline operation in quickstart.md

---

## Phase 5: User Story 3 - Maintain Extraction Quality (Priority: P3)

**Goal**: Achieve acceptable quality with local LLM vs external API baseline

**Independent Test**: Compare local vs API extraction on same documents, measure recall

**Acceptance Criteria**:
- Entity recall ≥ 70% of external API baseline
- At least 80% of entities found by reference API on 50 articles

### Implementation Tasks

- [X] T019 [P] [US3] Add quality comparison test utility in GraphMERT/test/local/
- [X] T020 [US3] Run comparison tests with sample set, document results
- [X] T021 [US3] Tune temperature/context parameters if quality below threshold

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T022 Add docstrings to all exported functions per constitution requirements
- [X] T023 Update quickstart.md with final usage examples and troubleshooting

---

## Phase 7: Integration - Hardcoded vs LLM Comparison

**Goal**: Wire LocalLLM into Wikipedia domain extraction and compare quality

**Independent Test**: Extract same article with both approaches, measure recall

**Acceptance Criteria**:
- LLM discovers entities when use_local=true
- Both hardcoded and LLM produce valid Entity objects
- Can compare precision/recall between approaches

### Implementation Tasks

- [X] T024 [P] [US1] Pass llm_client to discover_head_entities in GraphMERT/src/api/extraction.jl
- [X] T025 [US1] Modify Wikipedia domain to use LocalLLM when use_local=true in GraphMERT/src/domains/wikipedia/entities.jl
- [X] T026 [P] [US3] Create Ollama client module for local LLM in GraphMERT/src/llm/ollama.jl
- [X] T027 [US3] Run comparison on French monarchy articles, document results
- [X] T028 [US3] Analyze quality findings: recall, new entities discovered, recommendations

---

## Phase 8: Performance & Memory Verification

**Goal**: Verify FR-003 (5-min extraction) and FR-006 (4GB RAM limit)

### Implementation Tasks

- [ ] T029 [P] Add performance benchmark for entity extraction in GraphMERT/test/performance/
- [ ] T030 Verify extraction completes in under 5 minutes for standard Wikipedia article
- [ ] T031 Add memory usage verification test for 8GB RAM constraint
- [ ] T032 Document performance results in quickstart.md

---

## Dependencies

```
T001 ──┐
       ├──► T003 ──► T008 ──► T011 ──► T015 ──► T016 ──► T017 ──► T018
T002 ──┘     │                              │                         │
             ├──► T004 ──► T009 ───────────┤                         │
             │          │                   │                         │
             ├──► T005 ──► T010 ───────────┤                         │
             │          │                   │                         │
             ├──► T006 ──► T012 ───────────┤                         │
             │                              │                         │
             └──► T007 ──► T013 ───────────┘                         │
                                              ▼
             T019 ──► T020 ◄─────────────────┤
                      │
                      ▼
             T021 ◄──────────────────────────┘

T022, T023 ──► (Final polish, no dependents)

T024 ──► T025 ──► T027 ──► T028
T026 ──────────────────────────┘
```

---

## Parallel Execution Opportunities

| Tasks | Reason |
|-------|--------|
| T003, T004, T005 | Independent struct definitions |
| T008, T009, T010 | Independent method implementations |
| T016, T017, T018 | Independent verification tasks |
| T019, T020, T021 | Sequential quality testing |
| T024, T026 | Different files, can run in parallel |

---

## Independent Test Criteria

### User Story 1 (Offline Extraction)
- Can run extraction with network disabled
- All three extraction stages (entity, relation, tail) produce output

### User Story 2 (Cost Reduction)
- Zero HTTP calls to external APIs during extraction
- CPU-only inference verified

### User Story 3 (Quality)
- Recall ≥ 70% of baseline on same documents
- ≥ 80% entity overlap on 50-article set
