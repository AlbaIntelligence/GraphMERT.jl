# Tasks: CPU-only llama.cpp in Development Environment

**Input**: Design documents from `specs/004-devenv-llamacpp-cpu/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: No test tasks included; spec does not explicitly request TDD or test tasks. Verification tasks (T004, T005) validate acceptance scenarios.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2)
- Include exact file paths in descriptions

## Path Conventions

- Repository root: `devenv.nix`, `devenv.yaml`
- Spec and docs: `specs/004-devenv-llamacpp-cpu/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm environment and file layout before modifying configuration

- [x] T001 Verify `devenv.nix` exists at repository root and document its current `packages` layout and any existing llama-cpp comment for modification

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add CPU-only llama.cpp to the development environment so that entry triggers provisioning, reuse works via Nix store, and OpenBLAS is built when not present.

**⚠️ CRITICAL**: User story verification cannot pass until this phase is complete.

- [x] T002 Define CPU-only llama.cpp (OpenBLAS, no CUDA) in `devenv.nix` at repository root via Nixpkgs override (e.g. `pkgs.llama-cpp.override { cudaSupport = false; openblasSupport = true; }`) or a custom derivation/flake input that builds the OpenBLAS variant
- [x] T003 Add the CPU-only llama.cpp package to the `packages` list in `devenv.nix` at repository root so it is available on PATH when running `devenv shell`

**Checkpoint**: Foundation ready — entering the shell should provision or reuse llama.cpp; user story verification can begin.

---

## Phase 3: User Story 1 - Use llama.cpp from the development environment without CUDA (Priority: P1) 🎯 MVP

**Goal**: Developer can run `devenv shell` and have llama.cpp available on PATH, running with CPU only (no CUDA).

**Independent Test**: Run `devenv shell`, then run the llama.cpp binary (e.g. `llama-cli --help` or package executable); confirm it runs and uses CPU only.

### Implementation for User Story 1

- [x] T004 [US1] Verify that running `devenv shell` at repository root triggers provisioning and that the llama.cpp binary is on PATH inside the shell per `specs/004-devenv-llamacpp-cpu/spec.md` US1 acceptance scenario 1
- [x] T005 [US1] Verify that the llama.cpp binary runs (e.g. `llama-cli --help` or equivalent) and does not require or load CUDA per `specs/004-devenv-llamacpp-cpu/spec.md` US1 acceptance scenario 2

**Checkpoint**: User Story 1 is complete; developers can use llama.cpp from the dev environment without CUDA.

---

## Phase 4: User Story 2 - Reproducible CPU-only build (Priority: P2)

**Goal**: First entry builds or fetches OpenBLAS llama.cpp; subsequent entries reuse the existing build (no redundant download/build). Behavior is reproducible across machines.

**Independent Test**: On a clean or fresh environment, run `devenv shell` and confirm llama.cpp is present and CPU-only; run `devenv shell` again and confirm no full rebuild (reuse). Repeat on another machine to confirm reproducibility.

### Implementation for User Story 2

- [x] T006 [US2] Verify that a second run of `devenv shell` at repository root reuses the existing CPU-only build (no full rebuild) per `specs/004-devenv-llamacpp-cpu/spec.md` US2 acceptance scenario 2
- [x] T007 [US2] Document in `specs/004-devenv-llamacpp-cpu/quickstart.md` (or link from README) the first-time vs subsequent entry behavior, reuse semantics, and failure modes (network/build failure) per FR-006 and SC-004

**Checkpoint**: User Stories 1 and 2 are both satisfied; reproducible build and reuse are verified and documented.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Optional hardening and validation across both user stories.

- [x] T008 [P] Add optional `enterTest` in `devenv.nix` that asserts the llama.cpp binary is on PATH (e.g. `which llama-cli` or equivalent) so that `devenv test` can validate availability
- [x] T009 Run quickstart validation per `specs/004-devenv-llamacpp-cpu/quickstart.md`: enter environment, verify llama.cpp, and confirm failure-mode documentation is accurate

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001) — BLOCKS all user story verification
- **User Story 1 (Phase 3)**: Depends on Foundational (T002, T003) — verification tasks T004, T005
- **User Story 2 (Phase 4)**: Depends on Foundational; verification T006, T007 can run after US1 verification
- **Polish (Phase 5)**: Depends on Phase 3 and 4 verification being done

### User Story Dependencies

- **User Story 1 (P1)**: Implemented by Phase 2 (same Nix change). Verification (T004, T005) after T002–T003.
- **User Story 2 (P2)**: Same Nix change provides reuse and OpenBLAS build. Verification (T006, T007) after T002–T003; T007 (docs) can follow T006.

### Within Each User Story

- Phase 2 tasks T002 then T003 (define package, then add to list).
- US1 verification (T004, T005) can be done in parallel once Phase 2 is complete.
- US2 verification (T006, T007): T006 then T007 (verify reuse, then document).

### Parallel Opportunities

- T004 and T005 (US1 verification) can run in parallel (different acceptance checks).
- T008 and T009 (Polish) can run in parallel (enterTest vs quickstart validation).

---

## Parallel Example: User Story 1

```bash
# After Phase 2 is complete, run both US1 verification checks:
Task T004: "Verify that running devenv shell at repository root triggers provisioning and llama.cpp on PATH"
Task T005: "Verify llama.cpp binary runs and does not require CUDA"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002, T003)
3. Complete Phase 3: User Story 1 (T004, T005)
4. **STOP and VALIDATE**: Independent test for US1 — run `devenv shell`, run llama.cpp, confirm CPU-only
5. Demo: developer can use llama.cpp without CUDA

### Incremental Delivery

1. Setup + Foundational → llama.cpp available on entry
2. Add US1 verification → MVP complete
3. Add US2 verification and docs → Reuse and reproducibility confirmed and documented
4. Add Polish (enterTest, quickstart validation) → Cross-cutting validation

### Single-File Focus

All implementation is in `devenv.nix`. T002 and T003 are the only implementation tasks; the rest are verification and documentation. A single developer can complete T001 → T003, then run T004–T007 and T008–T009.

---

## Notes

- [P] tasks = different concerns or files, safe to run in parallel
- [US1]/[US2] label maps task to user story for traceability
- No separate test suite requested by spec; verification tasks validate acceptance scenarios
- Commit after T003 (implementation complete) and after T007 (docs complete)
- If T002/T003 fail (e.g. Nixpkgs override API differs), adjust override or use flake/overlay per research.md
