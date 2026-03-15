# Implementation Plan: CPU-only llama.cpp in Development Environment

**Branch**: `004-devenv-llamacpp-cpu` | **Date**: 2026-03-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/004-devenv-llamacpp-cpu/spec.md`

## Summary

Modify the Nix/devenv configuration so that entering the development environment (`devenv shell` or similar) automatically provides a CPU-only llama.cpp: check for an existing CPU-only build from a previous run and reuse it; when not available, download and build the OpenBLAS version and install it as a Nix derivation/flake so it is available like other packages. No CUDA dependency; provisioning is triggered on environment entry.

## Technical Context

**Language/Version**: Nix (flake/devenv); Nixpkgs for llama.cpp/OpenBLAS
**Primary Dependencies**: devenv, Nixpkgs (or flake inputs), llama.cpp source or package, OpenBLAS
**Storage**: Nix store (derivations, build outputs); no new external DB
**Testing**: devenv tests (e.g. enterTest), manual verification that llama.cpp is on PATH and CPU-only; optional automated check that binary runs without CUDA
**Target Platform**: Same as existing devenv (Linux/macOS; host platform from pkgs.stdenv.hostPlatform)
**Project Type**: Development environment configuration (Nix/devenv)
**Performance Goals**: First-time build may take minutes; subsequent entry reuses existing build (fast). No hard latency target; "within 5 minutes" for first-time setup per SC-001.
**Constraints**: CPU-only (no CUDA); reproducible; must integrate with existing devenv.nix and Nix store
**Scale/Scope**: Single developer machine or CI; one llama.cpp variant per environment; no multi-tenant or distributed scope

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre–Phase 0

- [x] **Scientific Accuracy**: N/A (dev-environment change; no algorithms). Reproducibility of build is aligned with Reproducible Research.
- [x] **Performance Excellence**: Reuse from previous run avoids redundant build; first-time build time documented/acceptable.
- [x] **Reproducible Research**: Nix derivations and flake provide deterministic, reproducible build; dependencies pinned via Nix.
- [x] **Comprehensive Testing**: Verification that `devenv shell` provides llama.cpp and that it is CPU-only (manual or enterTest); no public API coverage target for this feature.
- [x] **Clear Documentation**: Quickstart and plan document how to enter environment and use llama.cpp; failure modes documented.
- [x] **Code Quality**: Nix/devenv config follows project style; minimal, justified changes to devenv.nix.
- [x] **Package Management**: New dependency (llama.cpp CPU-only/OpenBLAS) justified by spec; integrated as Nix derivation/flake.

**Constitution Compliance**: ✅ **PASS** (no violations; principles applied where applicable to dev-environment work)

### Post–Phase 1 (design complete)

- [x] **Scientific Accuracy**: N/A.
- [x] **Performance Excellence**: Data model and contracts assume reuse and single build path.
- [x] **Reproducible Research**: research.md and data-model.md describe deterministic provisioning and reuse.
- [x] **Comprehensive Testing**: Contracts and quickstart define verification steps.
- [x] **Clear Documentation**: quickstart.md and contracts added.
- [x] **Code Quality**: Design is minimal and consistent with existing devenv layout.

**Constitution Compliance**: ✅ **PASS**

## Project Structure

### Documentation (this feature)

```text
specs/004-devenv-llamacpp-cpu/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (environment contract)
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created by /speckit.plan)
```

### Source Code (repository root)

Configuration only; no new source trees:

```text
devenv.nix               # Modified: add CPU-only llama.cpp (reuse check + OpenBLAS build/install)
devenv.yaml              # Optional: reference if needed for devenv version
```

**Structure Decision**: Single repo; changes are confined to `devenv.nix` (and optionally flake inputs or a local overlay/override). No new apps or libraries; llama.cpp is supplied by Nix and exposed in the dev environment.

## Phase 2: Task Planning

*Phase 2 is executed by the `/speckit.tasks` command. This section plans what Phase 2 should cover.*

### 2.1 Objective of Phase 2

Break the implementation into ordered, testable tasks: (1) add CPU-only llama.cpp to devenv (override or derivation), (2) ensure provisioning is triggered on `devenv shell`, (3) ensure reuse when existing build present, (4) document and verify.

### 2.2 Task categories (for tasks.md)

1. **Nix/devenv configuration**
   - Add or override package so that llama.cpp is built CPU-only (OpenBLAS, no CUDA). Options: pkgs.llama-cpp with override, or custom derivation/flake input.
   - Ensure the package is included in `packages` (or equivalent) so it is available when entering the shell.

2. **Trigger and reuse**
   - Provisioning is automatic on `devenv shell` (standard Nix/devenv behavior when the package is in `packages`). Document that entry triggers provisioning.
   - Reuse: Nix store naturally reuses existing derivations; ensure the same derivation is used so that a previous build is reused (no duplicate build).

3. **Verification and docs**
   - Quickstart: run `devenv shell`, then run llama.cpp (e.g. `llama-cli --help` or equivalent), confirm CPU-only.
   - enterTest or script: optional check that llama.cpp is on PATH and runs.
   - Document failure modes (network failure, build failure).

4. **Edge cases**
   - Network unavailable: Nix will fail fetch; document retry or offline cache.
   - Host has CUDA: CPU-only build is the one provided; no change to default behavior.

## Complexity Tracking

No constitution violations requiring justification.
