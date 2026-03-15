# Feature Specification: CPU-only llama.cpp in Development Environment

**Feature Branch**: `004-devenv-llamacpp-cpu`  
**Created**: 2026-03-15  
**Status**: Draft  
**Input**: User description: "The Nix version of Llamacpp requires a CUDA setup. The task is to modify the devenv.nix so that it downloads/compiles llama.cpp for CPU only, and makes it available to our development environment."

## Clarifications

### Session 2026-03-15

- Q: When is provisioning triggered and how is availability determined? → A: Provisioning is automatically triggered when the developer runs `devenv shell` (or similar commands) to enter the environment. The system checks for an existing CPU-only llama.cpp from a previous run; if present, it is reused. If not available, the system downloads and builds the OpenBLAS version and installs it as any other Nix derivation/flake.

## Summary

The development environment currently cannot provide llama.cpp without requiring a CUDA (GPU) setup. This feature ensures that developers can obtain and use llama.cpp from the development environment using a CPU-only build, with no CUDA or GPU dependency. Entering the environment (e.g. `devenv shell`) automatically triggers provisioning; an existing CPU-only build from a previous run is reused when present; otherwise the OpenBLAS variant is built and installed via the Nix derivation/flake.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use llama.cpp from the development environment without CUDA (Priority: P1)

As a developer, I want to enter the development environment and have llama.cpp available so that I can run local inference and tooling that depends on llama.cpp without installing CUDA or GPU drivers.

**Why this priority**: This is the core value: enabling use of llama.cpp on CPU-only machines (e.g. CI, laptops without GPU, or environments where CUDA is not desired).

**Independent Test**: Run `devenv shell` (or the equivalent entry command); provisioning runs automatically. Then run the llama.cpp binary or library and confirm it runs with CPU backend (no CUDA). Delivers value by making llama.cpp usable where GPU is not available.

**Acceptance Scenarios**:

1. **Given** a machine without CUDA or GPU drivers, **When** the developer runs `devenv shell` (or similar) to enter the environment, **Then** provisioning is triggered automatically and llama.cpp is available (e.g. on PATH) and runs using CPU only.
2. **Given** the development environment is up to date, **When** the developer invokes llama.cpp (e.g. CLI or library), **Then** the build does not require or load CUDA and inference runs on CPU.

---

### User Story 2 - Reproducible CPU-only build (Priority: P2)

As a developer or CI, I want the development environment to obtain llama.cpp in a reproducible way (e.g. download or compile) so that everyone gets the same CPU-only variant without manual steps.

**Why this priority**: Ensures consistency across machines and avoids "works on my machine" issues; supports automation and documentation.

**Independent Test**: On a clean checkout (or fresh environment), run `devenv shell` and verify llama.cpp is present and CPU-only; on a subsequent run, verify that an existing CPU-only build from a previous run is reused when available. Repeat on another machine to confirm reproducibility.

**Acceptance Scenarios**:

1. **Given** a fresh or cleaned development environment setup, **When** the developer runs `devenv shell` (or similar) to enter the environment, **Then** the system checks for an existing CPU-only llama.cpp from a previous run; if not available, it downloads and builds the OpenBLAS version and installs it (e.g. as a Nix derivation/flake), and llama.cpp is available without manual intervention.
2. **Given** a previous run already provided CPU-only llama.cpp, **When** the developer runs `devenv shell` again, **Then** the existing build is detected and reused (no redundant download or build).

---

### Edge Cases

- What happens when the network is unavailable during first-time setup? The system should fail clearly (e.g. with a message that download or fetch failed) and not leave a broken or half-installed state that is hard to recover from.
- What happens when the host has CUDA installed? The development environment should still provide and use the CPU-only build by default so that behavior is consistent and optional GPU use is not forced.
- How does the system behave if the build or download of llama.cpp fails? The failure should be visible (e.g. environment entry fails or a clear error is shown) and documented so developers can fix or work around (e.g. retry, check network, or use a fallback).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Provisioning of llama.cpp MUST be automatically triggered when the developer enters the development environment (e.g. by running `devenv shell` or similar entry commands).
- **FR-002**: The system MUST check for an existing CPU-only llama.cpp from a previous run; when present, that build MUST be reused (no redundant download or build).
- **FR-003**: When no existing CPU-only llama.cpp is available, the system MUST download and build the OpenBLAS version and install it in the same way as other Nix derivations/flake packages so that llama.cpp is available in the environment.
- **FR-004**: The development environment MUST provide llama.cpp for CPU only (no CUDA dependency) when the environment is entered; the CPU-only build MUST be the default (or only) variant so that behavior is consistent.
- **FR-005**: Developers MUST be able to use llama.cpp from within the development environment (e.g. executable on PATH or equivalent) without installing CUDA or GPU drivers on the host.
- **FR-006**: When the environment cannot provide llama.cpp (e.g. build or fetch failure), the system MUST fail visibly or document the failure mode so that developers can diagnose and retry or work around.

### Key Entities

- **Development environment**: The reproducible environment (e.g. Nix/devenv) that developers enter via `devenv shell` or similar; it supplies tools and dependencies and triggers provisioning automatically.
- **llama.cpp**: The tool or library that provides local LLM inference; in this feature it is provided in a CPU-only variant (OpenBLAS build when built from source), installed as a Nix derivation/flake.
- **Configuration**: The file(s) that define how the development environment is built (e.g. devenv.nix); they must express CPU-only for llama.cpp, reuse from previous run when available, and build/install the OpenBLAS variant when not.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer on a machine without CUDA can enter the development environment and run llama.cpp (e.g. run the main binary or a standard example) within 5 minutes of a successful first-time setup.
- **SC-002**: 100% of developers using the documented "enter environment" flow receive a CPU-only llama.cpp (verifiable by absence of CUDA dependency in the provided build).
- **SC-003**: No manual CUDA or GPU setup is required to use llama.cpp from the development environment; success is achievable with CPU-only host.
- **SC-004**: Build or fetch failures for llama.cpp are clearly reported (e.g. error message or failed step) so that developers can retry or seek help.
- **SC-005**: When a CPU-only llama.cpp from a previous run is present, the system reuses it on subsequent entry (no redundant download or build).

## Assumptions

- The project uses a Nix-based development environment (devenv) and the change will be made in the existing configuration (e.g. devenv.nix). Entering the environment is done via `devenv shell` or similar commands; provisioning is triggered automatically on entry.
- When a CPU-only llama.cpp from a previous run is already available, it is reused; when not, the OpenBLAS version is downloaded and built and installed as any other Nix derivation/flake.
- "CPU only" means no dependency on CUDA or GPU drivers; inference runs on CPU. The CPU-only variant used when building is the OpenBLAS version. Other backends (e.g. Metal, ROCm) are out of scope unless they are the only way to get a non-CUDA build on a given platform.
- First-time setup may involve a longer delay (download or compile) than subsequent entries; subsequent entries reuse the existing build when present.
