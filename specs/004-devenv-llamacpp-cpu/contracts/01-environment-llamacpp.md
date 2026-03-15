# Environment Contract: CPU-only llama.cpp (004-devenv-llamacpp-cpu)

## Overview

This contract defines what the development environment guarantees to the developer after this feature is implemented: how to enter the environment, that llama.cpp is available and CPU-only, and how reuse and failures behave.

**Status**: Design complete, ready for implementation  
**Last Updated**: 2026-03-15  
**Feature**: [spec.md](../spec.md)

---

## 1. Entry and provisioning

### 1.1 Trigger

**Requirement**: Provisioning of llama.cpp MUST be triggered automatically when the developer enters the development environment (FR-001).

**Contract**:
- **Entry command**: `devenv shell` (or the project's documented equivalent to enter the dev environment).
- **Behavior**: Running the entry command causes the environment to be evaluated; packages (including CPU-only llama.cpp) are built or fetched as needed and made available in the shell.
- **No separate step**: The developer does not need to run a separate "install llama.cpp" or "provision" command before or after entry.

---

## 2. Availability and variant

### 2.1 CPU-only llama.cpp on PATH

**Requirement**: The development environment MUST provide llama.cpp for CPU only (no CUDA dependency) and developers MUST be able to use it (e.g. on PATH) (FR-004, FR-005).

**Contract**:
- **Availability**: Inside the shell started by `devenv shell`, the llama.cpp binary (or main executable from the package) is available on PATH (or via the same mechanism as other packages in the environment).
- **Variant**: The provided build is CPU-only (OpenBLAS, no CUDA). No GPU or CUDA drivers are required on the host.
- **Verification**: The user can run the binary (e.g. `llama-cli --help` or the actual executable name) and confirm it runs; optional: confirm that it reports CPU backend or does not load CUDA.

---

## 3. Reuse

### 3.1 Reuse of existing build

**Requirement**: The system MUST check for an existing CPU-only llama.cpp from a previous run and reuse it when present (FR-002, SC-005).

**Contract**:
- **Behavior**: If the same derivation was already built in a previous run (same Nix store), entering the environment again reuses that build (no redundant download or build).
- **Observable**: A second (or later) run of `devenv shell` after a successful first run should not trigger a full rebuild of llama.cpp, assuming the config and inputs are unchanged.

---

## 4. When not available: build and install

### 4.1 Build and install as Nix derivation

**Requirement**: When no existing CPU-only llama.cpp is available, the system MUST download and build the OpenBLAS version and install it like other Nix derivations/flake packages (FR-003).

**Contract**:
- **Build**: The OpenBLAS variant of llama.cpp is built (or provided by a Nix derivation/flake that builds it).
- **Install**: The result is installed in the same way as other packages in the environment (e.g. part of the Nix store and exposed in the shell).
- **Reproducibility**: The build is deterministic and reproducible (same Nix inputs → same output).

---

## 5. Failure visibility

### 5.1 Build or fetch failure

**Requirement**: When the environment cannot provide llama.cpp (e.g. build or fetch failure), the system MUST fail visibly (FR-006, SC-004).

**Contract**:
- **Build/fetch failure**: If the Nix build or source fetch fails, the failure is visible (e.g. `devenv shell` fails or shows a Nix build error). No silent fallback to a broken or half-installed state.
- **Documentation**: Quickstart or project docs describe that network or build failures can occur and how to retry (e.g. check network, rerun, or clear cache and retry).

---

## 6. Summary table

| Requirement | Contract |
|-------------|----------|
| Trigger on entry | `devenv shell` (or equivalent) triggers provisioning; no separate command. |
| CPU-only on PATH | llama.cpp binary available in shell; OpenBLAS build, no CUDA. |
| Reuse | Same derivation → Nix reuses store output on subsequent entry. |
| Build when not available | OpenBLAS version built and installed as Nix derivation/flake. |
| Failure | Build/fetch failure causes visible error; documented for retry. |
