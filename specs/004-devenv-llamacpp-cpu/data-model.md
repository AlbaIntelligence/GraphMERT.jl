# Data Model: CPU-only llama.cpp in Development Environment (004-devenv-llamacpp-cpu)

**Status**: Design for implementation  
**Last Updated**: 2026-03-15  
**Feature**: [spec.md](spec.md)

---

## Overview

This feature is about development-environment configuration, not application data. The "model" here describes the entities and lifecycle that the implementation must respect: the development environment, the llama.cpp package (CPU-only), configuration, and the provisioning state.

---

## 1. Development environment

The reproducible environment (Nix/devenv) that developers enter via `devenv shell` or similar.

| Concept | Description |
|--------|-------------|
| Entry | Command to enter: `devenv shell` (or equivalent). Entry triggers evaluation of Nix config and provisioning of packages. |
| Packages | Set of tools/dependencies provided in the shell (e.g. git, CPU-only llama.cpp; workflow uses llama-cpp instead of ollama). |
| Configuration | Defined in `devenv.nix` (and optionally flake or overlays). |

**Relationship**: Contains the llama.cpp package when provisioning succeeds. Configuration determines how llama.cpp is built and included.

---

## 2. llama.cpp package (CPU-only variant)

The package that provides the llama.cpp binary/library for local LLM inference, built for CPU only (OpenBLAS, no CUDA).

| Concept | Description |
|--------|-------------|
| Variant | OpenBLAS build when built from source; no CUDA dependency. |
| Availability | On PATH (or equivalent) inside the development environment shell after successful provisioning. |
| Source | Nix derivation or flake; either override of Nixpkgs `llama-cpp` or custom derivation. |
| Reuse | When the same derivation was built in a previous run, Nix store provides the same output (no redundant build). |

**Relationship**: Provided by the development environment. Built and installed as a Nix derivation/flake (FR-003).

---

## 3. Configuration

The file(s) that define how the development environment is built.

| Concept | Description |
|--------|-------------|
| Primary file | `devenv.nix` at repo root. |
| Change | Add or override package so that llama.cpp is CPU-only (OpenBLAS, no CUDA) and included in `packages`. |
| Reuse behavior | Same derivation input → same store path → reuse on subsequent entry. |

**Relationship**: Configuration drives what the development environment provides; it must express CPU-only and reuse (via Nix semantics).

---

## 4. Provisioning state (lifecycle)

| State | Description | Transition |
|-------|-------------|------------|
| Not present | No CPU-only llama.cpp in the environment (e.g. first run or after store cleanup). | User runs `devenv shell` → Nix evaluates config → build/fetch runs. |
| Building/fetching | Nix is building or fetching the derivation. | Success → Present; Failure → Error (visible). |
| Present | CPU-only llama.cpp is available in the shell (on PATH). | User exits shell; next entry reuses (same derivation) → remains Present. |
| Error | Build or fetch failed. | User sees error; retry (e.g. network) or fix config. |

No persistent database; state is reflected by Nix store and whether the derivation is already built.

---

## 5. Validation rules (from requirements)

- **FR-001**: Provisioning MUST be triggered on entry (`devenv shell` or similar).
- **FR-002**: When an existing CPU-only build (same derivation) is in the store, it MUST be reused.
- **FR-003**: When not available, system MUST build OpenBLAS version and install as Nix derivation/flake.
- **FR-006**: On failure, error MUST be visible (Nix/devenv build failure or clear message).
