# Research: CPU-only llama.cpp in Development Environment (004-devenv-llamacpp-cpu)

**Status**: Complete  
**Last Updated**: 2026-03-15  
**Feature**: [spec.md](spec.md)

---

## 1. When provisioning runs (trigger)

**Decision**: Provisioning is triggered automatically when the developer runs `devenv shell` (or equivalent entry command). In devenv, packages listed in `packages` are built and made available when the shell is entered; no separate "provision" step is required.

**Rationale**: Spec FR-001 and clarifications require automatic trigger on entry. Devenv's model is that entering the shell evaluates the Nix config and builds/installs anything in `packages`; adding llama.cpp to `packages` satisfies the trigger requirement.

**Alternatives considered**:
- **Manual provision script**: Rejected; spec requires automatic trigger on entry.
- **Separate task/command before shell**: Rejected; would add an extra step.

---

## 2. Reuse of existing CPU-only build

**Decision**: The system checks for an existing CPU-only llama.cpp from a previous run; when present, that build is reused. In Nix, this is achieved by using the same derivation: if the derivation is already in the store (from a previous `devenv shell` or build), Nix will reuse it and not rebuild. No separate "check" step is needed beyond Nix's normal store lookup.

**Rationale**: Spec FR-002 and SC-005 require reuse to avoid redundant download/build. Nix's content-addressed store naturally provides this: the same derivation hash means the same output is reused.

**Alternatives considered**:
- **Explicit "check and skip" script**: Redundant; Nix already does this when evaluating the same derivation.
- **Always rebuild**: Rejected; spec requires reuse when available.

---

## 3. OpenBLAS variant and installation as Nix derivation

**Decision**: When no existing build is available, the system downloads and builds the **OpenBLAS** version of llama.cpp and installs it like any other Nix derivation/flake package. Implementation options: (a) use Nixpkgs `llama-cpp` with an override to disable CUDA and enable OpenBLAS, or (b) use a flake/overlay that provides a CPU-only OpenBLAS build. The result must be exposed in the devenv `packages` list so it appears on PATH (or equivalent) in the shell.

**Rationale**: Spec FR-003 and clarifications explicitly require the OpenBLAS version and installation as a Nix derivation/flake. OpenBLAS is a common CPU-only BLAS backend for llama.cpp and avoids CUDA.

**Alternatives considered**:
- **Generic "CPU" build without naming BLAS**: Spec and user clarification name OpenBLAS; we keep it.
- **Prebuilt binary instead of Nix build**: Rejected for reproducibility; Nix build is reproducible and matches "download and build" in the spec.
- **CUDA-enabled package with runtime flag**: Rejected; spec requires CPU-only, no CUDA dependency.

---

## 4. Integration with existing devenv.nix

**Decision**: Modify the existing `devenv.nix` at repo root. Add (or uncomment and override) a package entry for CPU-only llama.cpp so that it is built with OpenBLAS and no CUDA, and include it in the `packages` list. Preserve existing layout (e.g. `packages = (with pkgs; [ ... ]) ++ ...`).

**Rationale**: Spec and assumptions state the change is made in the existing configuration (devenv.nix). The current file already has a commented `# llama-cpp` with a note about CUDA; the implementation will add an override or alternative that is CPU-only.

**Alternatives considered**:
- **Separate file included from devenv.nix**: Acceptable if the main change is still in devenv.nix (e.g. `import ./nix/llama-cpp-cpu.nix`).
- **Different config file**: Rejected; spec says devenv.nix.

---

## 5. Verification and failure visibility

**Decision**: Verification: after `devenv shell`, the user (or enterTest) can run the llama.cpp binary (e.g. `llama-cli --help` or the main executable name from the package) and confirm it runs without CUDA. Failure visibility: if the Nix build or fetch fails, Nix/devenv will fail the shell entry or build step with an error message; document this in quickstart so developers can retry or check network.

**Rationale**: Spec FR-006 and SC-004 require visible failure; SC-001–SC-005 define measurable outcomes. Nix already fails loudly on build/fetch errors; we document the failure mode.

**Alternatives considered**: None; requirement is clear.
