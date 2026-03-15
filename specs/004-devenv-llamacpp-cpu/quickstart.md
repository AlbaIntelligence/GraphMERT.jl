# Quickstart: CPU-only llama.cpp in Development Environment (004-devenv-llamacpp-cpu)

**Purpose**: Get llama.cpp (CPU-only) from the development environment and verify it works.  
**Last Updated**: 2026-03-15  
**Feature**: [spec.md](spec.md)

---

## Prerequisites

- Nix (with flakes and/or devenv) installed and working.
- Project checked out; `devenv.nix` includes the CPU-only llama.cpp configuration (after this feature is implemented).

---

## 1. Enter the development environment

Run the usual entry command so that the environment (and llama.cpp) is provisioned automatically:

```bash
devenv shell
```

- **First time**: Nix may build or fetch the OpenBLAS version of llama.cpp; this can take several minutes. When it finishes, you are in the shell with packages available.
- **Subsequent times**: If the same build is already in the Nix store, it is reused and entry is fast.

---

## 2. Verify llama.cpp is available

Inside the shell, check that the llama.cpp binary is on PATH and runs (e.g. help or version):

```bash
# Nixpkgs may expose the main binary as llama or llama-cli
llama --help
# or
llama-cli --help
which llama || which llama-cli
```

You should see the binary run without errors. The build is CPU-only (no CUDA required).

---

## 3. Local inference server (llama-server)

The workflow uses **llama-cpp** instead of Ollama. When you use devenv processes (e.g. `devenv up` or the process manager), the **llama-server** process runs by default:

- **Endpoint**: `http://127.0.0.1:8080`
- **Binary**: `llama-server` from the same CPU-only llama-cpp package.

Load a model and use the OpenAI-compatible API on port 8080. The previous Ollama process (port 11434) has been replaced by this llama-cpp server.

---

## 4. (Optional) Confirm CPU-only

If the binary supports a flag or output that indicates backend, confirm it uses CPU (e.g. no CUDA). Otherwise, running without a GPU and without CUDA installed is sufficient to confirm CPU-only.

---

## 5. First-time vs subsequent entry (reuse)

- **First time**: Running `devenv shell` triggers a build or fetch of the CPU-only llama.cpp derivation. This can take several minutes. Once the derivation is in the Nix store, you are in the shell with llama.cpp on PATH.
- **Subsequent times**: Nix reuses the same derivation from the store (same config → same store path). No redundant download or build; entry is fast.
- **After changing `devenv.nix`**: If you change the package set, Nix may rebuild; unchanged derivations are still reused.

## 6. If something fails (failure modes)

- **Build or fetch failed**: Nix/devenv will show an error when you run `devenv shell`. Common causes: network unavailable, source fetch failed, or build dependency issue. Fix (e.g. check network, retry) and run `devenv shell` again. The failure is visible (no silent half-install).
- **Binary not on PATH**: Ensure the feature is implemented and the package is in the environment's `packages` list in `devenv.nix`. Run `devenv up` or a fresh `devenv shell` to refresh the profile after config changes.
- **Network unavailable during first-time build**: The build will fail with a fetch error. Retry when network is available, or use a Nix cache that has the derivation.

---

## 7. Where to read more

- **Spec and behavior**: [spec.md](spec.md)
- **Data model and lifecycle**: [data-model.md](data-model.md)
- **Environment contract**: [contracts/01-environment-llamacpp.md](contracts/01-environment-llamacpp.md)
- **Research decisions**: [research.md](research.md)
