# Setup notes: 004-devenv-llamacpp-cpu

**T001 verification** (2026-03-15)

- **devenv.nix**: Present at repository root.
- **Layout**: `packages = (with pkgs; [ ... ]) ++ (with llmsPkgs; [ ... ]);` — first list is Nixpkgs, second is `inputs.llms.packages.${system}`.
- **llama-cpp**: Commented at line 31: `# llama-cpp  # Disabled: pulls in CUDA by default; enable only if you set up CUDA or use override`. Ready for CPU-only override and add to `packages`.
