# GraphMERT.jl Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-10

## Active Technologies
- Julia 1.10+ + LlamaCpp.jl (llama.cpp / GGUF), local model files (002-local-llm-helper)
- Julia 1.8+ + GraphMERT.jl (core), Flux.jl (ML), Wikipedia domain module (001-wikipedia-kg-testing)

## Project Structure

```text
src/
tests/
```

## Commands

# Add commands for Julia 1.8+

## Code Style

Julia 1.8+: Follow standard conventions

## Recent Changes
- 002-local-llm-helper: Local LLM via LlamaCpp.jl + GGUF; Ollama removed in favour of llama-cpp
- 004-devenv-llamacpp-cpu: devenv workflow uses llama-cpp (llama-server on 8080)

- 001-wikipedia-kg-testing: Added Julia 1.8+ + GraphMERT.jl (core), Flux.jl (ML), Wikipedia domain module

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
