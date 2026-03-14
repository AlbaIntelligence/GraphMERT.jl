# GraphMERT.jl Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-10

## Active Technologies
- Julia 1.10+ + LlamaCpp.jl (Julia wrapper around llama.cpp), TinyLlama 1.1B GGUF model (002-local-llm-helper)
- Local filesystem for model files (~700MB), optional cache for responses (002-local-llm-helper)
- Julia 1.10+ + Ollama (HTTP API), HTTP.jl, JSON.jl, GraphMERT.jl (002-local-llm-helper)
- Local filesystem (model files in ~/.ollama/models) (002-local-llm-helper)

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
- 002-local-llm-helper: Added Julia 1.10+ + Ollama (HTTP API), HTTP.jl, JSON.jl, GraphMERT.jl
- 002-local-llm-helper: Added Julia 1.10+ + LlamaCpp.jl (Julia wrapper around llama.cpp), TinyLlama 1.1B GGUF model

- 001-wikipedia-kg-testing: Added Julia 1.8+ + GraphMERT.jl (core), Flux.jl (ML), Wikipedia domain module

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
