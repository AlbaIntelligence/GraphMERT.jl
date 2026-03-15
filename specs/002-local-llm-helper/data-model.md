# Data Model: Local LLM Helper

**Feature**: Local LLM Helper for GraphMERT  
**Date**: 2026-03-13  
**Updated**: 2026-03-15 (llama-cpp / GGUF only; Ollama removed)

---

## Entities

### LocalLLMConfig

Configuration for local GGUF model loading (LlamaCpp.jl / llama-cpp).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | String | Yes | Path to GGUF model file |
| `context_length` | Int | No | Context window size (default: 2048) |
| `threads` | Int | No | CPU threads for inference (default: 4) |
| `temperature` | Float64 | No | Sampling temperature (default: 0.7) |
| `max_tokens` | Int | No | Max tokens to generate (default: 512) |
| `n_gpu_layers` | Int | No | GPU layers (default: 0, CPU-only) |

---

### LocalLLMClient

Client for in-process local inference via LlamaCpp.jl.

| Field | Type | Description |
|-------|------|-------------|
| `config` | LocalLLMConfig | Configuration |

---

## Relationships

```
LocalLLMClient
    ├── uses ──► LocalLLMConfig
    └── uses ──► LlamaCpp.jl (GGUF model at model_path)
```

---

## State Transitions

### Model Loading

```
Unloaded → Loading (GGUF) → Ready → Running
                              → Failed (file missing / load error)
```

### Inference

```
Ready → Processing → Ready (success)
                   → Failed (error)
```

---

## Validation Rules (LocalLLMConfig)

1. `model_path` must point to an existing file
2. `context_length` must be between 1 and 8192
3. `threads` must be ≥ 1
4. `temperature` must be between 0.0 and 2.0
5. `max_tokens` must be between 1 and 4096
6. `n_gpu_layers` must be non-negative
