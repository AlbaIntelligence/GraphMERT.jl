# Data Model: Local LLM Helper

**Feature**: Local LLM Helper for GraphMERT  
**Date**: 2026-03-13
**Updated**: 2026-03-15 (Ollama implementation)

---

## Entities

### OllamaConfig

Configuration for Ollama HTTP API client.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | String | Yes | Ollama model name (default: "lfm2.5-thinking:latest") |
| `base_url` | String | No | Ollama server URL (default: "http://localhost:11434") |
| `timeout` | Int | No | Request timeout in seconds (default: 120) |

---

### OllamaLLMClient

Client for Ollama HTTP API.

| Field | Type | Description |
|-------|------|-------------|
| `config` | OllamaConfig | Configuration |
| `session` | HTTP.Client | HTTP session |

---

### LocalLLMConfig (legacy)

Configuration for local GGUF model (when LlamaCpp.jl is fixed).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | String | Yes | Path to GGUF model file |
| `context_length` | Int | No | Context window size (default: 2048) |
| `threads` | Int | No | CPU threads for inference (default: 4) |
| `temperature` | Float64 | No | Sampling temperature (default: 0.7) |
| `max_tokens` | Int | No | Max tokens to generate (default: 512) |
| `n_gpu_layers` | Int | No | GPU layers (default: 0, CPU-only) |

---

## Relationships

```
OllamaLLMClient
    ├── uses ──► OllamaConfig
    └── calls ──► Ollama HTTP API (http://localhost:11434)
```

---

## State Transitions

### Model Loading (Ollama)
```
Unloaded → Connecting → Ready → Running
                              → Failed (connection error)
```

### Inference
```
Ready → Processing → Ready (success)
                   → Failed (error)
```

---

## Validation Rules (OllamaConfig)

1. `model` must be a valid Ollama model name
2. `base_url` must be a valid HTTP URL
3. `timeout` must be ≥ 1 second
