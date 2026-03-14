# Data Model: Local LLM Helper

**Feature**: Local LLM Helper for GraphMERT  
**Date**: 2026-03-13

---

## Entities

### LocalLLMConfig

Configuration for local LLM inference.

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

Client for local LLM inference.

| Field | Type | Description |
|-------|------|-------------|
| `config` | LocalLLMConfig | Configuration |
| `model` | LlamaCpp.jl model handle | Loaded model instance |
| `cache` | HelperLLMCache | Response cache (optional) |

---

### ModelMetadata

Metadata about available local models.

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Model display name |
| `filename` | String | GGUF filename |
| `params` | Int | Parameter count |
| `quantization` | String | Quantization level (Q4_0, Q5_1, etc.) |
| `ram_estimate` | Int | Estimated RAM in MB |
| `context_length` | Int | Context window |

---

## Relationships

```
LocalLLMClient
    ├── uses ──► LocalLLMConfig
    ├── loads ──► GGUF Model File
    └── optionally uses ──► HelperLLMCache (existing)
```

---

## State Transitions

### Model Loading
```
Unloaded → Loading → Ready → (error) → Failed
                        ↓
                     Running
```

### Inference
```
Ready → Processing → Ready (success)
                  → Failed (error)
```

---

## Validation Rules

1. `model_path` must point to existing GGUF file
2. `context_length` must be power of 2 ≤ 8192
3. `threads` must be ≥ 1 ≤ CPU count
4. `temperature` must be ≥ 0.0 ≤ 2.0
5. `max_tokens` must be ≥ 1 ≤ 4096

---

## Integration with Existing Types

The `LocalLLMClient` must implement the same interface as `HelperLLMClient`:

| Method | Description |
|--------|-------------|
| `discover_entities(client, text, domain)` | Extract entities from text |
| `match_relations(client, entities, text)` | Find relations between entities |
| `form_tail_from_tokens(client, tokens, text)` | Form coherent tails |

This allows drop-in replacement via `ProcessingOptions(use_local=true, local_config=...)`.
