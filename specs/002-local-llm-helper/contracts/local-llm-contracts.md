# Contract: Local LLM Helper

**Date**: 2026-03-13

## Contract Overview

This document defines the expected behavior of the local LLM helper for integration with GraphMERT's extraction pipeline.

## LocalLLMClient Interface Contract

### Construction

```julia
LocalLLMClient(config::LocalLLMConfig)
```

**Input**:
- `config`: LocalLLMConfig with valid model_path to GGUF file

**Expected Behavior**:
- Load model into memory
- Initialize inference context
- Return ready-to-use client

**Error Cases**:
- File not found: Throw `SystemError("Model file not found: $model_path")`
- Invalid GGUF: Throw `ArgumentError("Invalid GGUF model file")`
- Out of memory: Throw `OutOfMemoryError("Insufficient memory to load model")`

---

### discover_entities

```julia
discover_entities(client::LocalLLMClient, text::String, domain::DomainProvider)::Vector{String}
```

**Input**:
- `client`: LocalLLMClient instance
- `text`: Text to extract entities from
- `domain`: DomainProvider for prompt generation

**Expected Output**:
- Vector of entity strings extracted from text

**Contract Rules**:
1. **Non-empty input**: MUST handle empty text gracefully (return empty vector)
2. **Timeout**: MUST complete in reasonable time (<60s for 5000 words)
3. **No crashes**: MUST NOT throw unhandled exceptions
4. **Format**: Return vector of unique entity strings

---

### match_relations

```julia
match_relations(client::LocalLLMClient, entities::Vector{String}, text::String)::Dict{String, Dict{String, String}}
```

**Input**:
- `client`: LocalLLMClient instance
- `entities`: Vector of entity strings
- `text`: Original text for context

**Expected Output**:
- Dictionary mapping entity → relation dict → target entity

**Contract Rules**:
1. **Empty entities**: MUST handle empty entity list (return empty dict)
2. **Valid relations**: Output format must be `Dict{String, Dict{String, String}}`
3. **No crashes**: MUST NOT throw unhandled exceptions

---

### form_tail_from_tokens

```julia
form_tail_from_tokens(client::LocalLLMClient, tokens::Vector{Tuple{Int, Float64}}, text::String)::Vector{String}
```

**Input**:
- `client`: LocalLLMClient instance
- `tokens`: Vector of (token_id, probability) tuples
- `text`: Original text for context

**Expected Output**:
- Vector of formed tail entity strings

**Contract Rules**:
1. **Empty tokens**: MUST handle empty token list (return empty vector)
2. **Output format**: Return vector of unique strings
3. **No crashes**: MUST NOT throw unhandled exceptions

---

## ProcessingOptions Integration

The local LLM helper integrates via `ProcessingOptions`:

```julia
ProcessingOptions(
    domain::String = "wikipedia",
    use_helper_llm::Bool = true,
    use_local::Bool = false,           # NEW: Use local instead of cloud
    local_config::LocalLLMConfig = nothing,  # NEW: Local model config
)
```

**Contract Rules**:
1. **Default**: `use_local = false` preserves existing cloud behavior
2. **Conflict**: If `use_local = true` and `local_config = nothing`, throw error
3. **Backend selection**: When `use_local = true`, use `LocalLLMClient`; otherwise use `HelperLLMClient`

---

## Performance Contract

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model load time | <30s | Time to first inference |
| Inference (1000 words) | <30s | Entity extraction |
| Memory usage | <2GB | Peak RAM during inference |
| Batch (10 articles) | <5min | Total time |

---

## Offline Verification Contract

Users must be able to verify offline operation:

```julia
# Test offline - should complete without network
process_articles(articles; network_monitor = nothing)
```

**Contract Rules**:
1. **No network calls**: When `use_local = true`, no HTTP requests to external APIs
2. **Deterministic**: Same input produces same output (with temperature=0)
3. **Graceful failure**: Clear error messages when model unavailable
