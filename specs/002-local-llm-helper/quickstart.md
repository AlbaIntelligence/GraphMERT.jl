# Quickstart: Local LLM Helper

**Feature**: Local LLM Helper for GraphMERT  
**Date**: 2026-03-14  
**Updated**: 2026-03-15 (llama-cpp / GGUF only; Ollama removed)

---

## Installation

### 1. Obtain a GGUF Model (Helper LLM only)

The local LLM is a **helper** for entity/relation extraction only. It is **not** the RoBERTa encoder from the paper; models in `~/.cache/llama-cpp/models` are **not** RoBERTa and do not match the paper’s methodology. The paper’s encoder (RoBERTa + H-GAT) is a separate checkpoint loaded via `load_model(path)` — see `reports/REFERENCE_SOURCES_AND_ENCODER.md` and `reports/PROJECT_STATUS.md`.

Local helper inference uses **llama-cpp** (via LlamaCpp.jl) with GGUF-format models. Models in `~/.ollama/models` are not GGUF and cannot be used; download GGUF files instead.

- **Quick download**: run `GraphMERT/scripts/download_gguf_models.sh` to fetch TinyLlama (or another preset) into `~/.cache/llama-cpp/models`.
- **Docs**: see [GGUF models (helper LLM only)](../../GraphMERT/docs/src/getting_started/gguf_models.md) for the two model roles, presets, and Ollama→GGUF equivalents.

Place the `.gguf` file on disk, e.g. `~/.cache/llama-cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`.

### 2. Optional: llama-server (OpenAI-compatible)

For server-based inference you can run **llama-server** (from llama.cpp) on port 8080 and use an OpenAI-compatible client. The primary supported path in this project is **in-process** GGUF via `LocalLLMConfig` and `use_local`.

---

## Basic Usage

### Option 1: Using ProcessingOptions (Recommended)

```julia
using GraphMERT
using GraphMERT.LocalLLM

# Load Wikipedia domain
domain = load_wikipedia_domain()
register_domain!("wikipedia", domain)

# Configure for local GGUF extraction
options = ProcessingOptions(
    domain = "wikipedia",
    use_local = true,
    local_config = LocalLLMConfig(model_path = "path/to/model.gguf"),
)

# Extract knowledge graph
text = "Louis XIV was the King of France. He was born in Saint-Germain-en-Laye and ruled from Versailles."
kg = extract_knowledge_graph(text; options = options)

println("Extracted $(length(kg.entities)) entities and $(length(kg.relations)) relations")
```

### Option 2: Direct Client Usage

```julia
using GraphMERT.LocalLLM

config = LocalLLMConfig(model_path = "path/to/model.gguf")
client = LocalLLMClient(config)

# Entity discovery
text = "Louis XIV was King of France."
entities = discover_entities(client, text, "wikipedia")

# Relation matching
relations = match_relations(client, entities, text)

# Tail formation
tails = form_tail_from_tokens(client, ["Paris", "France"], "The capital is Paris")
```

---

## Configuration Options

### LocalLLMConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | String | (required) | Path to GGUF model file |
| `context_length` | Int | 2048 | Context window size |
| `threads` | Int | 4 | CPU threads for inference |
| `temperature` | Float64 | 0.7 | Sampling temperature |
| `max_tokens` | Int | 512 | Max tokens to generate |
| `n_gpu_layers` | Int | 0 | GPU layers (0 = CPU only) |

### ProcessingOptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_local` | Bool | false | Use local LLM (GGUF) for entity extraction |
| `local_config` | LocalLLMConfig | nothing | Required when `use_local` is true |

---

## Recommended Models

- **TinyLlama 1.1B**: Small, fast, CPU-friendly (~700MB). Good for testing.
- **Larger GGUF models** (e.g. 3B–7B): Better entity/relation quality; ensure sufficient RAM.

Download GGUF models from Hugging Face or other sources and point `model_path` to the file.

---

## Offline Verification

### Quick Check

Ensure `model_path` exists and is readable. No separate server process is required for in-process inference.

```julia
using GraphMERT.LocalLLM

config = LocalLLMConfig(model_path = "path/to/model.gguf")
client = LocalLLMClient(config)
# Use client with extract_knowledge_graph or discover_entities, etc.
```

### Run Verification Tests

```bash
# Network verification test
julia --project=. GraphMERT/test/local/test_network_verification.jl

# Batch offline test
julia --project=. GraphMERT/test/local/test_batch_offline.jl
```

---

## Troubleshooting

### Error: model_path not found

**Solution**: Ensure `model_path` points to an existing `.gguf` file. Download a model if needed.

### Slow Inference

**Solutions**:
- Use a smaller GGUF model (e.g. 1.1B parameters).
- Increase `threads` in `LocalLLMConfig` (e.g. 8).
- Close other applications to free memory.

### Out of Memory

**Solution**: Use a smaller model or reduce `context_length` in `LocalLLMConfig`.

---

## Advanced Usage

### Custom LocalLLMConfig

```julia
config = LocalLLMConfig(
    model_path = "path/to/model.gguf",
    context_length = 4096,
    threads = 8,
    temperature = 0.5,
    max_tokens = 256,
)

options = ProcessingOptions(
    domain = "wikipedia",
    use_local = true,
    local_config = config,
)
```

---

## Next Steps

- See [contracts/local-llm-contracts.md](contracts/local-llm-contracts.md) for interface contracts
- See [data-model.md](data-model.md) for entity definitions

---

## Performance

Run performance tests (when available) to verify throughput and memory usage. Targets: entity extraction on a 5,000-word article in under 5 minutes, memory under 4GB for typical models.
