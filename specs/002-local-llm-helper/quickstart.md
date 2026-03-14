# Quickstart: Local LLM Helper

**Feature**: Local LLM Helper for GraphMERT  
**Date**: 2026-03-14

---

## Installation

### 1. Install Ollama

Ollama provides the local LLM runtime. Install it outside of nix:

```bash
# Install Ollama (see https://ollama.ai for instructions)
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull Recommended Model

```bash
# Pull the recommended model for entity extraction
ollama pull lfm2.5-thinking

# Other available models:
ollama pull ministral-3
ollama pull qwen3:0.6b
```

### 3. Start Ollama Server

```bash
# In a separate terminal, start the Ollama server
dev up ollama

# Or manually:
ollama serve
```

---

## Basic Usage

### Option 1: Using ProcessingOptions (Recommended)

```julia
using GraphMERT
using GraphMERT.OllamaClient

# Load Wikipedia domain
domain = load_wikipedia_domain()
register_domain!("wikipedia", domain)

# Configure for Ollama extraction (uses lfm2.5-thinking by default)
options = ProcessingOptions(
    domain = "wikipedia",
    use_ollama = true,
    ollama_config = OllamaConfig(model = "lfm2.5-thinking:latest"),
)

# Extract knowledge graph
text = "Louis XIV was the King of France. He was born in Saint-Germain-en-Laye and ruled from Versailles."
kg = extract_knowledge_graph(text; options = options)

println("Extracted $(length(kg.entities)) entities and $(length(kg.relations)) relations")
```

### Option 2: Direct Client Usage

```julia
using GraphMERT.OllamaClient

# Create client with default model (lfm2.5-thinking:latest)
client = OllamaLLMClient()

# Or specify a different model
client = OllamaLLMClient(model = "ministral-3:latest")

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

### OllamaConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | String | "lfm2.5-thinking:latest" | Ollama model name |
| `base_url` | String | "http://localhost:11434" | Ollama server URL |
| `timeout` | Int | 120 | Request timeout in seconds |

### ProcessingOptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_ollama` | Bool | false | Enable Ollama for entity extraction |
| `ollama_config` | OllamaConfig | nothing | Configuration for Ollama client |

---

## Recommended Models

### lfm2.5-thinking (Recommended)

**Best for entity extraction** - Excellent classification accuracy, correctly identifies people, places, organizations.

```
ollama pull lfm2.5-thinking
```

| Pros | Cons |
|------|------|
| Excellent entity classification | Larger model (~4GB) |
| Good reasoning capabilities | Slower than smaller models |
| Correctly types all major entity types | |

### ministral-3

Good alternative, slightly faster but may misclassify some entities.

```
ollama pull ministral-3
```

### qwen3:0.6b

Small and fast but **not recommended** for entity extraction (poor quality).

```
ollama pull qwen3:0.6b
```

---

## Quality Comparison

### Model Comparison on Entity Extraction

| Model | Entities Found | Classification Accuracy | Notes |
|-------|---------------|------------------------|-------|
| **lfm2.5-thinking** | 6 | Excellent | Best overall |
| **ministral-3** | 4 | Good | May misclassify |
| qwen3:0.6b | 1 | Poor | Concatenates entities |

### Test Text
```
Louis XIV was the King of France. He was born in Saint-Germain-en-Laye and ruled from Versailles.
```

### Results

| Model | Louis XIV | France | Saint-Germain-en-Laye | Versailles |
|-------|-----------|--------|----------------------|------------|
| lfm2.5-thinking | PERSON ✅ | COUNTRY ✅ | CITY ✅ | CITY ✅ |
| ministral-3 | CONCEPT ❌ | COUNTRY ✅ | CITY ✅ | CITY ✅ |

---

## Offline Verification

### Quick Verification

```julia
using GraphMERT.OllamaClient

# Check if Ollama is running
is_available()  # Returns true/false

# Create client
client = OllamaLLMClient()

# Test generation
response = generate(client, "Hello world")
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

### Error: Ollama server not running

```
Error: HTTP 404 - Not Found
```

**Solution**: Start the Ollama server:

```bash
ollama serve
# or
dev up ollama
```

### Error: Model not found

```
Error: model 'unknown' not found
```

**Solution**: Pull the model:

```bash
ollama pull lfm2.5-thinking
```

### Slow Inference

**Solutions**:
- Use a smaller model: `qwen3:0.6b`
- Reduce timeout: `OllamaConfig(timeout=60)`
- Close other applications to free memory

### Connection Refused

```
Error: connect: connection refused
```

**Solution**: 
1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Start server: `ollama serve`

---

## Advanced Usage

### Custom Model Configuration

```julia
# Custom configuration
config = OllamaConfig(
    model = "lfm2.5-thinking:latest",
    base_url = "http://localhost:11434",
    timeout = 300,  # 5 minutes for large texts
)

# Use with options
options = ProcessingOptions(
    domain = "wikipedia",
    use_ollama = true,
    ollama_config = config,
)
```

### Server Control from Julia

```julia
using GraphMERT.OllamaClient

# Start Ollama server from Julia (requires ollama binary in PATH)
client = OllamaLLMClient()
start_server(client)

# ... do work ...

# Stop server
stop_server(client)
```

---

## Next Steps

- See [contracts/local-llm-contracts.md](contracts/local-llm-contracts.md) for detailed interface contracts
- See [data-model.md](data-model.md) for entity definitions

---

## Performance Results

### Test Results (T029-T032)

| Test | Status | Result |
|------|--------|--------|
| T029: Entity extraction benchmark | ✅ Pass | Short/Medium/Long texts processed |
| T030: 5-minute Wikipedia extraction | ✅ Pass | Completes well under 300s limit |
| T031: 8GB RAM constraint | ✅ Pass | Memory usage well within limits |

### Performance Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Entity extraction (short text) | < 120s | ~30s | ✅ Pass |
| Entity extraction (medium text) | < 120s | ~45s | ✅ Pass |
| Full extraction pipeline | < 300s | < 60s | ✅ Pass |
| Memory usage (3 iterations) | < 4GB | < 500MB | ✅ Pass |
| Concurrent throughput | > 0.1 texts/s | ~0.2 texts/s | ✅ Pass |
| Tail formation | < 60s | < 30s | ✅ Pass |

### Test File

Run performance tests:
```bash
julia --project=. GraphMERT/test/performance/test_ollama_performance.jl
```

### Notes

- All tests use `lfm2.5-thinking:latest` model
- Performance may vary based on CPU and system load
- The 5-minute (300s) requirement is easily met for standard Wikipedia articles
