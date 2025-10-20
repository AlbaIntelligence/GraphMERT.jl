# GraphMERT Core API Contract

## Overview

This document defines the core public API for GraphMERT knowledge graph extraction. These are the primary functions that users will interact with.

**Status**: Design complete, ready for implementation
**Last Updated**: 2025-01-20

---

## 1. Main Extraction API

### `extract_knowledge_graph`

Primary function for extracting knowledge graphs from text.

**Signature**:
```julia
function extract_knowledge_graph(
    text::String,
    model::GraphMERTModel;
    options::ProcessingOptions = default_processing_options()
)::KnowledgeGraph
```

**Parameters**:
- `text`: Input biomedical text (PubMed abstract, medical document, etc.)
- `model`: Trained GraphMERT model (loaded from checkpoint)
- `options`: Processing options (batch size, device, thresholds, etc.)

**Returns**:
- `KnowledgeGraph`: Complete extracted knowledge graph with entities, relations, and provenance

**Requirements** (from spec.md):
- REQ-007: Generate knowledge graph with nodes (entities) and edges (relations)
- REQ-008: Provide confidence scores for entities and relations
- REQ-017: Simple API interface
- REQ-020: Support biomedical entity and relation types

**Example Usage**:
```julia
using GraphMERT

# Load trained model
model = load_model("path/to/checkpoint.jld2")

# Extract KG from text
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
hyperglycemia. Metformin is commonly used to treat type 2 diabetes.
"""

kg = extract_knowledge_graph(text, model)

# Access results
println("Extracted $(length(kg.entities)) entities")
println("Extracted $(length(kg.relations)) relations")

for entity in kg.entities
    println("  $(entity.text) [$(entity.entity_type)] (conf: $(entity.confidence))")
end
```

**Error Conditions**:
- `ArgumentError`: Invalid text (empty, too long > max_length)
- `ModelError`: Model not properly loaded or initialized
- `ProcessingError`: Runtime error during extraction

**Performance**:
- NFR-001: Process 5,000 tokens per second on laptop hardware
- NFR-002: Memory usage < 4GB for inputs up to 100K tokens
- NFR-003: Linear scaling with input size

---

## 2. Batch Processing API

### `extract_knowledge_graph_batch`

Process multiple documents in batch for efficiency.

**Signature**:
```julia
function extract_knowledge_graph_batch(
    texts::Vector{String},
    model::GraphMERTModel;
    options::ProcessingOptions = default_processing_options()
)::Vector{KnowledgeGraph}
```

**Parameters**:
- `texts`: Vector of input texts
- `model`: Trained GraphMERT model
- `options`: Processing options

**Returns**:
- `Vector{KnowledgeGraph}`: One KG per input text

**Requirements**:
- REQ-018: Support batch processing for multiple documents
- NFR-003: Scale linearly with input size

**Example Usage**:
```julia
# Process multiple PubMed abstracts
abstracts = load_pubmed_abstracts("diabetes_dataset.json")
kgs = extract_knowledge_graph_batch(abstracts, model,
    ProcessingOptions(batch_size=32, device=:cuda))

# Merge all KGs
merged_kg = merge_knowledge_graphs(kgs)
```

**Performance**:
- Batching improves throughput by ~3x compared to sequential processing
- Automatic batch size optimization based on available memory

---

## 3. Model Loading/Saving API

### `load_model`

Load a trained GraphMERT model from checkpoint.

**Signature**:
```julia
function load_model(
    checkpoint_path::String;
    device::Symbol = :cpu,
    strict::Bool = true
)::GraphMERTModel
```

**Parameters**:
- `checkpoint_path`: Path to model checkpoint (.jld2 or BSON format)
- `device`: Target device (`:cpu` or `:cuda`)
- `strict`: Whether to require exact parameter match

**Returns**:
- `GraphMERTModel`: Loaded model ready for inference

**Example**:
```julia
model = load_model("checkpoints/graphmert_diabetes_epoch10.jld2", device=:cuda)
```

---

### `save_model`

Save a trained model to checkpoint.

**Signature**:
```julia
function save_model(
    model::GraphMERTModel,
    checkpoint_path::String;
    include_config::Bool = true,
    include_optimizer_state::Bool = false
)::Nothing
```

**Parameters**:
- `model`: Model to save
- `checkpoint_path`: Output path
- `include_config`: Whether to save configuration
- `include_optimizer_state`: Whether to save optimizer state (for training resumption)

**Example**:
```julia
save_model(model, "checkpoints/my_model.jld2")
```

---

## 4. Configuration API

### `default_processing_options`

Create default processing options.

**Signature**:
```julia
function default_processing_options(;
    batch_size::Int = 32,
    max_length::Int = 1024,
    device::Symbol = :cpu,
    use_amp::Bool = false,
    num_workers::Int = 1,
    seed::Union{Int, Nothing} = nothing,
    top_k_predictions::Int = 20,
    similarity_threshold::Float64 = 0.8,
    enable_provenance_tracking::Bool = true
)::ProcessingOptions
```

**Requirements**:
- REQ-019: Provide configuration options for model parameters
- NFR-009: Intuitive API for Julia users

---

### `default_graphmert_config`

Create default GraphMERT model configuration.

**Signature**:
```julia
function default_graphmert_config(;
    vocab_size::Int = 30522,
    hidden_size::Int = 512,
    num_hidden_layers::Int = 12,
    num_attention_heads::Int = 8,
    # ... other parameters with defaults
)::GraphMERTConfig
```

**Example**:
```julia
# Create custom config
config = default_graphmert_config(
    hidden_size=768,  # Increase model size
    num_hidden_layers=16
)

# Initialize model with config
model = GraphMERTModel(config)
```

---

## 5. Evaluation API

### `evaluate_factscore`

Evaluate knowledge graph using FActScore* metric.

**Signature**:
```julia
function evaluate_factscore(
    kg::KnowledgeGraph,
    source_text::String;
    llm_client::Union{LLMClient, Nothing} = nothing
)::FActScoreResult
```

**Parameters**:
- `kg`: Knowledge graph to evaluate
- `source_text`: Original source text (for context retrieval)
- `llm_client`: Helper LLM client for validation (optional)

**Returns**:
- `FActScoreResult`: Factuality scores per triple and overall

**Requirements**:
- REQ-011: Achieve FActScore within 5% of paper results (69.8%)
- REQ-016: Implement GraphRAG evaluation methodology

**Example**:
```julia
result = evaluate_factscore(kg, original_text, llm_client)
println("FActScore: $(result.overall_score)")
println("Supported: $(result.num_supported) / $(result.num_total)")
```

---

### `evaluate_validity`

Evaluate ontological validity of knowledge graph.

**Signature**:
```julia
function evaluate_validity(
    kg::KnowledgeGraph;
    llm_client::Union{LLMClient, Nothing} = nothing,
    umls_client::Union{UMLSClient, Nothing} = nothing
)::ValidityScoreResult
```

**Requirements**:
- REQ-012: Achieve ValidityScore within 5% of paper results (68.8%)

---

### `evaluate_graphrag`

Evaluate knowledge graph using GraphRAG QA benchmark.

**Signature**:
```julia
function evaluate_graphrag(
    kg::KnowledgeGraph,
    questions::Vector{String},
    ground_truth::Vector{String};
    llm_client::LLMClient
)::GraphRAGResult
```

---

## 6. Training API

### `train_graphmert`

Train a GraphMERT model from scratch.

**Signature**:
```julia
function train_graphmert(
    train_data::Vector{String},
    config::GraphMERTConfig;
    seed_kg::Union{Vector{SemanticTriple}, Nothing} = nothing,
    mlm_config::MLMConfig = default_mlm_config(),
    mnm_config::MNMConfig = default_mnm_config(),
    injection_config::Union{SeedInjectionConfig, Nothing} = nothing,
    num_epochs::Int = 10,
    checkpoint_dir::String = "checkpoints",
    validation_data::Union{Vector{String}, Nothing} = nothing
)::GraphMERTModel
```

**Parameters**:
- `train_data`: Training text corpus
- `config`: Model configuration
- `seed_kg`: Optional seed knowledge graph for injection
- `mlm_config`: MLM training configuration
- `mnm_config`: MNM training configuration
- `injection_config`: Seed injection configuration
- `num_epochs`: Number of training epochs
- `checkpoint_dir`: Directory for saving checkpoints
- `validation_data`: Optional validation set

**Returns**:
- `GraphMERTModel`: Trained model

**Requirements**:
- REQ-004: Implement seed KG injection for training
- REQ-005: Implement MLM + MNM training objectives
- REQ-013: Support datasets up to 124.7M tokens
- REQ-015: Generate reproducible results with random seeds

**Example**:
```julia
# Load training data
train_texts = load_pubmed_abstracts("diabetes_train.json")

# Load seed KG
seed_triples = load_umls_triples("seed_kg.json")

# Train model
model = train_graphmert(
    train_texts,
    default_graphmert_config(),
    seed_kg=seed_triples,
    injection_config=default_seed_injection_config(),
    num_epochs=10,
    checkpoint_dir="checkpoints/diabetes"
)
```

**Callbacks**:
```julia
# Custom training callback
function training_callback(epoch, step, loss, metrics)
    println("Epoch $epoch, Step $step, Loss: $loss")
    if step % 100 == 0
        save_checkpoint(model, "checkpoint_epoch$(epoch)_step$(step).jld2")
    end
end

model = train_graphmert(train_data, config, callback=training_callback)
```

---

## 7. Helper LLM Integration API

### `create_llm_client`

Create a helper LLM client for entity discovery and relation matching.

**Signature**:
```julia
function create_llm_client(
    provider::Symbol;  # :openai, :anthropic, :local, etc.
    api_key::Union{String, Nothing} = nothing,
    model_name::String = "gpt-4",
    cache_enabled::Bool = true,
    cache_dir::String = ".cache/llm"
)::LLMClient
```

**Requirements**:
- REQ-006: Implement helper LLM integration
- REQ-006a: Support OpenAI GPT-4 API
- REQ-006c: Handle rate limits with queuing
- REQ-006f: Cache responses to reduce costs

**Example**:
```julia
# OpenAI client
llm = create_llm_client(:openai, api_key=ENV["OPENAI_API_KEY"])

# Local client (Llama 3)
llm = create_llm_client(:local, model_name="llama3-70b-instruct")
```

---

## 8. UMLS Integration API

### `create_umls_client`

Create UMLS client for entity linking and validation.

**Signature**:
```julia
function create_umls_client(
    api_key::String;
    cache_enabled::Bool = true,
    cache_dir::String = ".cache/umls",
    rate_limit::Int = 100  # requests per minute
)::UMLSClient
```

**Requirements**:
- REQ-003: Integrate UMLS biomedical knowledge base
- REQ-003a: Connect to UMLS REST API with authentication
- REQ-003b: Support rate limiting and retry logic
- REQ-003d: Cache mappings locally

**Example**:
```julia
umls = create_umls_client(ENV["UMLS_API_KEY"])

# Link entity to UMLS
result = link_entity(umls, "diabetes mellitus")
println("CUI: $(result.cui)")
println("Preferred name: $(result.preferred_name)")
```

---

## 9. Utility APIs

### `merge_knowledge_graphs`

Merge multiple knowledge graphs into one.

**Signature**:
```julia
function merge_knowledge_graphs(
    graphs::Vector{KnowledgeGraph};
    deduplicate::Bool = true,
    confidence_aggregation::Symbol = :max  # :max, :mean, :weighted
)::KnowledgeGraph
```

---

### `filter_knowledge_graph`

Filter knowledge graph by confidence thresholds.

**Signature**:
```julia
function filter_knowledge_graph(
    kg::KnowledgeGraph;
    min_entity_confidence::Float64 = 0.5,
    min_relation_confidence::Float64 = 0.5,
    entity_types::Union{Vector{String}, Nothing} = nothing,
    relation_types::Union{Vector{String}, Nothing} = nothing
)::KnowledgeGraph
```

---

### `export_knowledge_graph`

Export knowledge graph to various formats.

**Signature**:
```julia
function export_knowledge_graph(
    kg::KnowledgeGraph,
    output_path::String;
    format::Symbol = :json  # :json, :jsonld, :rdf, :neo4j
)::Nothing
```

**Example**:
```julia
# Export to JSON
export_knowledge_graph(kg, "output.json", format=:json)

# Export to RDF
export_knowledge_graph(kg, "output.rdf", format=:rdf)

# Export to Neo4j Cypher
export_knowledge_graph(kg, "output.cypher", format=:neo4j)
```

---

## API Design Principles

### 1. **Simplicity** (REQ-017, NFR-009)
- Primary use case (extraction) requires only 3 lines of code
- Sensible defaults for all parameters
- Progressive disclosure of complexity

### 2. **Type Safety** (REQ-025, NFR-015)
- Leverage Julia's type system for compile-time checks
- Clear type signatures for all functions
- Validation at API boundaries

### 3. **Extensibility** (REQ-024, NFR-016)
- Use multiple dispatch for customization
- Provide hooks for callbacks and customization
- Abstract interfaces for LLM/UMLS clients

### 4. **Error Handling** (NFR-005, NFR-011)
- Descriptive error messages
- Graceful degradation when possible
- Clear documentation of failure modes

### 5. **Performance** (NFR-001-003)
- Batch processing by default
- Memory-efficient implementations
- Async/parallel processing where appropriate

---

## Error Handling Contract

All public APIs must handle errors consistently:

### Error Types

```julia
# Base error type
abstract type GraphMERTError <: Exception end

# Specific errors
struct ModelLoadError <: GraphMERTError
    message::String
    checkpoint_path::String
end

struct ExtractionError <: GraphMERTError
    message::String
    text_snippet::String
    cause::Union{Exception, Nothing}
end

struct ValidationError <: GraphMERTError
    message::String
    invalid_field::String
    invalid_value::Any
end

struct APIError <: GraphMERTError  # UMLS, LLM API errors
    message::String
    service::String
    http_status::Union{Int, Nothing}
end
```

### Error Handling Guidelines

1. **Validate inputs early**: Check at function entry
2. **Provide context**: Include relevant data in error messages
3. **Fail gracefully**: Return partial results when possible
4. **Log errors**: Use logging framework for debugging
5. **Retry transient failures**: Especially for API calls

---

## API Versioning

Version: **1.0.0** (initial release)

**Semantic Versioning**:
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Deprecation Policy**:
- Deprecated functions remain for at least one MINOR version
- Clear warnings with migration guide
- Documentation of deprecated APIs

---

## Testing Contract

Each API function must have:

1. **Unit tests**: Test function in isolation
2. **Integration tests**: Test with real models/data
3. **Example code**: Runnable examples in docstrings
4. **Performance tests**: Verify NFR requirements

**Example Test Structure**:

```julia
@testset "extract_knowledge_graph" begin
    @testset "basic extraction" begin
        # Test happy path
    end

    @testset "error handling" begin
        # Test invalid inputs
        @test_throws ArgumentError extract_knowledge_graph("", model)
    end

    @testset "performance" begin
        # Verify NFR-001: 5,000 tokens/sec
        @test throughput > 5000.0
    end
end
```

---

## Documentation Contract

Each public API function must have:

1. **Docstring** with:
   - Purpose and use case
   - Parameter descriptions
   - Return value description
   - At least one example
   - Error conditions
   - Related functions

2. **Type signatures**: Fully annotated

3. **Examples**: Working code snippets

4. **Performance notes**: Expected complexity, memory usage

---

## Implementation Checklist

| API Function                    | Status     | Tests | Docs | Priority |
| ------------------------------- | ---------- | ----- | ---- | -------- |
| `extract_knowledge_graph`       | 🔴 Missing  | 🔴     | 🔴    | P0       |
| `extract_knowledge_graph_batch` | 🔴 Missing  | 🔴     | 🔴    | P0       |
| `load_model` / `save_model`     | 🟡 Partial  | 🟡     | 🟡    | P0       |
| `default_processing_options`    | 🟡 Partial  | 🟡     | 🟡    | P0       |
| `default_graphmert_config`      | ✅ Complete | ✅     | ✅    | P0       |
| `evaluate_factscore`            | 🔴 Missing  | 🔴     | 🔴    | P1       |
| `evaluate_validity`             | 🔴 Missing  | 🔴     | 🔴    | P1       |
| `evaluate_graphrag`             | 🔴 Missing  | 🔴     | 🔴    | P1       |
| `train_graphmert`               | 🔴 Missing  | 🔴     | 🔴    | P0       |
| `create_llm_client`             | 🔴 Missing  | 🔴     | 🔴    | P1       |
| `create_umls_client`            | 🟡 Partial  | 🔴     | 🔴    | P1       |
| Utility functions               | 🔴 Missing  | 🔴     | 🔴    | P2       |

---

## Next Steps

1. Implement core extraction API (P0)
2. Implement training API (P0)
3. Implement evaluation APIs (P1)
4. Write comprehensive tests
5. Generate API documentation
6. Create quickstart guide

**Status**: API contract ✅ **COMPLETE** - Ready for implementation
