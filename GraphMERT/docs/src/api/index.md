# GraphMERT.jl API Reference

Complete API documentation for GraphMERT.jl.

## Core Functions

### `extract_knowledge_graph(text, model, options)`

Extract a knowledge graph from input text using a GraphMERT model.

**Parameters:**

- `text::Union{String, Vector{String}}` - Input text or batch of texts
- `model::Union{GraphMERTModel, String}` - Model instance or path to model
- `options::ProcessingOptions` - Processing configuration (optional, must include `domain` field)

**Returns:**

- `KnowledgeGraph` - Extracted knowledge graph

**Example:**

```julia
# Load and register domain
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Extract with domain
text = "Diabetes affects blood sugar levels."
options = ProcessingOptions(domain="biomedical")
graph = extract_knowledge_graph(text, model; options=options)
```

### `load_model(model_path, config)`

Load a GraphMERT model from file.

**Parameters:**

- `model_path::String` - Path to model file
- `config::GraphMERTConfig` - Model configuration (optional)

**Returns:**

- `GraphMERTModel` - Loaded model instance

**Example:**

```julia
model = load_model("path/to/model.onnx")
```

### `preprocess_text(text, options)`

Preprocess text for GraphMERT processing.

**Parameters:**

- `text::String` - Input text
- `options::ProcessingOptions` - Processing options (must include `domain` field)

**Returns:**

- `PreprocessedText` - Preprocessed text ready for processing

## Domain System API

See [Domain API Reference](domain.md) for complete domain system documentation.

**Key Functions:**

- `register_domain!(domain_name, provider)` - Register a domain
- `get_domain(domain_name)` - Get a domain provider
- `list_domains()` - List all registered domains
- `set_default_domain(domain_name)` - Set default domain
- `get_default_domain()` - Get default domain

**Domain Provider Methods:**

- `extract_entities(domain, text, config)` - Extract domain-specific entities
- `extract_relations(domain, entities, text, config)` - Extract domain-specific relations
- `validate_entity(domain, entity_text, entity_type, context)` - Validate entities
- `validate_relation(domain, head, relation_type, tail, context)` - Validate relations
- `calculate_entity_confidence(domain, entity_text, entity_type, context)` - Calculate confidence
- `calculate_relation_confidence(domain, head, relation_type, tail, context)` - Calculate confidence
- `link_entity(domain, entity_text, config)` - Link to knowledge base (optional)
- `create_seed_triples(domain, entity_text, config)` - Create seed triples (optional)
- `create_evaluation_metrics(domain, kg)` - Create domain metrics (optional)
- `create_prompt(domain, task_type, context)` - Generate LLM prompts (optional)

## Data Structures

For the exact field layouts, see `GraphMERT/src/types.jl`. This section gives a compact, conceptual view; the tests in `GraphMERT/test/unit/test_api.jl` and `GraphMERT/test/unit/test_extraction.jl` are the canonical API spec.

### `KnowledgeGraph`

Main output structure containing the extracted knowledge graph.

- Wraps **entities**, **relations**, a **metadata** dictionary, and a **created_at** timestamp.
- Metadata typically includes at least `domain`, `source_text`, and simple counts; callers should not rely on a fixed metadata schema.

Internally, entities and relations are stored as `KnowledgeEntity` / `KnowledgeRelation` values; helper constructors allow building a `KnowledgeGraph` from generic `Entity` / `Relation` vectors.

### `Entity` (generic, domain-agnostic)

Represents an extracted entity (works for all domains).

**Fields:**

- `id::String` - Unique identifier
- `text::String` - Original text
- `label::String` - Entity label
- `entity_type::String` - Domain-specific entity type (e.g., "DISEASE", "PERSON")
- `domain::String` - Domain identifier (e.g., "biomedical", "wikipedia")
- `attributes::Dict{String, Any}` - Domain-specific attributes (e.g., CUI, QID)
- `position::TextPosition` - Position in text
- `confidence::Float64` - Confidence score (0.0-1.0)
- `provenance::String` - Source text

### `Relation` (generic, domain-agnostic)

Represents an extracted relation (works for all domains).

**Fields:**

- `head::String` - Head entity ID
- `tail::String` - Tail entity ID
- `relation_type::String` - Domain-specific relation type (e.g., "TREATS", "BORN_IN")
- `confidence::Float64` - Confidence score (0.0-1.0)
- `attributes::Dict{String, Any}` - Domain-specific attributes
- `created_at::DateTime` - Creation timestamp

## Configuration

### `ProcessingOptions`

Configuration for text processing and extraction.

- **Required**: `domain::String` (e.g. `"biomedical"`, `"wikipedia"`).
- Core knobs:
  - `max_length::Int`, `batch_size::Int`
  - `confidence_threshold::Float64`
  - `use_umls::Bool`, `use_helper_llm::Bool`
  - `similarity_threshold::Float64`, `top_k_predictions::Int`
- Additional fields control device (`device`), AMP, workers, seeding, provenance tracking, caching, and verbosity; see `ProcessingOptions` in `types.jl` for the full list.

Example:

```julia
options = ProcessingOptions(
    domain = "biomedical",
    confidence_threshold = 0.8,
    use_umls = true,
)
```

### `GraphMERTConfig`

Configuration for building `GraphMERTModel` instances (see `models/graphmert.jl`). It bundles RoBERTa, H-GAT, and attention settings plus high-level model dimensions. For most users, loading a model via `load_model(path)` and passing `ProcessingOptions` is sufficient; advanced configurations are primarily relevant to training code and follow the spec in `original_paper/expanded_rewrite/`.

## Evaluation Functions

### `evaluate_factscore` (Updated for Domain System)

```julia
evaluate_factscore(kg::KnowledgeGraph, text::String;
                   domain_name::Union{String, Nothing}=nothing,
                   include_domain_metrics::Bool=true,
                   ...) -> FActScoreResult
```

Calculate FActScore for a knowledge graph with optional domain-specific metrics.

**Parameters:**

- `kg::KnowledgeGraph` - Knowledge graph to evaluate
- `text::String` - Original source text
- `domain_name::Union{String, Nothing}` - Optional domain name (inferred from `kg.metadata["domain"]` if not provided)
- `include_domain_metrics::Bool` - Whether to include domain-specific metrics in result

**Returns:**

- `FActScoreResult` with `metadata["domain_metrics"]` if domain metrics are included

**Example:**

```julia
result = evaluate_factscore(kg, text, domain_name="biomedical")
if haskey(result.metadata, "domain_metrics")
    domain_metrics = result.metadata["domain_metrics"]
    println("UMLS linking coverage: ", domain_metrics["umls_linking_coverage"])
end
```

### `evaluate_validity` (Updated for Domain System)

```julia
evaluate_validity(kg::KnowledgeGraph;
                  domain_name::Union{String, Nothing}=nothing,
                  include_domain_metrics::Bool=true,
                  ...) -> ValidityScoreResult
```

Calculate ValidityScore for a knowledge graph with optional domain-specific metrics.

**Parameters:**

- `kg::KnowledgeGraph` - Knowledge graph to evaluate
- `domain_name::Union{String, Nothing}` - Optional domain name (inferred from `kg.metadata["domain"]` if not provided)
- `include_domain_metrics::Bool` - Whether to include domain-specific metrics in result

**Returns:**

- `ValidityScoreResult` with `metadata["domain_metrics"]` if domain metrics are included

**Example:**

```julia
result = evaluate_validity(kg, domain_name="biomedical")
if haskey(result.metadata, "domain_metrics")
    domain_metrics = result.metadata["domain_metrics"]
    println("UMLS validation rate: ", domain_metrics["umls_validation_rate"])
end
```

### `evaluate_with_graphrag(graph, questions)`

Evaluate knowledge graph using GraphRAG methodology.

**Parameters:**

- `graph::KnowledgeGraph` - Knowledge graph to evaluate
- `questions::Vector{String}` - Questions to test

**Returns:**

- `GraphRAGResults` - Evaluation results

## Utility Functions

### `validate_graph(graph)`

Validate a knowledge graph structure.

**Parameters:**

- `graph::KnowledgeGraph` - Graph to validate

**Returns:**

- `Bool` - True if valid

### `filter_by_confidence(graph, threshold)`

Filter graph elements by confidence threshold.

**Parameters:**

- `graph::KnowledgeGraph` - Graph to filter
- `threshold::Float64` - Confidence threshold

**Returns:**

- `KnowledgeGraph` - Filtered graph

### `merge_graphs(graphs)`

Merge multiple knowledge graphs.

**Parameters:**

- `graphs::Vector{KnowledgeGraph}` - Graphs to merge

**Returns:**

- `KnowledgeGraph` - Merged graph

## Error Handling

### `GraphMERTError`

Base exception for GraphMERT operations.

### `ModelLoadingError`

Error loading a model file.

### `EntityExtractionError`

Error during entity extraction.

### `RelationExtractionError`

Error during relation extraction.

### Domain Registration Errors

When a domain is not registered, clear error messages are provided with instructions on how to register the domain.

## Performance Monitoring

### `monitor_performance(f, args...)`

Monitor performance of a function.

**Parameters:**

- `f::Function` - Function to monitor
- `args...` - Function arguments

**Returns:**

- `Tuple` - (result, performance_metrics)

### `get_memory_usage()`

Get current memory usage.

**Returns:**

- `Int` - Memory usage in bytes

### `get_processing_speed()`

Get current processing speed.

**Returns:**

- `Float64` - Tokens per second

## Related Documentation

- [Domain API Reference](domain.md) - Complete domain system API
- [Core API Reference](core.md) - Core GraphMERT functions
- [Domain Usage Guide](../../DOMAIN_USAGE_GUIDE.md) - User guide for domains
- [Domain Developer Guide](../../DOMAIN_DEVELOPER_GUIDE.md) - Guide for creating custom domains
