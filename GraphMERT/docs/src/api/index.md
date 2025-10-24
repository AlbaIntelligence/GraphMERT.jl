# GraphMERT.jl API Reference

Complete API documentation for GraphMERT.jl.

## Core Functions

### `extract_knowledge_graph(text, model, options)`

Extract a knowledge graph from input text using a GraphMERT model.

**Parameters:**

- `text::Union{String, Vector{String}}` - Input text or batch of texts
- `model::Union{GraphMERTModel, String}` - Model instance or path to model
- `options::ProcessingOptions` - Processing configuration (optional)

**Returns:**

- `KnowledgeGraph` - Extracted knowledge graph

**Example:**

```julia
graph = extract_knowledge_graph("Diabetes affects blood sugar levels.", model)
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
- `options::ProcessingOptions` - Processing options

**Returns:**

- `PreprocessedText` - Preprocessed text ready for processing

## Data Structures

### `KnowledgeGraph`

Main output structure containing the complete knowledge graph.

**Fields:**

- `entities::Vector{BiomedicalEntity}` - List of extracted entities
- `relations::Vector{BiomedicalRelation}` - List of extracted relations
- `metadata::GraphMetadata` - Graph-level metadata
- `confidence_threshold::Float64` - Minimum confidence threshold
- `created_at::DateTime` - Creation timestamp
- `model_info::GraphMERTModelInfo` - Model information
- `umls_mappings::Dict{String, String}` - UMLS CUI mappings
- `fact_score::Float64` - FActScore for the graph
- `validity_score::Float64` - ValidityScore for the graph

### `BiomedicalEntity`

Represents an extracted biomedical entity.

**Fields:**

- `id::String` - Unique identifier
- `text::String` - Original text
- `label::String` - Entity type (DISEASE, DRUG, etc.)
- `confidence::Float64` - Confidence score (0.0-1.0)
- `position::EntityPosition` - Position in text
- `attributes::Dict{String, Any}` - Additional attributes
- `created_at::DateTime` - Extraction timestamp

### `BiomedicalRelation`

Represents an extracted biomedical relation.

**Fields:**

- `head::String` - Head entity
- `tail::String` - Tail entity
- `relation_type::String` - Relation type (TREATS, CAUSES, etc.)
- `confidence::Float64` - Confidence score (0.0-1.0)
- `attributes::Dict{String, Any}` - Additional attributes
- `created_at::DateTime` - Extraction timestamp

## Configuration

### `ProcessingOptions`

Configuration for text processing.

**Fields:**

- `confidence_threshold::Float64` - Minimum confidence for inclusion
- `max_entities::Int` - Maximum number of entities to extract
- `max_relations::Int` - Maximum number of relations to extract
- `umls_enabled::Bool` - Enable UMLS integration
- `helper_llm_enabled::Bool` - Enable helper LLM
- `performance_mode::Symbol` - Performance mode (:fast, :balanced, :accurate)

### `GraphMERTConfig`

Comprehensive configuration for GraphMERT.

**Fields:**

- `model_path::String` - Path to model file
- `umls_enabled::Bool` - Enable UMLS integration
- `helper_llm_enabled::Bool` - Enable helper LLM
- `performance_mode::Symbol` - Performance mode
- `memory_limit::Int` - Memory limit in MB
- `batch_size::Int` - Batch size for processing

## Evaluation Functions

### `calculate_factscore(graph)`

Calculate FActScore for a knowledge graph.

**Parameters:**

- `graph::KnowledgeGraph` - Knowledge graph to evaluate

**Returns:**

- `Float64` - FActScore (0.0-1.0)

### `calculate_validity_score(graph)`

Calculate ValidityScore for a knowledge graph.

**Parameters:**

- `graph::KnowledgeGraph` - Knowledge graph to evaluate

**Returns:**

- `Float64` - ValidityScore (0.0-1.0)

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