# GraphMERT API Contracts

**Purpose:** Define API contracts and interfaces for GraphMERT implementation
**Date:** 2024-12-19
**Feature:** GraphMERT Algorithm Replication

## Core API Functions

### Primary Interface

#### `extract_knowledge_graph(text, model, options)`

**Purpose:** Main function to extract knowledge graph from text

**Parameters:**

- `text::Union{String, Vector{String}}` - Input text or batch of texts
- `model::Union{ONNXModel, String}` - Model instance or path to model
- `options::ProcessingOptions` - Processing configuration

**Returns:**

- `KnowledgeGraph` - Extracted knowledge graph

**Exceptions:**

- `ModelLoadingError` - If model cannot be loaded
- `EntityExtractionError` - If entity extraction fails
- `RelationExtractionError` - If relation extraction fails

**Examples:**

```julia
# Single text processing
text = "John works at Microsoft in Seattle."
model = load_model("distilbert-base-uncased.onnx")
options = ProcessingOptions(confidence_threshold=0.8)
graph = extract_knowledge_graph(text, model, options)

# Batch processing
texts = ["John works at Microsoft.", "Mary lives in New York."]
graph = extract_knowledge_graph(texts, model, options)
```

#### `load_model(model_path, config)`

**Purpose:** Load ONNX model for inference

**Parameters:**

- `model_path::String` - Path to ONNX model file
- `config::TinyBERTConfig` - Model configuration

**Returns:**

- `ONNXModel` - Loaded model instance

**Exceptions:**

- `ModelLoadingError` - If model loading fails
- `FileNotFoundError` - If model file doesn't exist

**Examples:**

```julia
config = TinyBERTConfig(
    model_path="models/distilbert-base-uncased.onnx",
    max_length=512,
    batch_size=32
)
model = load_model("models/distilbert-base-uncased.onnx", config)
```

#### `preprocess_text(text, options)`

**Purpose:** Preprocess text for entity and relation extraction

**Parameters:**

- `text::String` - Input text
- `options::ProcessingOptions` - Processing configuration

**Returns:**

- `Vector{String}` - Preprocessed text chunks

**Examples:**

```julia
text = "Dr. Smith works at the University of California."
chunks = preprocess_text(text, options)
```

#### `extract_entities(text, model, options)`

**Purpose:** Extract entities from text using the model

**Parameters:**

- `text::Union{String, Vector{String}}` - Input text
- `model::ONNXModel` - Loaded model
- `options::ProcessingOptions` - Processing configuration

**Returns:**

- `Vector{Entity}` - Extracted entities

**Exceptions:**

- `EntityExtractionError` - If extraction fails

**Examples:**

```julia
entities = extract_entities(text, model, options)
```

#### `extract_relations(entities, text, model, options)`

**Purpose:** Extract relations between entities

**Parameters:**

- `entities::Vector{Entity}` - Extracted entities
- `text::String` - Original text
- `model::ONNXModel` - Loaded model
- `options::ProcessingOptions` - Processing configuration

**Returns:**

- `Vector{Relation}` - Extracted relations

**Exceptions:**

- `RelationExtractionError` - If extraction fails

**Examples:**

```julia
relations = extract_relations(entities, text, model, options)
```

### Utility Functions

#### `create_knowledge_graph(entities, relations, metadata)`

**Purpose:** Create knowledge graph from extracted elements

**Parameters:**

- `entities::Vector{Entity}` - Extracted entities
- `relations::Vector{Relation}` - Extracted relations
- `metadata::GraphMetadata` - Graph metadata

**Returns:**

- `KnowledgeGraph` - Constructed knowledge graph

**Examples:**

```julia
graph = create_knowledge_graph(entities, relations, metadata)
```

#### `validate_graph(graph)`

**Purpose:** Validate knowledge graph consistency

**Parameters:**

- `graph::KnowledgeGraph` - Knowledge graph to validate

**Returns:**

- `Bool` - True if valid, false otherwise

**Examples:**

```julia
is_valid = validate_graph(graph)
```

#### `filter_by_confidence(graph, threshold)`

**Purpose:** Filter graph elements by confidence threshold

**Parameters:**

- `graph::KnowledgeGraph` - Input knowledge graph
- `threshold::Float64` - Confidence threshold

**Returns:**

- `KnowledgeGraph` - Filtered knowledge graph

**Examples:**

```julia
filtered_graph = filter_by_confidence(graph, 0.8)
```

### Configuration Functions

#### `ProcessingOptions(; kwargs...)`

**Purpose:** Create processing options with default values

**Parameters:**

- `confidence_threshold::Float64 = 0.5` - Minimum confidence
- `max_entities::Int = 1000` - Maximum entities
- `max_relations::Int = 1000` - Maximum relations
- `batch_size::Int = 32` - Batch size
- `chunk_size::Int = 512` - Chunk size
- `enable_parallel::Bool = true` - Enable parallel processing
- `random_seed::Union{Int, Nothing} = nothing` - Random seed

**Returns:**

- `ProcessingOptions` - Configuration object

**Examples:**

```julia
options = ProcessingOptions(
    confidence_threshold=0.8,
    max_entities=500,
    enable_parallel=true
)
```

#### `TinyBERTConfig(; kwargs...)`

**Purpose:** Create TinyBERT model configuration

**Parameters:**

- `model_path::String` - Path to model file
- `vocab_path::String` - Path to vocabulary file
- `max_length::Int = 512` - Maximum sequence length
- `batch_size::Int = 32` - Batch size
- `device::String = "cpu"` - Device for inference
- `precision::String = "fp32"` - Model precision

**Returns:**

- `TinyBERTConfig` - Configuration object

**Examples:**

```julia
config = TinyBERTConfig(
    model_path="models/distilbert.onnx",
    max_length=512,
    device="cpu"
)
```

## Error Handling

### Exception Hierarchy

```julia
abstract type GraphMERTError <: Exception end

struct ModelLoadingError <: GraphMERTError
    message::String
    model_path::String
end

struct EntityExtractionError <: GraphMERTError
    message::String
    text::String
end

struct RelationExtractionError <: GraphMERTError
    message::String
    entities::Vector{Entity}
end

struct ValidationError <: GraphMERTError
    message::String
    element::Any
end
```

### Error Handling Patterns

```julia
function safe_extract_knowledge_graph(text, model, options)
    try
        extract_knowledge_graph(text, model, options)
    catch e
        if e isa ModelLoadingError
            @error "Model loading failed: $(e.message)"
            rethrow()
        elseif e isa EntityExtractionError
            @error "Entity extraction failed: $(e.message)"
            rethrow()
        else
            @error "Unexpected error: $(e.message)"
            rethrow()
        end
    end
end
```

## Performance Contracts

### Memory Usage

- **Single text processing:** < 1GB RAM
- **Batch processing (100 texts):** < 4GB RAM
- **Model loading:** < 500MB RAM

### Processing Speed

- **Text preprocessing:** > 10,000 tokens/second
- **Entity extraction:** > 5,000 tokens/second
- **Relation extraction:** > 2,000 tokens/second
- **Graph construction:** < 1 second for 1000 entities

### Accuracy Requirements

- **Entity F1 score:** > 0.85 on benchmark dataset
- **Relation F1 score:** > 0.80 on benchmark dataset
- **Overall F1 score:** > 0.82 on benchmark dataset

## API Design Principles

### Elegant Julia Design

- **Multiple Dispatch:** Use method dispatch for different input types
- **Type Safety:** Leverage Julia's type system for compile-time checks
- **Composability:** Design functions that compose well together
- **Broadcasting:** Support broadcasting for vectorized operations

### Examples of Elegant Design

```julia
# Multiple dispatch for different input types
extract_knowledge_graph(text::String, model, options) = ...
extract_knowledge_graph(texts::Vector{String}, model, options) = ...

# Broadcasting support
texts = ["Text 1", "Text 2", "Text 3"]
graphs = extract_knowledge_graph.(texts, model, options)

# Composability
graph = text |>
        x -> preprocess_text(x, options) |>
        x -> extract_entities(x, model, options) |>
        x -> extract_relations(x, text, model, options) |>
        x -> create_knowledge_graph(x, relations, metadata)
```

## Testing Contracts

### Unit Testing

- All public functions must have unit tests
- Test coverage must be > 90%
- All error conditions must be tested
- Performance benchmarks must be included

### Integration Testing

- End-to-end workflow testing
- Model loading and inference testing
- Memory usage validation
- Cross-platform compatibility testing

### Scientific Testing

- Accuracy validation against benchmark datasets
- Reproducibility testing with fixed random seeds
- Performance comparison with reference implementation
- Memory usage validation on target hardware

## Documentation Requirements

### Function Documentation

- All public functions must have docstrings
- Examples must be provided for all functions
- Parameter descriptions must be complete
- Return value descriptions must be clear
- Exception conditions must be documented

### Tutorial Documentation

- Quick start guide for basic usage
- Advanced usage examples
- Performance optimization guide
- Troubleshooting guide

### API Reference

- Complete API documentation
- Type definitions and relationships
- Configuration options
- Error handling guide
