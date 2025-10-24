# Core API Reference

This page documents the core functions and data structures in GraphMERT.jl.

## Main Functions

### `extract_knowledge_graph`

```julia
extract_knowledge_graph(text::String; kwargs...) -> KnowledgeGraph
```

Extract a knowledge graph from biomedical text.

**Arguments:**
- `text::String`: Input biomedical text
- `config::GraphMERTConfig`: Configuration options (optional)
- `monitor::PerformanceMonitor`: Performance monitoring (optional)

**Returns:**
- `KnowledgeGraph`: Extracted knowledge graph

**Example:**
```julia
text = "Diabetes is treated with insulin."
graph = extract_knowledge_graph(text)
```

### `extract_knowledge_graph_batch`

```julia
extract_knowledge_graph_batch(documents::Vector{String}, config::BatchProcessingConfig) -> BatchProcessingResult
```

Process multiple documents in batches for efficiency.

**Arguments:**
- `documents::Vector{String}`: Input documents
- `config::BatchProcessingConfig`: Batch processing configuration

**Returns:**
- `BatchProcessingResult`: Results with merged knowledge graph

**Example:**
```julia
documents = ["Text 1...", "Text 2...", "Text 3..."]
config = BatchProcessingConfig(batch_size=10)
result = extract_knowledge_graph_batch(documents, config)
```

## Data Structures

### `KnowledgeGraph`

```julia
struct KnowledgeGraph
    entities::Vector{BiomedicalEntity}
    relations::Vector{BiomedicalRelation}
    metadata::Dict{String, Any}
    created_at::DateTime
end
```

Main data structure representing a knowledge graph.

### `BiomedicalEntity`

```julia
struct BiomedicalEntity
    text::String
    label::String
    id::String
    confidence::Float64
    position::TextPosition
    attributes::Dict{String, Any}
    created_at::DateTime
end
```

Represents a biomedical entity with confidence and metadata.

### `BiomedicalRelation`

```julia
struct BiomedicalRelation
    head::String
    tail::String
    relation_type::String
    confidence::Float64
    attributes::Dict{String, Any}
    created_at::DateTime
end
```

Represents a relationship between two entities.

### `TextPosition`

```julia
struct TextPosition
    start::Int
    stop::Int
    start_char::Int
    stop_char::Int
end
```

Represents the position of text in the original document.

## Configuration

### `GraphMERTConfig`

```julia
struct GraphMERTConfig
    min_confidence::Float64
    max_entities::Int
    enable_umls::Bool
    umls_threshold::Float64
    enable_llm::Bool
    llm_model::String
    batch_size::Int
    memory_limit::Float64
end
```

Configuration options for knowledge graph extraction.

**Fields:**
- `min_confidence`: Minimum confidence threshold (default: 0.7)
- `max_entities`: Maximum entities to extract (default: 100)
- `enable_umls`: Enable UMLS integration (default: true)
- `umls_threshold`: UMLS matching threshold (default: 0.8)
- `enable_llm`: Enable LLM assistance (default: false)
- `llm_model`: LLM model to use (default: "gpt-3.5-turbo")
- `batch_size`: Batch size for processing (default: 10)
- `memory_limit`: Memory limit in GB (default: 4.0)

### `BatchProcessingConfig`

```julia
struct BatchProcessingConfig
    batch_size::Int
    enable_parallel::Bool
    memory_limit::Float64
    progress_callback::Function
end
```

Configuration for batch processing.

## Utility Functions

### `merge_knowledge_graphs`

```julia
merge_knowledge_graphs(graphs::Vector{KnowledgeGraph}) -> KnowledgeGraph
```

Merge multiple knowledge graphs into a single graph.

### `filter_knowledge_graph`

```julia
filter_knowledge_graph(kg::KnowledgeGraph; 
                     min_confidence::Float64=0.0,
                     entity_types::Vector{String}=String[],
                     relation_types::Vector{String}=String[]) -> KnowledgeGraph
```

Filter knowledge graph based on confidence and type criteria.

### `export_knowledge_graph`

```julia
export_knowledge_graph(kg::KnowledgeGraph, format::String; filepath::String="") -> String
```

Export knowledge graph in various formats (JSON, CSV, RDF, Turtle).

## Performance Monitoring

### `PerformanceMonitor`

```julia
struct PerformanceMonitor
    start_time::DateTime
    memory_usage::Vector{Float64}
    throughput::Vector{Float64}
    metrics::Dict{String, Any}
end
```

Monitor performance during processing.

### `create_performance_monitor`

```julia
create_performance_monitor() -> PerformanceMonitor
```

Create a new performance monitor.

### `get_performance_metrics`

```julia
get_performance_metrics(monitor::PerformanceMonitor) -> Dict{String, Any}
```

Get performance metrics from monitor.

## Error Handling

### `GraphMERTError`

```julia
struct GraphMERTError <: Exception
    message::String
    code::Int
    context::Dict{String, Any}
end
```

Custom error type for GraphMERT operations.

## Constants

### Entity Types

```julia
const DISEASE = "DISEASE"
const DRUG = "DRUG"
const PROTEIN = "PROTEIN"
const GENE = "GENE"
const ANATOMY = "ANATOMY"
const SYMPTOM = "SYMPTOM"
const PROCEDURE = "PROCEDURE"
const ORGANISM = "ORGANISM"
const CHEMICAL = "CHEMICAL"
const CELL_TYPE = "CELL_TYPE"
```

### Relation Types

```julia
const TREATS = "TREATS"
const CAUSES = "CAUSES"
const INDICATES = "INDICATES"
const LOCATED_IN = "LOCATED_IN"
const PART_OF = "PART_OF"
const REGULATES = "REGULATES"
const INTERACTS_WITH = "INTERACTS_WITH"
const ADMINISTERED_FOR = "ADMINISTERED_FOR"
```

## Examples

### Basic Usage

```julia
using GraphMERT

# Simple extraction
text = "Diabetes is treated with metformin."
graph = extract_knowledge_graph(text)

# With configuration
config = GraphMERTConfig(min_confidence=0.8)
graph = extract_knowledge_graph(text, config)

# With monitoring
monitor = create_performance_monitor()
graph = extract_knowledge_graph(text, monitor=monitor)
```

### Batch Processing

```julia
# Process multiple documents
documents = ["Text 1...", "Text 2...", "Text 3..."]
config = BatchProcessingConfig(batch_size=5)
result = extract_knowledge_graph_batch(documents, config)

# Access results
println("Processed: ", result.total_documents)
println("Entities: ", length(result.graph.entities))
```

### Filtering and Export

```julia
# Filter by confidence
filtered = filter_knowledge_graph(graph, min_confidence=0.8)

# Filter by entity types
diseases = filter_knowledge_graph(graph, entity_types=["DISEASE"])

# Export results
export_knowledge_graph(graph, "json", filepath="output.json")
export_knowledge_graph(graph, "csv", filepath="output")
```