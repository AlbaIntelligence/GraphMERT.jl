# Core API Reference

This page documents the core functions and data structures in GraphMERT.jl.

## Main Functions

### `extract_knowledge_graph`

```julia
extract_knowledge_graph(text::String; kwargs...) -> KnowledgeGraph
```

Extract a knowledge graph from text using domain-specific providers.

**Arguments:**
- `text::String`: Input text
- `model::GraphMERTModel`: GraphMERT model instance
- `options::ProcessingOptions`: Processing options (must include `domain` field)

**Returns:**
- `KnowledgeGraph`: Extracted knowledge graph

**Example:**
```julia
# Load and register domain
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Extract with domain
text = "Diabetes is treated with insulin."
options = ProcessingOptions(domain="biomedical")
graph = extract_knowledge_graph(text, model; options=options)
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
    entities::Vector{Entity}           # Generic entities (domain-agnostic)
    relations::Vector{Relation}         # Generic relations (domain-agnostic)
    metadata::Dict{String, Any}         # Graph metadata (includes domain field)
    created_at::DateTime
end
```

Main data structure representing a knowledge graph. Uses generic `Entity` and `Relation` types that work with all domains.

**Note:** The `metadata` field includes a `domain` field indicating which domain was used for extraction.

### `Entity` (Generic, Domain-Agnostic)

```julia
struct Entity
    id::String
    text::String
    label::String
    entity_type::String      # Domain-specific entity type
    domain::String          # Domain identifier
    attributes::Dict{String, Any}
    position::TextPosition
    confidence::Float64
    provenance::String
end
```

Generic entity structure used by all domains.

**Key Fields:**
- `entity_type::String` - Domain-specific entity type (e.g., "DISEASE", "PERSON")
- `domain::String` - Domain identifier (e.g., "biomedical", "wikipedia")
- `attributes::Dict{String, Any}` - Domain-specific attributes (e.g., CUI for biomedical, QID for Wikipedia)

### `Relation` (Generic, Domain-Agnostic)

```julia
struct Relation
    head::String             # Entity ID of head entity
    tail::String             # Entity ID of tail entity
    relation_type::String    # Domain-specific relation type
    confidence::Float64
    attributes::Dict{String, Any}
    created_at::DateTime
end
```

Generic relation structure used by all domains.

**Key Fields:**
- `relation_type::String` - Domain-specific relation type (e.g., "TREATS", "BORN_IN")
- `head::String` - Entity ID of head entity
- `tail::String` - Entity ID of tail entity

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

### `ProcessingOptions` (Updated for Domain System)

```julia
struct ProcessingOptions
    domain::String                    # Required: Domain identifier
    max_length::Int
    batch_size::Int
    use_umls::Bool                   # Domain-specific (biomedical)
    use_helper_llm::Bool
    confidence_threshold::Float64
    entity_types::Vector{String}
    relation_types::Vector{String}
    cache_enabled::Bool
    parallel_processing::Bool
    verbose::Bool
end
```

Configuration options for GraphMERT processing with domain support.

**Key Field:**
- `domain::String` - **Required**: Domain identifier (e.g., "biomedical", "wikipedia")

**Fields:**
- `domain::String` - Domain identifier (must match registered domain)
- `confidence_threshold::Float64` - Minimum confidence threshold (default: 0.5)
- `max_entities::Int` - Maximum entities to extract
- `max_relations::Int` - Maximum relations to extract
- `use_umls::Bool` - Enable UMLS integration (biomedical domain only)
- `use_helper_llm::Bool` - Enable helper LLM
- `batch_size::Int` - Batch size for processing (default: 32)
- `max_length::Int` - Maximum text length (default: 512)
- `cache_enabled::Bool` - Enable caching (default: true)
- `parallel_processing::Bool` - Enable parallel processing (default: false)
- `verbose::Bool` - Enable verbose output (default: false)

**Example:**
```julia
# Biomedical domain
options_bio = ProcessingOptions(
    domain="biomedical",
    confidence_threshold=0.8,
    use_umls=true
)

# Wikipedia domain
options_wiki = ProcessingOptions(
    domain="wikipedia",
    confidence_threshold=0.7
)
```

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

### Entity Types (Domain-Specific)

Entity types are domain-specific. Common types include:

**Biomedical Domain:**
- `DISEASE`, `DRUG`, `PROTEIN`, `GENE`, `ANATOMY`, `SYMPTOM`, `PROCEDURE`, etc.

**Wikipedia Domain:**
- `PERSON`, `ORGANIZATION`, `LOCATION`, `CONCEPT`, `EVENT`, `TECHNOLOGY`, etc.

### Relation Types (Domain-Specific)

Relation types are domain-specific. Common types include:

**Biomedical Domain:**
- `TREATS`, `CAUSES`, `ASSOCIATED_WITH`, `PREVENTS`, `INDICATES`, etc.

**Wikipedia Domain:**
- `BORN_IN`, `DIED_IN`, `WORKED_AT`, `FOUNDED`, `CREATED_BY`, etc.

## Examples

### Basic Usage with Domain System

```julia
using GraphMERT

# Load and register domain
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Extract with domain
text = "Diabetes is treated with metformin."
options = ProcessingOptions(domain="biomedical", confidence_threshold=0.8)
model = create_graphmert_model(GraphMERTConfig())
graph = extract_knowledge_graph(text, model; options=options)

# Access results
println("Extracted $(length(graph.entities)) entities")
println("Extracted $(length(graph.relations)) relations")
```

### Domain Switching

```julia
# Load multiple domains
include("GraphMERT/src/domains/biomedical.jl")
include("GraphMERT/src/domains/wikipedia.jl")

bio_domain = load_biomedical_domain()
wiki_domain = load_wikipedia_domain()
register_domain!("biomedical", bio_domain)
register_domain!("wikipedia", wiki_domain)

# Extract with different domains
bio_text = "Diabetes is treated with metformin."
bio_options = ProcessingOptions(domain="biomedical")
bio_graph = extract_knowledge_graph(bio_text, model; options=bio_options)

wiki_text = "Leonardo da Vinci was born in Vinci, Italy."
wiki_options = ProcessingOptions(domain="wikipedia")
wiki_graph = extract_knowledge_graph(wiki_text, model; options=wiki_options)
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