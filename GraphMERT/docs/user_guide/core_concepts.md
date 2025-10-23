# Core Concepts

This guide explains the fundamental concepts behind GraphMERT.jl and how they work together to extract knowledge graphs from biomedical text.

## Knowledge Graphs

A **knowledge graph** is a structured representation of information where:
- **Entities** represent real-world objects (diseases, drugs, proteins, etc.)
- **Relations** represent relationships between entities
- **Metadata** provides additional context and confidence scores

```julia
# Example knowledge graph structure
struct KnowledgeGraph
    entities::Vector{BiomedicalEntity}
    relations::Vector{BiomedicalRelation}
    metadata::Dict{String, Any}
    created_at::DateTime
end
```

## Biomedical Entities

Biomedical entities represent medical concepts with specific attributes:

```julia
struct BiomedicalEntity
    text::String           # Surface form: "diabetes mellitus"
    label::String          # Type: "DISEASE"
    id::String            # Unique identifier: "C0012634"
    confidence::Float64    # Confidence score: 0.95
    position::TextPosition # Location in text
    attributes::Dict{String, Any}
    created_at::DateTime
end
```

### Entity Types

GraphMERT.jl recognizes these biomedical entity types:

- **DISEASE**: Medical conditions (diabetes, cancer, hypertension)
- **DRUG**: Medications (metformin, insulin, aspirin)
- **PROTEIN**: Biological proteins (insulin, hemoglobin)
- **GENE**: Genetic elements (BRCA1, TP53)
- **ANATOMY**: Body parts (pancreas, heart, brain)
- **SYMPTOM**: Clinical signs (pain, fever, fatigue)
- **PROCEDURE**: Medical procedures (surgery, biopsy)
- **ORGANISM**: Living organisms (E. coli, human)
- **CHEMICAL**: Chemical compounds (glucose, cholesterol)
- **CELL_TYPE**: Cell types (T-cell, neuron)

## Biomedical Relations

Relations connect entities and represent semantic relationships:

```julia
struct BiomedicalRelation
    head::String           # Source entity: "diabetes"
    tail::String           # Target entity: "insulin"
    relation_type::String  # Relation: "TREATS"
    confidence::Float64    # Confidence: 0.87
    attributes::Dict{String, Any}
    created_at::DateTime
end
```

### Relation Types

Common biomedical relation types include:

- **TREATS**: Drug treats disease
- **CAUSES**: Disease causes symptom
- **INDICATES**: Symptom indicates disease
- **LOCATED_IN**: Organ located in body part
- **PART_OF**: Component relationship
- **REGULATES**: Protein regulates process
- **INTERACTS_WITH**: Drug interacts with protein
- **ADMINISTERED_FOR**: Drug administered for condition

## Text Processing Pipeline

GraphMERT.jl processes text through several stages:

### 1. Text Preprocessing

```julia
# Clean and normalize text
processed_text = preprocess_text(
    text,
    max_length = 512,
    remove_stopwords = true
)
```

### 2. Entity Recognition

```julia
# Extract biomedical entities
entities = extract_biomedical_terms(
    processed_text,
    min_confidence = 0.7
)
```

### 3. Relation Extraction

```julia
# Identify relationships
relations = match_relations_for_entities(
    entities,
    processed_text
)
```

### 4. Knowledge Graph Construction

```julia
# Build final knowledge graph
graph = KnowledgeGraph(
    entities,
    relations,
    metadata,
    now()
)
```

## Leafy Chain Graphs

GraphMERT uses a **leafy chain graph** structure to represent text:

```julia
struct LeafyChainGraph
    root_nodes::Vector{ChainGraphNode}    # Main concepts
    leaf_nodes::Vector{ChainGraphNode}    # Supporting details
    edges::Vector{GraphEdge}              # Connections
    metadata::Dict{String, Any}
end
```

### Graph Structure

- **Root Nodes**: Primary entities (diseases, drugs)
- **Leaf Nodes**: Supporting information (symptoms, procedures)
- **Edges**: Semantic relationships between nodes
- **Metadata**: Confidence scores and context

## UMLS Integration

The Unified Medical Language System (UMLS) provides:

### Concept Linking

```julia
# Link entity to UMLS concept
umls_result = link_entity_to_umls(
    entity,
    threshold = 0.8
)

# Access UMLS information
println("CUI: ", umls_result.cui)
println("Semantic Types: ", umls_result.semantic_types)
```

### Semantic Types

UMLS semantic types provide detailed categorization:

- **T047**: Disease or Syndrome
- **T121**: Pharmacologic Substance
- **T116**: Amino Acid, Peptide, or Protein
- **T017**: Anatomical Structure
- **T184**: Sign or Symptom

## Confidence Scoring

GraphMERT.jl provides multiple confidence metrics:

### Entity Confidence

```julia
# Calculate entity confidence
confidence = calculate_entity_confidence(
    entity,
    context = text,
    umls_score = 0.9
)
```

### Relation Confidence

```julia
# Calculate relation confidence
confidence = calculate_relation_confidence(
    relation,
    head_entity,
    tail_entity,
    context = text
)
```

### Factors Affecting Confidence

1. **Text Frequency**: How often entity appears
2. **Context Quality**: Surrounding text relevance
3. **UMLS Match**: Quality of UMLS linking
4. **Co-occurrence**: Entity pair frequency
5. **Linguistic Patterns**: Grammatical structure

## Performance Considerations

### Memory Usage

```julia
# Monitor memory usage
monitor = create_memory_monitor()
graph = extract_knowledge_graph(text, monitor=monitor)

# Check memory usage
println("Memory used: ", get_memory_usage(monitor), " MB")
```

### Throughput Optimization

```julia
# Batch processing for efficiency
batch_config = BatchProcessingConfig(
    batch_size = 10,
    enable_parallel = true,
    memory_limit = 4.0
)

results = extract_knowledge_graph_batch(documents, batch_config)
```

### Quality vs Speed Trade-offs

- **High Quality**: Lower confidence thresholds, more processing
- **High Speed**: Higher confidence thresholds, batch processing
- **Balanced**: Medium thresholds, optimized batch sizes

## Best Practices

### Text Preparation

1. **Clean Text**: Remove formatting artifacts
2. **Normalize**: Standardize medical terminology
3. **Segment**: Break long texts into manageable chunks
4. **Validate**: Check for encoding issues

### Configuration Tuning

1. **Start Conservative**: High confidence thresholds
2. **Iterate**: Gradually lower thresholds
3. **Monitor**: Track performance metrics
4. **Validate**: Check output quality

### Error Handling

```julia
try
    graph = extract_knowledge_graph(text)
catch e
    println("Error: ", e)
    # Handle gracefully
end
```

## Next Steps

Now that you understand the core concepts:

- [Knowledge Graph Extraction](@ref): Detailed extraction guide
- [Model Training](@ref): Training custom models
- [Evaluation Metrics](@ref): Quality assessment
- [Performance Optimization](@ref): Speed and memory optimization
