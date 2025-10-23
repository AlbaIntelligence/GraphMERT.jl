# Quick Start

This guide will get you up and running with GraphMERT.jl in just a few minutes.

## Basic Example

```julia
using GraphMERT

# Sample biomedical text
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by high blood glucose levels.
Insulin is a hormone produced by the pancreas that regulates blood sugar.
Metformin is commonly prescribed to treat type 2 diabetes.
"""

# Extract knowledge graph
graph = extract_knowledge_graph(text)

# Display results
println("Extracted $(length(graph.entities)) entities and $(length(graph.relations)) relations")

# Show entities
for entity in graph.entities
    println("- $(entity.text) ($(entity.label)) - Confidence: $(entity.confidence)")
end

# Show relations
for relation in graph.relations
    println("- $(relation.head) --[$(relation.relation_type)]--> $(relation.tail)")
end
```

## Configuration

### Basic Configuration

```julia
# Create configuration
config = GraphMERTConfig(
    min_confidence = 0.7,
    max_entities = 100,
    enable_umls = true
)

# Extract with configuration
graph = extract_knowledge_graph(text, config)
```

### Advanced Configuration

```julia
# Advanced configuration
config = GraphMERTConfig(
    min_confidence = 0.8,
    max_entities = 200,
    enable_umls = true,
    umls_threshold = 0.9,
    enable_llm = true,
    llm_model = "gpt-3.5-turbo"
)

graph = extract_knowledge_graph(text, config)
```

## Batch Processing

For processing multiple documents:

```julia
# Prepare documents
documents = [
    "Diabetes is a chronic condition...",
    "Insulin regulates blood sugar...",
    "Metformin treats diabetes..."
]

# Batch processing configuration
batch_config = BatchProcessingConfig(
    batch_size = 10,
    enable_parallel = true,
    memory_limit = 4.0  # GB
)

# Process in batches
results = extract_knowledge_graph_batch(documents, batch_config)

# Merge results
merged_graph = merge_knowledge_graphs([r.graph for r in results])
```

## Evaluation

Evaluate your knowledge graph quality:

```julia
# Evaluate with FActScore
factscore_result = evaluate_factscore(graph)
println("FActScore: $(factscore_result.score)")

# Evaluate with ValidityScore
validity_result = evaluate_validity(graph)
println("ValidityScore: $(validity_result.score)")

# Evaluate with GraphRAG
graphrag_result = evaluate_graphrag(graph)
println("GraphRAG Score: $(graphrag_result.score)")
```

## Export Results

Export your knowledge graph in various formats:

```julia
# Export to JSON
export_knowledge_graph(graph, "json", filepath="results.json")

# Export to CSV
export_knowledge_graph(graph, "csv", filepath="results")

# Export to RDF
export_knowledge_graph(graph, "rdf", filepath="results.rdf")

# Export to Turtle
export_knowledge_graph(graph, "ttl", filepath="results.ttl")
```

## Performance Monitoring

Monitor performance during processing:

```julia
# Create performance monitor
monitor = create_performance_monitor()

# Process with monitoring
graph = extract_knowledge_graph(text, monitor=monitor)

# Get performance metrics
metrics = get_performance_metrics(monitor)
println("Processing time: $(metrics.processing_time)s")
println("Memory usage: $(metrics.memory_usage)MB")
println("Throughput: $(metrics.throughput) tokens/s")
```

## Next Steps

Now that you have the basics, explore:

- [User Guide](@ref): Comprehensive usage guide
- [Examples](@ref): Detailed examples and tutorials
- [API Reference](@ref): Complete API documentation
- [Scientific Background](@ref): Algorithm details and research

## Common Patterns

### Processing PubMed Abstracts

```julia
# Load PubMed data
abstracts = load_pubmed_abstracts("diabetes", max_results=100)

# Process abstracts
graphs = [extract_knowledge_graph(abstract) for abstract in abstracts]

# Merge all graphs
combined_graph = merge_knowledge_graphs(graphs)
```

### Training Custom Models

```julia
# Load training data
training_data = load_biomedical_corpus("path/to/corpus")

# Create training configuration
training_config = TrainingConfig(
    epochs = 10,
    batch_size = 32,
    learning_rate = 1e-4
)

# Train model
model = train_graphmert(training_data, training_config)
```

### Filtering and Post-processing

```julia
# Filter by confidence
filtered_graph = filter_knowledge_graph(
    graph, 
    min_confidence = 0.8
)

# Filter by entity types
disease_graph = filter_knowledge_graph(
    graph,
    entity_types = ["DISEASE", "DRUG"]
)

# Filter by relation types
treatment_graph = filter_knowledge_graph(
    graph,
    relation_types = ["TREATS", "INDICATES"]
)
```
