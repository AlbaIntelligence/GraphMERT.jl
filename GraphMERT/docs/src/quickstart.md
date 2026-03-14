# GraphMERT.jl Quick Start Guide

Get up and running with GraphMERT.jl in 30 minutes!

## Installation

```julia
using Pkg
Pkg.add("GraphMERT")
```

## Basic Usage

### 1. Load the Package

```julia
using GraphMERT
```

### 2. Load a Pre-trained Model

```julia
# Load a pre-trained GraphMERT model
model = load_model("path/to/graphmert_model.onnx")
```

### 3. Extract Knowledge Graph

```julia
# Input text
text = "Diabetes is a chronic condition that affects blood sugar levels. Insulin therapy is commonly used for treatment."

# Extract knowledge graph
graph = extract_knowledge_graph(text, model)

# View results
println("Extracted $(length(graph.entities)) entities")
println("Extracted $(length(graph.relations)) relations")
```

### 4. Access Results

```julia
# Access entities
for entity in graph.entities
    println("Entity: $(entity.text) (Type: $(entity.label), Confidence: $(entity.confidence))")
end

# Access relations
for relation in graph.relations
    println("Relation: $(relation.head) --[$(relation.relation_type)]--> $(relation.tail)")
end
```

## Configuration

### Basic Configuration

```julia
# Create processing options
options = ProcessingOptions(
    confidence_threshold = 0.8,
    max_entities = 100,
    max_relations = 50
)

# Extract with custom options
graph = extract_knowledge_graph(text, model, options)
```

### Advanced Configuration

```julia
# Create comprehensive configuration
config = GraphMERTConfig(
    model_path = "path/to/model.onnx",
    umls_enabled = true,
    helper_llm_enabled = true,
    performance_mode = :balanced
)

# Use configuration
graph = extract_knowledge_graph(text, config)
```

## Batch Processing

```julia
# Process multiple documents
texts = [
    "Diabetes affects blood sugar levels.",
    "Insulin is used to treat diabetes.",
    "Metformin is an oral diabetes medication."
]

# Batch processing
graphs = extract_knowledge_graph(texts, model)

# Merge results
combined_graph = merge_graphs(graphs)
```

## Evaluation

```julia
# Calculate evaluation metrics
fact_score = calculate_factscore(graph)
validity_score = calculate_validity_score(graph)

println("FActScore: $(fact_score)")
println("ValidityScore: $(validity_score)")
```

## Quality Comparison

Compare local LLM extraction quality against external API baseline:

```julia
using GraphMERT

local_config = LocalLLMConfig(;
    model_path="/path/to/tinyllama-1.1b.gguf",
    temperature=0.2,
    context_length=4096,
    max_tokens=512
)

articles = [
    ("article1", "First article text..."),
    ("article2", "Second article text..."),
    # Add more articles as needed
]

result = run_batch_comparison(articles, local_config)

println("Average recall: $(round(result.average_recall * 100, digits=1))%")
println("Pass rate: $(round(result.pass_rate * 100, digits=1))%")
```

**Success Criteria:**
- SC-003: Entity recall ≥ 70% compared to external API baseline
- SC-004: Entity overlap ≥ 80% on 50 articles

If quality is below threshold, try tuning:
- **Temperature**: Lower (0.1-0.3) for more deterministic output
- **Context length**: Increase to 4096+ for complex articles
- **Max tokens**: Increase to 512-1024 for longer entity names

See `GraphMERT/test/local/test_quality_comparison.jl` for detailed usage.

## Next Steps

- [Core Concepts](user_guide/core_concepts.md) - Learn more about core functionality
- [API Reference](api/core.md) - Complete API documentation
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure the model file exists and is compatible
2. **Memory Issues**: Reduce batch size or enable memory optimization
3. **Performance Issues**: Check configuration and hardware requirements

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [API Reference](api/core.md)
- Open an issue on GitHub

## Examples

See the documentation for complete working examples:

- [Basic Extraction](getting_started/quickstart.md)
- [Biomedical Processing](user_guide/core_concepts.md)