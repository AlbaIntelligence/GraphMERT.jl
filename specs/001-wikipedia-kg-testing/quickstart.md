# Quickstart: Wikipedia Knowledge Graph Testing

**Date**: 2026-03-10

## Prerequisites

- Julia 1.8+
- GraphMERT.jl installed
- Wikipedia domain module loaded

## Running Tests

### 1. Load the Wikipedia Domain

```julia
using GraphMERT

# Load and register Wikipedia domain
include("GraphMERT/src/domains/wikipedia.jl")
wiki_domain = load_wikipedia_domain()
register_domain!("wikipedia", wiki_domain)
```

### 2. Prepare Test Articles

```julia
# Sample French monarchy articles (text content)
test_articles = [
    "Louis XIV (1638-1715) was King of France from 1643 until his death...",
    "Henry IV (1553-1610) was the first Bourbon king of France...",
    # ... more articles
]

# Or load from Wikipedia API
using HTTP
response = HTTP.get("https://en.wikipedia.org/api/rest_v1/page/summary/Louis_XIV")
```

### 3. Run Entity Extraction

```julia
options = ProcessingOptions(domain="wikipedia", confidence_threshold=0.5)

for article in test_articles
    entities = extract_entities(wiki_domain, article, options)
    println("Extracted $(length(entities)) entities")
end
```

### 4. Run Full Extraction Pipeline

```julia
# Create a model (or use mock for testing)
model = create_graphmert_model(GraphMERTConfig())

# Extract knowledge graphs
for article in test_articles
    kg = extract_knowledge_graph(article, model; options=options)
    println("KG: $(length(kg.entities)) entities, $(length(kg.relations)) relations")
end
```

### 5. Evaluate Quality

```julia
# Compare against reference facts
reference_facts = [
    ReferenceFact("Louis XIV", "reigned_after", "Louis XIII", "Louis XIV", true),
    ReferenceFact("Louis XIV", "parent_of", "Louis XV", "Louis XIV", true),
    # ... more facts
]

# Calculate metrics
metrics = evaluate_knowledge_graph(kg, reference_facts)
println("Precision: $(metrics.entity_precision)")
println("Recall: $(metrics.entity_recall)")
```

## Expected Output

```
Extracted entities from Louis XIV article: 15
Extracted entities from Henry IV article: 12
Knowledge Graph - Entities: 15, Relations: 8
Entity Precision: 0.85
Relation Precision: 0.72
Fact Capture Rate: 0.78
```

## Common Issues

1. **No entities extracted**: Check domain registration and options
2. **Low precision**: Lower confidence threshold
3. **Missing relations**: Verify relation extraction is enabled
4. **Performance issues**: Reduce batch size

## Next Steps

- Run full test suite: `julia GraphMERT/test/runtests.jl`
- Check existing examples: `examples/wikipedia/`
- Review evaluation metrics in `GraphMERT/src/evaluation/`
