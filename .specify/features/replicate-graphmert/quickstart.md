# GraphMERT Quickstart Guide

## Get Started in 5 Minutes

This guide shows you how to use GraphMERT to extract knowledge graphs from biomedical text.

**Last Updated**: 2025-01-20

---

## Installation

```julia
# Install GraphMERT
using Pkg
Pkg.add("GraphMERT")

# Or install from source
Pkg.add(url="https://github.com/yourusername/GraphMERT.jl")
```

**Requirements**:
- Julia 1.10+
- 8GB RAM minimum (16GB recommended)
- Optional: CUDA-capable GPU for training

---

## Quick Example: Extract Knowledge Graph

```julia
using GraphMERT

# 1. Load a pre-trained model
model = load_model("models/graphmert-diabetes.jld2")

# 2. Prepare your text
text = """
Diabetes mellitus is a chronic metabolic disorder characterized by
elevated blood glucose levels. Metformin is the first-line medication
for treating type 2 diabetes. It works by decreasing glucose production
in the liver and improving insulin sensitivity.
"""

# 3. Extract knowledge graph
kg = extract_knowledge_graph(text, model)

# 4. Explore results
println("Extracted $(length(kg.entities)) entities and $(length(kg.relations)) relations")

# Print entities
for entity in kg.entities
    println("Entity: $(entity.text)")
    println("  Type: $(entity.entity_type)")
    println("  CUI: $(entity.cui)")
    println("  Confidence: $(round(entity.confidence, digits=2))")
    println()
end

# Print relations
for (i, rel) in enumerate(kg.relations)
    head = kg.entities[rel.head_entity_id]
    tail = kg.entities[rel.tail_entity_id]
    println("Triple $i: $(head.text) --[$(rel.relation_type)]--> $(tail.text)")
    println("  Confidence: $(round(rel.confidence, digits=2))")
    println()
end
```

**Expected Output**:
```
Extracted 5 entities and 3 relations

Entity: Diabetes mellitus
  Type: Disease
  CUI: C0011849
  Confidence: 0.95

Entity: Metformin
  Type: Drug
  CUI: C0025598
  Confidence: 0.92

Triple 1: Metformin --[treats]--> Diabetes mellitus
  Confidence: 0.88
```

---

## Batch Processing

Process multiple documents efficiently:

```julia
# Load multiple abstracts
abstracts = [
    "Text about diabetes...",
    "Text about cardiovascular disease...",
    "Text about cancer..."
]

# Process in batch
kgs = extract_knowledge_graph_batch(abstracts, model,
    ProcessingOptions(batch_size=32, device=:cuda))

# Merge into single knowledge graph
merged_kg = merge_knowledge_graphs(kgs)

println("Total extracted: $(length(merged_kg.entities)) entities")
```

---

## Configuration Options

Customize processing with `ProcessingOptions`:

```julia
options = ProcessingOptions(
    batch_size=32,                    # Batch size for processing
    max_length=1024,                  # Maximum sequence length
    device=:cuda,                      # Use GPU
    top_k_predictions=20,             # Top-k for tail prediction
    similarity_threshold=0.8,         # Filtering threshold
    enable_provenance_tracking=true   # Track source sentences
)

kg = extract_knowledge_graph(text, model, options=options)
```

---

## Exporting Results

Export your knowledge graph to various formats:

```julia
# Export to JSON
export_knowledge_graph(kg, "output.json", format=:json)

# Export to RDF
export_knowledge_graph(kg, "output.rdf", format=:rdf)

# Export to Neo4j Cypher
export_knowledge_graph(kg, "output.cypher", format=:neo4j)
```

---

## Filtering Results

Filter knowledge graph by confidence and types:

```julia
# Filter by confidence
filtered_kg = filter_knowledge_graph(kg,
    min_entity_confidence=0.7,
    min_relation_confidence=0.6
)

# Filter by entity types
disease_kg = filter_knowledge_graph(kg,
    entity_types=["Disease", "Symptom"]
)

# Filter by relation types
treatment_kg = filter_knowledge_graph(kg,
    relation_types=["treats", "prevents"]
)
```

---

## Training Your Own Model

Train a custom GraphMERT model on your domain:

```julia
# 1. Prepare training data
train_texts = load_pubmed_abstracts("my_training_data.json")

# 2. (Optional) Prepare seed knowledge graph
seed_triples = load_umls_triples("seed_kg.json")

# 3. Configure model
config = default_graphmert_config(
    hidden_size=512,
    num_hidden_layers=12
)

# 4. Train
model = train_graphmert(
    train_texts,
    config,
    seed_kg=seed_triples,
    num_epochs=10,
    checkpoint_dir="checkpoints/my_model"
)

# 5. Save trained model
save_model(model, "my_trained_model.jld2")
```

---

## Evaluation

Evaluate your knowledge graph quality:

```julia
using GraphMERT.Evaluation

# FActScore (factuality)
factscore_result = evaluate_factscore(kg, original_text)
println("FActScore: $(factscore_result.overall_score)")

# ValidityScore (ontological validity)
validity_result = evaluate_validity(kg)
println("ValidityScore: $(validity_result.overall_score)")

# GraphRAG (QA utility)
questions = ["What treats diabetes?", "What are symptoms of diabetes?"]
answers = ["Metformin", "Hyperglycemia"]
graphrag_result = evaluate_graphrag(kg, questions, answers)
println("GraphRAG Accuracy: $(graphrag_result.accuracy)")
```

---

## Integration with UMLS

Link entities to UMLS biomedical ontology:

```julia
# Create UMLS client
umls = create_umls_client(ENV["UMLS_API_KEY"])

# Link entity
result = link_entity(umls, "diabetes mellitus")
println("CUI: $(result.cui)")
println("Semantic Types: $(result.semantic_types)")

# Get related concepts
related = get_related_concepts(umls, result.cui)
for concept in related
    println("$(concept.cui): $(concept.preferred_name)")
end
```

---

## Integration with Helper LLM

Use a helper LLM for enhanced extraction:

```julia
# Create LLM client (OpenAI)
llm = create_llm_client(:openai, api_key=ENV["OPENAI_API_KEY"])

# Or use local model
llm = create_llm_client(:local, model_name="llama3-70b-instruct")

# Extract with LLM assistance
kg = extract_knowledge_graph(text, model, llm_client=llm)
```

---

## Common Use Cases

### 1. Extract from PubMed Abstracts

```julia
using GraphMERT.BiomedicalText

# Fetch PubMed abstracts
abstracts = fetch_pubmed_abstracts("diabetes mellitus", max_results=100)

# Extract KG
kgs = extract_knowledge_graph_batch(abstracts, model)
merged_kg = merge_knowledge_graphs(kgs)

# Analyze results
entity_types = count_entity_types(merged_kg)
relation_types = count_relation_types(merged_kg)
```

### 2. Build Domain-Specific Knowledge Base

```julia
# Load domain corpus
corpus = load_text_corpus("cardiology_papers/")

# Extract and merge
kg = build_knowledge_base_from_corpus(corpus, model,
    batch_size=32,
    min_confidence=0.7,
    deduplicate=true
)

# Export for downstream use
export_knowledge_graph(kg, "cardiology_kb.json")
```

### 3. Continuous Knowledge Graph Construction

```julia
# Initialize empty KG
kg = KnowledgeGraph()

# Process new documents as they arrive
for doc in document_stream
    new_kg = extract_knowledge_graph(doc, model)
    kg = merge_knowledge_graphs([kg, new_kg], deduplicate=true)

    # Periodically checkpoint
    if length(kg.entities) % 1000 == 0
        save_knowledge_graph(kg, "checkpoint_$(timestamp()).json")
    end
end
```

---

## Performance Tips

### 1. Use GPU Acceleration

```julia
# Move model to GPU
model = to_device(model, :cuda)

# Process with GPU
kg = extract_knowledge_graph(text, model,
    options=ProcessingOptions(device=:cuda))
```

### 2. Batch Similar-Length Documents

```julia
# Sort by length for efficient batching
sorted_texts = sort(texts, by=length)

# Process in batches
kgs = extract_knowledge_graph_batch(sorted_texts, model,
    ProcessingOptions(batch_size=32))
```

### 3. Enable Caching

```julia
# Enable LLM response caching
llm = create_llm_client(:openai,
    api_key=ENV["OPENAI_API_KEY"],
    cache_enabled=true,
    cache_dir=".cache/llm"
)

# Enable UMLS caching
umls = create_umls_client(ENV["UMLS_API_KEY"],
    cache_enabled=true,
    cache_dir=".cache/umls"
)
```

---

## Troubleshooting

### Model Loading Errors

```julia
# Check model file exists
if !isfile("model.jld2")
    error("Model file not found!")
end

# Load with error handling
try
    model = load_model("model.jld2")
catch e
    @error "Failed to load model" exception=e
    # Try loading with less strict requirements
    model = load_model("model.jld2", strict=false)
end
```

### Memory Issues

```julia
# Reduce batch size
options = ProcessingOptions(batch_size=16)

# Process in smaller chunks
chunk_size = 100
for i in 1:chunk_size:length(texts)
    chunk = texts[i:min(i+chunk_size-1, end)]
    kgs_chunk = extract_knowledge_graph_batch(chunk, model, options=options)
    # Process results...
end
```

### API Rate Limiting

```julia
# UMLS rate limiting
umls = create_umls_client(ENV["UMLS_API_KEY"],
    rate_limit=50  # Reduce from default 100
)

# LLM rate limiting
llm = create_llm_client(:openai,
    api_key=ENV["OPENAI_API_KEY"],
    rate_limit=100  # Requests per minute
)
```

---

## Next Steps

- **Advanced Topics**: See full documentation for advanced features
- **API Reference**: Browse complete API documentation
- **Examples**: Check `examples/` directory for more use cases
- **Training Guide**: Learn how to train custom models
- **Evaluation Guide**: Deep dive into evaluation metrics

---

## Getting Help

- **Documentation**: [https://graphmert.readthedocs.io](https://graphmert.readthedocs.io)
- **Issues**: [https://github.com/yourusername/GraphMERT.jl/issues](https://github.com/yourusername/GraphMERT.jl/issues)
- **Discussions**: [https://github.com/yourusername/GraphMERT.jl/discussions](https://github.com/yourusername/GraphMERT.jl/discussions)

---

## Example Projects

- **Diabetes Knowledge Base**: Extract entities and relations from diabetes literature
- **Drug Discovery**: Build knowledge graphs from pharmaceutical research papers
- **Clinical Trials**: Extract structured data from clinical trial reports
- **Medical Question Answering**: Use GraphRAG for biomedical QA systems

---

**Ready to dive deeper?** Check out the [full documentation](docs/README.md) and [API reference](docs/api/index.md).
