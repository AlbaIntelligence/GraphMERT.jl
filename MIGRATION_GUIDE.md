# Migration Guide: From Biomedical-Specific API to Domain System

This guide helps you migrate your existing GraphMERT.jl code from the old biomedical-specific API to the new domain-agnostic system.

## Overview

GraphMERT.jl has been refactored to support multiple domains (biomedical, Wikipedia, and custom domains) through a pluggable domain system. While the old biomedical-specific functions still work for backward compatibility, they now emit deprecation warnings and should be migrated to the new domain system.

## Quick Migration Checklist

- [ ] Replace `BiomedicalEntity` with `Entity` (domain="biomedical")
- [ ] Replace `BiomedicalRelation` with `Relation` (domain="biomedical")
- [ ] Load and register the biomedical domain explicitly
- [ ] Update extraction calls to use domain providers
- [ ] Update seed injection calls to pass domain parameter
- [ ] Update evaluation calls to use domain metrics

## Deprecated Types and Functions

### 1. `BiomedicalEntity` → `Entity`

**Old Code:**
```julia
entity = BiomedicalEntity(
    "entity_1",
    "diabetes",
    "diabetes",
    cui="C0011849",
    semantic_types=["T047"],
    position=TextPosition(0, 0, 0, 7),
    confidence=0.95,
    provenance="text"
)
```

**New Code:**
```julia
attributes = Dict{String,Any}(
    "cui" => "C0011849",
    "semantic_types" => ["T047"]
)
entity = Entity(
    "entity_1",
    "diabetes",
    "diabetes",
    "biomedical",  # domain parameter
    attributes,
    TextPosition(0, 0, 0, 7),
    0.95,
    "text"
)
```

**Key Changes:**
- Use `Entity` instead of `BiomedicalEntity`
- Pass `domain="biomedical"` as the 5th parameter
- Store `cui` and `semantic_types` in the `attributes` dictionary

### 2. `BiomedicalRelation` → `Relation`

**Old Code:**
```julia
relation = BiomedicalRelation(
    "entity_1",
    "entity_2",
    "TREATS",
    0.85,
    provenance="text",
    evidence="insulin treats diabetes"
)
```

**New Code:**
```julia
relation = Relation(
    "entity_1",
    "entity_2",
    "TREATS",
    0.85,
    "biomedical",  # domain parameter
    "text",
    "insulin treats diabetes",
    Dict{String,Any}(),  # attributes
    ""  # id (optional)
)
```

**Key Changes:**
- Use `Relation` instead of `BiomedicalRelation`
- Pass `domain="biomedical"` as the 5th parameter
- All parameters are positional (domain, provenance, evidence, attributes, id)

### 3. Entity Extraction

**Old Code:**
```julia
using GraphMERT

text = "Diabetes is treated with metformin."
options = ProcessingOptions()
entities = extract_entities(text, options)  # Would use biomedical defaults
```

**New Code:**
```julia
using GraphMERT

# Load and register biomedical domain
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Extract entities using domain provider
text = "Diabetes is treated with metformin."
options = ProcessingOptions(domain="biomedical")
entities = extract_entities(bio_domain, text, options)
```

**Key Changes:**
- Explicitly load and register the biomedical domain
- Pass the domain provider to `extract_entities`
- Specify `domain="biomedical"` in `ProcessingOptions`

### 4. Relation Extraction

**Old Code:**
```julia
relations = extract_relations(entities, text, options)  # Would use biomedical defaults
```

**New Code:**
```julia
relations = extract_relations(bio_domain, entities, text, options)
```

**Key Changes:**
- Pass the domain provider as the first parameter

### 5. Knowledge Graph Extraction

**Old Code:**
```julia
text = "Diabetes is treated with metformin."
options = ProcessingOptions()  # Would default to biomedical
graph = extract_knowledge_graph(text, model; options=options)
```

**New Code:**
```julia
# Load and register domain first
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Extract knowledge graph
text = "Diabetes is treated with metformin."
options = ProcessingOptions(domain="biomedical")
graph = extract_knowledge_graph(text, model; options=options)
```

**Key Changes:**
- Explicitly specify `domain="biomedical"` in `ProcessingOptions`
- Ensure domain is registered before extraction

### 6. Seed Injection Functions

**Old Code:**
```julia
config = SeedInjectionConfig()
linked = link_entity_sapbert("diabetes", config)
triples = select_triples_for_entity("C0011849", config)
injected = inject_seed_kg(sequences, seed_kg, config)
```

**New Code:**
```julia
# Load and register domain first
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Use seed injection with domain parameter
config = SeedInjectionConfig()
linked = link_entity_sapbert("diabetes", config, bio_domain)
triples = select_triples_for_entity("C0011849", config, bio_domain)
injected = inject_seed_kg(sequences, seed_kg, config, bio_domain)
```

**Key Changes:**
- Pass domain provider as the last parameter to all seed injection functions
- Functions without domain parameter are deprecated and will emit warnings

### 7. Evaluation Functions

**Old Code:**
```julia
factscore_result = evaluate_factscore(graph, reference)
validity_result = evaluate_validity(graph, reference)
```

**New Code:**
```julia
# Option 1: Let evaluation functions infer domain from graph metadata
factscore_result = evaluate_factscore(graph, reference; domain_name="biomedical", include_domain_metrics=true)
validity_result = evaluate_validity(graph, reference; domain_name="biomedical", include_domain_metrics=true)

# Option 2: Domain is automatically inferred from graph.metadata["domain"]
factscore_result = evaluate_factscore(graph, reference; include_domain_metrics=true)
```

**Key Changes:**
- Pass `domain_name` parameter explicitly (optional if graph has domain metadata)
- Set `include_domain_metrics=true` to get domain-specific metrics

## Common Patterns

### Pattern 1: Setting Up Domain

```julia
using GraphMERT

# Load and register biomedical domain
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Or set as default domain
set_default_domain("biomedical")
```

### Pattern 2: Extracting Knowledge Graph

```julia
# Create processing options
options = ProcessingOptions(
    domain="biomedical",
    max_entities=100,
    max_relations=50,
    # ... other options
)

# Extract knowledge graph
graph = extract_knowledge_graph(text, model; options=options)
```

### Pattern 3: Working with Entities and Relations

```julia
# Entities are now generic Entity objects
for entity in graph.entities
    println("Entity: $(entity.text), Type: $(entity.entity_type), Domain: $(entity.domain)")
    if entity.domain == "biomedical" && haskey(entity.attributes, "cui")
        println("  UMLS CUI: $(entity.attributes["cui"])")
    end
end

# Relations are now generic Relation objects
for relation in graph.relations
    println("Relation: $(relation.head) -[$(relation.relation_type)]-> $(relation.tail)")
    println("  Domain: $(relation.domain), Confidence: $(relation.confidence)")
end
```

### Pattern 4: Accessing Biomedical-Specific Attributes

```julia
# Access biomedical attributes from Entity
entity = graph.entities[1]
if entity.domain == "biomedical"
    cui = get(entity.attributes, "cui", nothing)
    semantic_types = get(entity.attributes, "semantic_types", String[])
    if cui !== nothing
        println("UMLS CUI: $cui")
    end
end
```

## Benefits of Migration

1. **Multi-Domain Support**: Use multiple domains in the same session
2. **Extensibility**: Easily add custom domains
3. **Type Safety**: Generic types work across all domains
4. **Domain-Specific Metrics**: Get domain-specific evaluation metrics automatically
5. **Future-Proof**: New features will be added to the domain system

## Troubleshooting

### Error: "Domain 'biomedical' is not registered"

**Solution:**
```julia
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)
```

### Error: "No default domain set"

**Solution:**
```julia
# Option 1: Register and set default domain
register_domain!("biomedical", bio_domain)
set_default_domain("biomedical")

# Option 2: Pass domain explicitly to functions
extract_entities(bio_domain, text, options)
```

### Warning: Deprecation warnings

**Solution:**
- Follow the migration guide to update deprecated functions
- Warnings will disappear once you use the new API
- Old API will continue to work but may be removed in future versions

## Additional Resources

- **Domain Usage Guide**: See `DOMAIN_USAGE_GUIDE.md` for detailed examples
- **Domain Developer Guide**: See `DOMAIN_DEVELOPER_GUIDE.md` for creating custom domains
- **API Reference**: See `GraphMERT/docs/src/api/domain.md` for complete API documentation

## Questions?

If you encounter issues during migration, please:
1. Check this guide and the domain usage guide
2. Review examples in `examples/biomedical/` and `examples/wikipedia/`
3. Check the API documentation in `GraphMERT/docs/src/api/`
4. Open an issue on GitHub with details about your use case
