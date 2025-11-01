# Domain System Usage Guide

This guide demonstrates how to use the domain-agnostic GraphMERT system with different domain modules.

## Loading Domains

### Biomedical Domain

```julia
using GraphMERT

# Load biomedical domain module
include("GraphMERT/src/domains/biomedical.jl")

# Create and register domain
biomedical_domain = load_biomedical_domain()
register_domain!("biomedical", biomedical_domain)

# Use domain for extraction
text = "Diabetes is treated with metformin."
options = ProcessingOptions(domain="biomedical")
model = create_graphmert_model(GraphMERTConfig())
graph = extract_knowledge_graph(text, model; options=options)
```

### Wikipedia Domain

```julia
using GraphMERT

# Load Wikipedia domain module
include("GraphMERT/src/domains/wikipedia.jl")

# Create and register domain
wikipedia_domain = load_wikipedia_domain()
register_domain!("wikipedia", wikipedia_domain)

# Use domain for extraction
text = "Leonardo da Vinci was born in Vinci, Italy. He painted the Mona Lisa."
options = ProcessingOptions(domain="wikipedia")
model = create_graphmert_model(GraphMERTConfig())
graph = extract_knowledge_graph(text, model; options=options)
```

### Loading Both Domains

```julia
using GraphMERT

# Load both domains
include("GraphMERT/src/domains/biomedical.jl")
include("GraphMERT/src/domains/wikipedia.jl")

# Register both
biomedical_domain = load_biomedical_domain()
wikipedia_domain = load_wikipedia_domain()
register_domain!("biomedical", biomedical_domain)
register_domain!("wikipedia", wikipedia_domain)

# List available domains
println("Available domains: ", list_domains())

# Switch between domains
bio_options = ProcessingOptions(domain="biomedical")
wiki_options = ProcessingOptions(domain="wikipedia")
```

## Domain Methods

Each domain provides:

- `extract_entities(domain, text, config)` - Extract domain-specific entities
- `extract_relations(domain, entities, text, config)` - Extract domain-specific relations
- `validate_entity(domain, entity_text, entity_type, context)` - Validate entities
- `validate_relation(domain, head, relation_type, tail, context)` - Validate relations
- `calculate_entity_confidence(domain, entity_text, entity_type, context)` - Calculate confidence
- `calculate_relation_confidence(domain, head, relation_type, tail, context)` - Calculate confidence
- `create_prompt(domain, task_type, context)` - Generate LLM prompts
- `link_entity(domain, entity_text, config)` - Link to knowledge base (optional)
- `create_seed_triples(domain, entity_text, config)` - Create seed triples (optional)

## Domain-Specific Features

### Biomedical Domain
- Entity types: DISEASE, DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM, etc.
- Relation types: TREATS, CAUSES, ASSOCIATED_WITH, PREVENTS, etc.
- UMLS integration for entity linking
- Biomedical-specific LLM prompts

### Wikipedia Domain
- Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, etc.
- Relation types: BORN_IN, DIED_IN, WORKED_AT, FOUNDED, CREATED_BY, etc.
- Wikidata integration (placeholder for future implementation)
- General knowledge LLM prompts

## Creating Custom Domains

To create a new domain:

1. Create a domain directory: `GraphMERT/src/domains/yourdomain/`
2. Implement `DomainProvider` interface:
   - Create `domain.jl` with your domain struct
   - Implement all required methods
   - Create `entities.jl` for entity extraction
   - Create `relations.jl` for relation extraction
   - Create `prompts.jl` for LLM prompts (optional)
3. Register your domain: `register_domain!("yourdomain", your_domain_instance)`

See `GraphMERT/src/domains/interface.jl` for the complete interface specification.
