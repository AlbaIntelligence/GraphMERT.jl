# Domain API Reference

This page documents the domain system API for GraphMERT.jl. The domain system allows you to customize knowledge graph extraction for different application domains.

## Domain System Overview

GraphMERT.jl uses a pluggable domain system where domain-specific logic is encapsulated in `DomainProvider` implementations. Each domain provides entity extraction, relation extraction, validation, and confidence calculation tailored to its domain.

## Domain Registry

### `register_domain!`

```julia
register_domain!(domain_name::String, provider::DomainProvider)
```

Register a domain provider in the global registry.

**Parameters:**
- `domain_name::String` - Domain identifier (e.g., "biomedical", "wikipedia")
- `provider::DomainProvider` - Domain provider instance

**Example:**
```julia
using GraphMERT
include("GraphMERT/src/domains/biomedical.jl")

bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)
```

### `get_domain`

```julia
get_domain(domain_name::String) -> Union{DomainProvider, Nothing}
```

Get a domain provider from the registry.

**Parameters:**
- `domain_name::String` - Domain identifier

**Returns:**
- Domain provider instance or `nothing` if not found

**Example:**
```julia
domain = get_domain("biomedical")
if domain !== nothing
    entities = extract_entities(domain, text, options)
end
```

### `list_domains`

```julia
list_domains() -> Vector{String}
```

List all registered domain names.

**Returns:**
- Vector of domain name strings

**Example:**
```julia
domains = list_domains()
println("Available domains: ", join(domains, ", "))
```

### `set_default_domain`

```julia
set_default_domain(domain_name::String)
```

Set the default domain for operations.

**Parameters:**
- `domain_name::String` - Domain identifier to set as default

**Example:**
```julia
set_default_domain("biomedical")
```

### `get_default_domain`

```julia
get_default_domain() -> Union{DomainProvider, Nothing}
```

Get the default domain provider.

**Returns:**
- Default domain provider instance or `nothing` if no default is set

**Example:**
```julia
domain = get_default_domain()
if domain !== nothing
    println("Default domain: ", get_domain_name(domain))
end
```

### `has_domain`

```julia
has_domain(domain_name::String) -> Bool
```

Check if a domain is registered.

**Parameters:**
- `domain_name::String` - Domain identifier

**Returns:**
- `true` if domain is registered, `false` otherwise

**Example:**
```julia
if has_domain("biomedical")
    println("Biomedical domain is available")
end
```

## Domain Provider Interface

All domain providers implement the `DomainProvider` abstract type and provide the following methods:

### Required Methods

#### `register_entity_types`

```julia
register_entity_types(domain::DomainProvider) -> Dict{String, Dict{String, Any}}
```

Register entity types for this domain.

**Returns:**
- Dictionary mapping entity type names to metadata

**Example:**
```julia
entity_types = register_entity_types(domain)
for (type_name, metadata) in entity_types
    println("Entity type: $type_name")
end
```

#### `register_relation_types`

```julia
register_relation_types(domain::DomainProvider) -> Dict{String, Dict{String, Any}}
```

Register relation types for this domain.

**Returns:**
- Dictionary mapping relation type names to metadata

**Example:**
```julia
relation_types = register_relation_types(domain)
for (type_name, metadata) in relation_types
    println("Relation type: $type_name")
end
```

#### `extract_entities`

```julia
extract_entities(domain::DomainProvider, text::String, config::ProcessingOptions) -> Vector{Entity}
```

Extract entities from text using domain-specific patterns and rules.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `text::String` - Input text
- `config::ProcessingOptions` - Processing options

**Returns:**
- Vector of `Entity` objects

**Example:**
```julia
text = "Diabetes is treated with metformin."
options = ProcessingOptions(domain="biomedical")
entities = extract_entities(domain, text, options)

for entity in entities
    println("$(entity.text) [$(entity.entity_type)] (confidence: $(entity.confidence))")
end
```

#### `extract_relations`

```julia
extract_relations(domain::DomainProvider, entities::Vector{Entity}, text::String, config::ProcessingOptions) -> Vector{Relation}
```

Extract relations between entities using domain-specific patterns and rules.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `entities::Vector{Entity}` - Extracted entities
- `text::String` - Input text
- `config::ProcessingOptions` - Processing options

**Returns:**
- Vector of `Relation` objects

**Example:**
```julia
entities = extract_entities(domain, text, options)
relations = extract_relations(domain, entities, text, options)

for relation in relations
    head_entity = find_entity_by_id(entities, relation.head)
    tail_entity = find_entity_by_id(entities, relation.tail)
    println("$(head_entity.text) --[$(relation.relation_type)]--> $(tail_entity.text)")
end
```

#### `validate_entity`

```julia
validate_entity(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any}) -> Bool
```

Validate that an entity text matches its claimed type according to domain rules.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `entity_text::String` - Entity text to validate
- `entity_type::String` - Claimed entity type
- `context::Dict{String, Any}` - Additional context (optional)

**Returns:**
- `true` if valid, `false` otherwise

**Example:**
```julia
is_valid = validate_entity(domain, "metformin", "DRUG", Dict{String, Any}())
```

#### `validate_relation`

```julia
validate_relation(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any}) -> Bool
```

Validate that a relation is valid according to domain rules.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `head::String` - Head entity text
- `relation_type::String` - Relation type
- `tail::String` - Tail entity text
- `context::Dict{String, Any}` - Additional context (optional)

**Returns:**
- `true` if valid, `false` otherwise

**Example:**
```julia
is_valid = validate_relation(domain, "metformin", "TREATS", "diabetes", Dict{String, Any}())
```

#### `calculate_entity_confidence`

```julia
calculate_entity_confidence(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any}) -> Float64
```

Calculate confidence score for an entity based on domain-specific rules.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `entity_text::String` - Entity text
- `entity_type::String` - Entity type
- `context::Dict{String, Any}` - Additional context (optional)

**Returns:**
- Float64 between 0.0 and 1.0

**Example:**
```julia
confidence = calculate_entity_confidence(domain, "metformin", "DRUG", Dict{String, Any}())
```

#### `calculate_relation_confidence`

```julia
calculate_relation_confidence(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any}) -> Float64
```

Calculate confidence score for a relation based on domain-specific rules.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `head::String` - Head entity text
- `relation_type::String` - Relation type
- `tail::String` - Tail entity text
- `context::Dict{String, Any}` - Additional context (optional)

**Returns:**
- Float64 between 0.0 and 1.0

**Example:**
```julia
confidence = calculate_relation_confidence(domain, "metformin", "TREATS", "diabetes", Dict{String, Any}())
```

#### `get_domain_name`

```julia
get_domain_name(domain::DomainProvider) -> String
```

Get the name/identifier of this domain.

**Returns:**
- Domain name string

**Example:**
```julia
domain_name = get_domain_name(domain)
println("Using domain: $domain_name")
```

#### `get_domain_config`

```julia
get_domain_config(domain::DomainProvider) -> DomainConfig
```

Get the configuration for this domain.

**Returns:**
- `DomainConfig` object

**Example:**
```julia
config = get_domain_config(domain)
println("Entity types: ", config.entity_types)
println("Relation types: ", config.relation_types)
```

### Optional Methods

#### `link_entity`

```julia
link_entity(domain::DomainProvider, entity_text::String, config::Any) -> Union{Dict, Nothing}
```

Link entity to external knowledge base (e.g., UMLS for biomedical, Wikidata for Wikipedia).

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `entity_text::String` - Entity text to link
- `config::Any` - Configuration (can be SeedInjectionConfig or ProcessingOptions)

**Returns:**
- Dict with `:candidates` or `:candidate` key, or `nothing` if not supported

**Example:**
```julia
linking_result = link_entity(domain, "metformin", config)
if linking_result !== nothing
    if haskey(linking_result, :candidate)
        candidate = linking_result[:candidate]
        println("Linked to KB: $(candidate[:kb_id])")
    end
end
```

#### `create_seed_triples`

```julia
create_seed_triples(domain::DomainProvider, entity_text::String, config::Any) -> Vector{Any}
```

Create seed KG triples for an entity from domain-specific knowledge base.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `entity_text::String` - Entity text or KB ID
- `config::Any` - Configuration

**Returns:**
- Vector of `SemanticTriple` objects or Dicts

**Example:**
```julia
triples = create_seed_triples(domain, "metformin", config)
for triple in triples
    println("$(triple.head) --[$(triple.relation)]--> $(triple.tail)")
end
```

#### `create_evaluation_metrics`

```julia
create_evaluation_metrics(domain::DomainProvider, kg::KnowledgeGraph) -> Dict{String, Any}
```

Create domain-specific evaluation metrics.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `kg::KnowledgeGraph` - Knowledge graph to evaluate

**Returns:**
- Dictionary with metric names and values

**Example:**
```julia
metrics = create_evaluation_metrics(domain, kg)
println("Total entities: ", metrics["total_entities"])
println("Linking coverage: ", metrics["umls_linking_coverage"])
```

#### `create_prompt`

```julia
create_prompt(domain::DomainProvider, task_type::Symbol, context::Dict{String, Any}) -> String
```

Generate LLM prompt for domain-specific task.

**Parameters:**
- `domain::DomainProvider` - Domain provider instance
- `task_type::Symbol` - Task type (`:entity_discovery`, `:relation_matching`, `:tail_formation`)
- `context::Dict{String, Any}` - Context dictionary

**Returns:**
- String prompt

**Example:**
```julia
context = Dict("text" => "Diabetes is treated with metformin.")
prompt = create_prompt(domain, :entity_discovery, context)
```

## Domain Configuration

### `DomainConfig`

```julia
struct DomainConfig
    name::String
    entity_types::Vector{String}
    relation_types::Vector{String}
    validation_rules::Dict{String, Any}
    extraction_patterns::Dict{String, Any}
    confidence_strategies::Dict{String, Any}
end
```

Configuration structure for a domain provider.

**Fields:**
- `name::String` - Domain name
- `entity_types::Vector{String}` - List of entity type names
- `relation_types::Vector{String}` - List of relation type names
- `validation_rules::Dict{String, Any}` - Domain-specific validation rules
- `extraction_patterns::Dict{String, Any}` - Domain-specific extraction patterns
- `confidence_strategies::Dict{String, Any}` - Domain-specific confidence calculation strategies

### `ProcessingOptions` (Updated for Domain System)

```julia
struct ProcessingOptions
    domain::String                    # Domain identifier (e.g., "biomedical", "wikipedia")
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

Processing options with domain specification.

**Key Field:**
- `domain::String` - Domain identifier. Must match a registered domain name.

**Example:**
```julia
# Biomedical domain
options_bio = ProcessingOptions(domain="biomedical", confidence_threshold=0.8)

# Wikipedia domain
options_wiki = ProcessingOptions(domain="wikipedia", confidence_threshold=0.7)
```

## Domain Loading Functions

### Biomedical Domain

```julia
load_biomedical_domain(umls_client::Union{Any, Nothing} = nothing) -> BiomedicalDomain
```

Load and return the biomedical domain instance.

**Parameters:**
- `umls_client::Union{Any, Nothing}` - Optional UMLS client for entity linking

**Returns:**
- `BiomedicalDomain` instance

**Example:**
```julia
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)
```

### Wikipedia Domain

```julia
load_wikipedia_domain(wikidata_client::Union{Any, Nothing} = nothing) -> WikipediaDomain
```

Load and return the Wikipedia domain instance.

**Parameters:**
- `wikidata_client::Union{Any, Nothing}` - Optional Wikidata client for entity linking

**Returns:**
- `WikipediaDomain` instance

**Example:**
```julia
include("GraphMERT/src/domains/wikipedia.jl")
wiki_domain = load_wikipedia_domain()
register_domain!("wikipedia", wiki_domain)
```

## Complete Usage Example

```julia
using GraphMERT

# Load domains
include("GraphMERT/src/domains/biomedical.jl")
include("GraphMERT/src/domains/wikipedia.jl")

# Register domains
bio_domain = load_biomedical_domain()
wiki_domain = load_wikipedia_domain()
register_domain!("biomedical", bio_domain)
register_domain!("wikipedia", wiki_domain)

# List available domains
println("Available domains: ", list_domains())

# Extract with biomedical domain
text_bio = "Diabetes is treated with metformin."
options_bio = ProcessingOptions(domain="biomedical", confidence_threshold=0.8)
model = create_graphmert_model(GraphMERTConfig())
graph_bio = extract_knowledge_graph(text_bio, model; options=options_bio)

# Extract with Wikipedia domain
text_wiki = "Leonardo da Vinci was born in Vinci, Italy."
options_wiki = ProcessingOptions(domain="wikipedia", confidence_threshold=0.7)
graph_wiki = extract_knowledge_graph(text_wiki, model; options=options_wiki)

# Get domain-specific metrics
bio_metrics = create_evaluation_metrics(bio_domain, graph_bio)
wiki_metrics = create_evaluation_metrics(wiki_domain, graph_wiki)
```

## Domain-Specific Data Structures

### Generic `Entity` (Domain-Agnostic)

```julia
struct Entity
    id::String
    text::String
    label::String
    entity_type::String      # Domain-specific entity type
    domain::String           # Domain identifier
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

### Generic `Relation` (Domain-Agnostic)

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

## Error Handling

### Domain Not Registered Error

When a domain is not registered, you'll receive a helpful error message:

```
Domain 'yourdomain' is not registered.
Available domains: biomedical, wikipedia

To register a domain, use:
  include("GraphMERT/src/domains/yourdomain.jl")
  domain = load_yourdomain_domain()
  register_domain!("yourdomain", domain)
```

## See Also

- [Domain Usage Guide](DOMAIN_USAGE_GUIDE.md) - User guide for using domains
- [Domain Developer Guide](DOMAIN_DEVELOPER_GUIDE.md) - Guide for creating custom domains
- [Core API Reference](api/core.md) - Core GraphMERT API
