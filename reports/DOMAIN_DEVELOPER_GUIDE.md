# Domain Developer Guide

This guide explains how to create custom domain modules for GraphMERT.jl. Domain modules allow you to customize knowledge graph extraction for specific application domains.

## Table of Contents

1. [Overview](#overview)
2. [Domain Provider Interface](#domain-provider-interface)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Required Methods](#required-methods)
5. [Optional Methods](#optional-methods)
6. [Best Practices](#best-practices)
7. [Common Patterns](#common-patterns)
8. [Testing Your Domain](#testing-your-domain)
9. [Troubleshooting](#troubleshooting)

## Overview

The GraphMERT domain system uses a **DomainProvider** interface that allows you to create domain-specific modules. Each domain module encapsulates:

- **Entity Types**: Domain-specific entity categories (e.g., DISEASE, PERSON, LOCATION)
- **Relation Types**: Domain-specific relationship types (e.g., TREATS, BORN_IN, WORKS_FOR)
- **Extraction Logic**: Pattern-based and rule-based entity/relation extraction
- **Validation Rules**: Domain-specific validation of entities and relations
- **Confidence Calculation**: Domain-specific confidence scoring
- **Knowledge Base Integration**: Optional integration with external KBs (UMLS, Wikidata, etc.)
- **LLM Prompts**: Domain-specific prompt generation for LLM assistance

## Domain Provider Interface

All domains must implement the `DomainProvider` abstract type and provide the following methods:

### Required Methods

1. `register_entity_types(domain)` - Return entity type metadata
2. `register_relation_types(domain)` - Return relation type metadata
3. `extract_entities(domain, text, config)` - Extract entities from text
4. `extract_relations(domain, entities, text, config)` - Extract relations between entities
5. `validate_entity(domain, entity_text, entity_type, context)` - Validate entity
6. `validate_relation(domain, head, relation_type, tail, context)` - Validate relation
7. `calculate_entity_confidence(domain, entity_text, entity_type, context)` - Calculate entity confidence
8. `calculate_relation_confidence(domain, head, relation_type, tail, context)` - Calculate relation confidence
9. `get_domain_name(domain)` - Return domain identifier
10. `get_domain_config(domain)` - Return domain configuration

### Optional Methods

1. `link_entity(domain, entity_text, config)` - Link entity to external KB
2. `create_seed_triples(domain, entity_text, config)` - Create seed KG triples
3. `create_evaluation_metrics(domain, kg)` - Create domain-specific metrics
4. `create_prompt(domain, task_type, context)` - Generate LLM prompts

## Step-by-Step Guide

### Step 1: Create Domain Directory Structure

Create a directory structure for your domain:

```
GraphMERT/src/domains/yourdomain/
├── domain.jl          # Main domain provider (required)
├── entities.jl         # Entity extraction logic (required)
├── relations.jl        # Relation extraction logic (required)
├── prompts.jl         # LLM prompt generation (optional)
└── yourdomain.jl      # Loader module (optional)
```

### Step 2: Define Your Domain Struct

Create `domain.jl` with your domain struct:

```julia
"""
YourDomain Domain Provider for GraphMERT.jl

This module implements the DomainProvider interface for the yourdomain domain.
"""

using Dates
using Logging

# Include submodules
include("entities.jl")
include("relations.jl")
include("prompts.jl")  # Optional

"""
    YourDomain

Domain provider for yourdomain knowledge graph extraction.
"""
mutable struct YourDomain <: DomainProvider
    config::DomainConfig
    entity_types::Dict{String, Dict{String, Any}}
    relation_types::Dict{String, Dict{String, Any}}
    # Add any domain-specific fields here
    
    function YourDomain()
        # Initialize entity types
        entity_types = Dict{String, Dict{String, Any}}(
            "ENTITY_TYPE_1" => Dict("domain" => "yourdomain", "category" => "category1"),
            "ENTITY_TYPE_2" => Dict("domain" => "yourdomain", "category" => "category2"),
            # ... more entity types
        )
        
        # Initialize relation types
        relation_types = Dict{String, Dict{String, Any}}(
            "RELATION_TYPE_1" => Dict("domain" => "yourdomain", "category" => "category1"),
            "RELATION_TYPE_2" => Dict("domain" => "yourdomain", "category" => "category2"),
            # ... more relation types
        )
        
        config = DomainConfig(
            "yourdomain";
            entity_types=collect(keys(entity_types)),
            relation_types=collect(keys(relation_types)),
        )
        
        new(config, entity_types, relation_types)
    end
end
```

### Step 3: Implement Required Methods

#### Entity Types Registration

```julia
function register_entity_types(domain::YourDomain)
    return domain.entity_types
end
```

#### Relation Types Registration

```julia
function register_relation_types(domain::YourDomain)
    return domain.relation_types
end
```

#### Entity Extraction

```julia
function extract_entities(domain::YourDomain, text::String, config::Any)
    # Delegate to your entities module
    return extract_yourdomain_entities(text, config, domain)
end
```

#### Relation Extraction

```julia
function extract_relations(domain::YourDomain, entities::Vector{Any}, text::String, config::Any)
    # Delegate to your relations module
    return extract_yourdomain_relations(entities, text, config, domain)
end
```

#### Validation

```julia
function validate_entity(domain::YourDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return validate_yourdomain_entity(entity_text, entity_type, context)
end

function validate_relation(domain::YourDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return validate_yourdomain_relation(head, relation_type, tail, context)
end
```

#### Confidence Calculation

```julia
function calculate_entity_confidence(domain::YourDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return calculate_yourdomain_entity_confidence(entity_text, entity_type, context)
end

function calculate_relation_confidence(domain::YourDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return calculate_yourdomain_relation_confidence(head, relation_type, tail, context)
end
```

#### Domain Name and Config

```julia
function get_domain_name(domain::YourDomain)
    return "yourdomain"
end

function get_domain_config(domain::YourDomain)
    return domain.config
end
```

### Step 4: Implement Entity Extraction (`entities.jl`)

```julia
"""
YourDomain Entity Extraction

This module provides entity extraction functionality for yourdomain.
"""

# Define extraction patterns
const ENTITY_PATTERNS = Dict(
    "ENTITY_TYPE_1" => [r"pattern1", r"pattern2"],
    "ENTITY_TYPE_2" => [r"pattern3", r"pattern4"],
)

"""
    extract_yourdomain_entities(text::String, config::Any, domain::YourDomain)

Extract entities from text using domain-specific patterns.
"""
function extract_yourdomain_entities(text::String, config::Any, domain::YourDomain)
    entities = Vector{Entity}()
    
    # Implement your extraction logic here
    # Example: pattern matching, rule-based extraction, etc.
    
    for (entity_type, patterns) in ENTITY_PATTERNS
        for pattern in patterns
            matches = eachmatch(pattern, text, overlap=false)
            for m in matches
                entity_text = m.match
                confidence = calculate_yourdomain_entity_confidence(entity_text, entity_type, Dict{String, Any}())
                
                entity = Entity(
                    "entity_$(hash(entity_text))",
                    entity_text,
                    entity_type,
                    entity_type,
                    "yourdomain",
                    Dict{String, Any}("entity_type" => entity_type),
                    TextPosition(m.offset, m.offset + length(entity_text) - 1, 1, m.offset),
                    confidence,
                    text,
                )
                push!(entities, entity)
            end
        end
    end
    
    return entities
end

"""
    validate_yourdomain_entity(entity_text::String, entity_type::String, context::Dict{String, Any})

Validate a yourdomain entity.
"""
function validate_yourdomain_entity(entity_text::String, entity_type::String, context::Dict{String, Any})
    # Implement validation logic
    # Return true if valid, false otherwise
    return !isempty(entity_text) && length(entity_text) > 1
end

"""
    calculate_yourdomain_entity_confidence(entity_text::String, entity_type::String, context::Dict{String, Any})

Calculate confidence score for a yourdomain entity.
"""
function calculate_yourdomain_entity_confidence(entity_text::String, entity_type::String, context::Dict{String, Any})
    # Implement confidence calculation
    # Return Float64 between 0.0 and 1.0
    base_confidence = 0.5
    
    # Adjust based on entity characteristics
    if length(entity_text) > 5
        base_confidence += 0.1
    end
    
    return min(base_confidence, 1.0)
end
```

### Step 5: Implement Relation Extraction (`relations.jl`)

```julia
"""
YourDomain Relation Extraction

This module provides relation extraction functionality for yourdomain.
"""

"""
    extract_yourdomain_relations(entities::Vector{Entity}, text::String, config::Any, domain::YourDomain)

Extract relations between entities using domain-specific patterns.
"""
function extract_yourdomain_relations(entities::Vector{Entity}, text::String, config::Any, domain::YourDomain)
    relations = Vector{Relation}()
    
    # Extract relations between entity pairs
    for i in 1:length(entities)
        for j in (i+1):length(entities)
            head_entity = entities[i]
            tail_entity = entities[j]
            
            # Classify relation type
            relation_type = classify_yourdomain_relation(head_entity, tail_entity, text)
            
            if relation_type !== nothing
                # Calculate confidence
                confidence = calculate_yourdomain_relation_confidence(
                    head_entity.text, relation_type, tail_entity.text, Dict{String, Any}()
                )
                
                # Validate relation
                if validate_yourdomain_relation(head_entity.text, relation_type, tail_entity.text, Dict{String, Any}())
                    relation = Relation(
                        head_entity.id,
                        tail_entity.id,
                        relation_type,
                        confidence,
                        Dict{String, Any}("domain" => "yourdomain"),
                        now(),
                    )
                    push!(relations, relation)
                end
            end
        end
    end
    
    return relations
end

"""
    classify_yourdomain_relation(head_entity::Entity, tail_entity::Entity, text::String)

Classify the relation type between two entities.
"""
function classify_yourdomain_relation(head_entity::Entity, tail_entity::Entity, text::String)
    # Implement relation classification logic
    # Return relation type string or nothing
    
    # Example: check for keywords between entities
    head_pos = head_entity.position.start
    tail_pos = tail_entity.position.start
    
    if head_pos < tail_pos
        between_text = text[head_pos:tail_pos]
        # Check for relation keywords
        if occursin(r"treats|cures", lowercase(between_text))
            return "TREATS"
        elseif occursin(r"causes|leads to", lowercase(between_text))
            return "CAUSES"
        end
    end
    
    return nothing  # or return "ASSOCIATED_WITH" as default
end

"""
    validate_yourdomain_relation(head::String, relation_type::String, tail::String, context::Dict{String, Any})

Validate a yourdomain relation.
"""
function validate_yourdomain_relation(head::String, relation_type::String, tail::String, context::Dict{String, Any})
    # Implement validation logic
    # Return true if valid, false otherwise
    return !isempty(head) && !isempty(tail) && !isempty(relation_type)
end

"""
    calculate_yourdomain_relation_confidence(head::String, relation_type::String, tail::String, context::Dict{String, Any})

Calculate confidence score for a yourdomain relation.
"""
function calculate_yourdomain_relation_confidence(head::String, relation_type::String, tail::String, context::Dict{String, Any})
    # Implement confidence calculation
    # Return Float64 between 0.0 and 1.0
    return 0.7  # Example: base confidence
end
```

### Step 6: Create Loader Module (Optional)

Create `yourdomain.jl`:

```julia
"""
YourDomain Domain Module Loader

This module provides a convenient way to load and register yourdomain domain.
"""

# Load the domain module
include("yourdomain/domain.jl")

"""
    load_yourdomain_domain()

Load and return the yourdomain domain instance.
"""
function load_yourdomain_domain()
    return YourDomain()
end

# Export
export YourDomain, load_yourdomain_domain
```

### Step 7: Register and Use Your Domain

```julia
using GraphMERT

# Load your domain
include("GraphMERT/src/domains/yourdomain/yourdomain.jl")

# Create and register domain
your_domain = load_yourdomain_domain()
register_domain!("yourdomain", your_domain)

# Use domain for extraction
text = "Your domain-specific text here."
options = ProcessingOptions(domain="yourdomain")
model = create_graphmert_model(GraphMERTConfig())
graph = extract_knowledge_graph(text, model; options=options)
```

## Required Methods

### register_entity_types

**Signature**: `register_entity_types(domain::DomainProvider) -> Dict{String, Dict{String, Any}}`

**Purpose**: Return a dictionary mapping entity type names to their metadata.

**Example**:
```julia
function register_entity_types(domain::YourDomain)
    return domain.entity_types
end
```

**Metadata Structure**:
```julia
Dict(
    "ENTITY_TYPE" => Dict(
        "domain" => "yourdomain",
        "category" => "category_name",
        # Add any other metadata
    )
)
```

### register_relation_types

**Signature**: `register_relation_types(domain::DomainProvider) -> Dict{String, Dict{String, Any}}`

**Purpose**: Return a dictionary mapping relation type names to their metadata.

**Example**:
```julia
function register_relation_types(domain::YourDomain)
    return domain.relation_types
end
```

### extract_entities

**Signature**: `extract_entities(domain::DomainProvider, text::String, config::ProcessingOptions) -> Vector{Entity}`

**Purpose**: Extract entities from text using domain-specific patterns and rules.

**Returns**: Vector of `Entity` objects with:
- `id`: Unique identifier
- `text`: Entity text
- `label`: Entity label (same as entity_type for simplicity)
- `entity_type`: Entity type string
- `domain`: Domain identifier ("yourdomain")
- `attributes`: Dictionary of domain-specific attributes
- `position`: TextPosition with location in text
- `confidence`: Confidence score (0.0 to 1.0)
- `provenance`: Source text

**Example**:
```julia
function extract_entities(domain::YourDomain, text::String, config::Any)
    entities = Vector{Entity}()
    # ... extraction logic ...
    return entities
end
```

### extract_relations

**Signature**: `extract_relations(domain::DomainProvider, entities::Vector{Entity}, text::String, config::ProcessingOptions) -> Vector{Relation}`

**Purpose**: Extract relations between entities using domain-specific patterns.

**Returns**: Vector of `Relation` objects with:
- `head`: Entity ID of head entity
- `tail`: Entity ID of tail entity
- `relation_type`: Relation type string
- `confidence`: Confidence score (0.0 to 1.0)
- `attributes`: Dictionary of domain-specific attributes
- `created_at`: DateTime of creation

**Example**:
```julia
function extract_relations(domain::YourDomain, entities::Vector{Any}, text::String, config::Any)
    relations = Vector{Relation}()
    # ... relation extraction logic ...
    return relations
end
```

### validate_entity

**Signature**: `validate_entity(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any}) -> Bool`

**Purpose**: Validate that an entity text matches its claimed type according to domain rules.

**Returns**: `true` if valid, `false` otherwise.

**Example**:
```julia
function validate_entity(domain::YourDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Check entity text against entity_type requirements
    return is_valid
end
```

### validate_relation

**Signature**: `validate_relation(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any}) -> Bool`

**Purpose**: Validate that a relation is valid according to domain rules.

**Returns**: `true` if valid, `false` otherwise.

**Example**:
```julia
function validate_relation(domain::YourDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    # Check if relation type is compatible with head and tail entities
    return is_valid
end
```

### calculate_entity_confidence

**Signature**: `calculate_entity_confidence(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any}) -> Float64`

**Purpose**: Calculate confidence score for an entity based on domain-specific rules.

**Returns**: Float64 between 0.0 and 1.0.

**Example**:
```julia
function calculate_entity_confidence(domain::YourDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    confidence = 0.5  # Base confidence
    # Adjust based on entity characteristics
    return min(confidence, 1.0)
end
```

### calculate_relation_confidence

**Signature**: `calculate_relation_confidence(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any}) -> Float64`

**Purpose**: Calculate confidence score for a relation based on domain-specific rules.

**Returns**: Float64 between 0.0 and 1.0.

**Example**:
```julia
function calculate_relation_confidence(domain::YourDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    confidence = 0.5  # Base confidence
    # Adjust based on relation characteristics
    return min(confidence, 1.0)
end
```

### get_domain_name

**Signature**: `get_domain_name(domain::DomainProvider) -> String`

**Purpose**: Return the domain identifier/name.

**Returns**: Domain name string (e.g., "yourdomain").

**Example**:
```julia
function get_domain_name(domain::YourDomain)
    return "yourdomain"
end
```

### get_domain_config

**Signature**: `get_domain_config(domain::DomainProvider) -> DomainConfig`

**Purpose**: Return the domain configuration.

**Returns**: `DomainConfig` object.

**Example**:
```julia
function get_domain_config(domain::YourDomain)
    return domain.config
end
```

## Optional Methods

### link_entity

**Signature**: `link_entity(domain::DomainProvider, entity_text::String, config::Any) -> Union{Dict, Nothing}`

**Purpose**: Link entity to external knowledge base (e.g., UMLS, Wikidata).

**Returns**: Dict with `:candidates` or `:candidate` key, or `nothing` if not supported.

**Example**:
```julia
function link_entity(domain::YourDomain, entity_text::String, config::Any)
    # Link to external KB
    # Return Dict(:candidate => Dict(:kb_id => "...", :name => "...", ...))
    # or Dict(:candidates => [...])
    return nothing  # If not supported
end
```

### create_seed_triples

**Signature**: `create_seed_triples(domain::DomainProvider, entity_text::String, config::Any) -> Vector{Any}`

**Purpose**: Create seed KG triples for an entity from domain-specific knowledge base.

**Returns**: Vector of `SemanticTriple` objects or Dicts.

**Example**:
```julia
function create_seed_triples(domain::YourDomain, entity_text::String, config::Any)
    # Query external KB for entity relations
    # Return Vector of SemanticTriple objects
    return Vector{Any}()  # If not supported
end
```

### create_evaluation_metrics

**Signature**: `create_evaluation_metrics(domain::DomainProvider, kg::KnowledgeGraph) -> Dict{String, Any}`

**Purpose**: Create domain-specific evaluation metrics.

**Returns**: Dict with metric names and values.

**Example**:
```julia
function create_evaluation_metrics(domain::YourDomain, kg::Any)
    metrics = Dict{String, Any}()
    # Calculate domain-specific metrics
    metrics["total_entities"] = length(kg.entities)
    metrics["total_relations"] = length(kg.relations)
    # ... more metrics ...
    return metrics
end
```

### create_prompt

**Signature**: `create_prompt(domain::DomainProvider, task_type::Symbol, context::Dict{String, Any}) -> String`

**Purpose**: Generate LLM prompt for domain-specific task.

**Task Types**: `:entity_discovery`, `:relation_matching`, `:tail_formation`

**Returns**: String prompt.

**Example**:
```julia
function create_prompt(domain::YourDomain, task_type::Symbol, context::Dict{String, Any})
    if task_type == :entity_discovery
        return "Extract yourdomain entities from: $(context["text"])"
    elseif task_type == :relation_matching
        return "Find relations between: $(context["head"]) and $(context["tail"])"
    else
        error("Unknown task type: $task_type")
    end
end
```

## Best Practices

### 1. Keep Domains Self-Contained

Each domain module should be self-contained:
- Include all entity/relation types within the domain module
- Don't depend on other domain modules
- Use generic types (`Entity`, `Relation`) rather than domain-specific types

### 2. Use Consistent Naming Conventions

- Domain name: lowercase (e.g., "yourdomain")
- Entity types: UPPERCASE (e.g., "ENTITY_TYPE")
- Relation types: UPPERCASE (e.g., "RELATION_TYPE")
- Function names: lowercase_with_underscores (e.g., `extract_yourdomain_entities`)

### 3. Implement Robust Validation

- Make validation lenient enough to accept valid entities
- Use pattern matching for strict validation
- Provide fallback validation for edge cases
- Consider context when validating

### 4. Provide Meaningful Confidence Scores

- Base confidence on multiple factors:
  - Pattern match quality
  - Entity text characteristics
  - Contextual clues
  - Domain-specific heuristics
- Return scores between 0.0 and 1.0
- Use consistent scoring across similar entities/relations

### 5. Handle Edge Cases Gracefully

- Empty input: Return empty vectors
- Invalid entity types: Return empty or use fallback
- Missing context: Use defaults or skip validation
- External KB failures: Return `nothing` or empty results

### 6. Document Your Domain

- Add docstrings to all public functions
- Document entity and relation types
- Provide usage examples
- Explain domain-specific logic

### 7. Test Thoroughly

- Test entity extraction with various texts
- Test relation extraction with different entity pairs
- Test validation with edge cases
- Test confidence calculation consistency
- Test error handling

## Common Patterns

### Pattern 1: Regex-Based Entity Extraction

```julia
function extract_entities_with_regex(text::String, pattern::Regex, entity_type::String)
    entities = Vector{Entity}()
    for m in eachmatch(pattern, text, overlap=false)
        entity_text = m.match
        confidence = calculate_confidence(entity_text, entity_type)
        entity = Entity(
            "entity_$(hash(entity_text))",
            entity_text,
            entity_type,
            entity_type,
            "yourdomain",
            Dict{String, Any}(),
            TextPosition(m.offset, m.offset + length(entity_text) - 1, 1, m.offset),
            confidence,
            text,
        )
        push!(entities, entity)
    end
    return entities
end
```

### Pattern 2: Context-Based Relation Classification

```julia
function classify_relation_by_context(head_entity::Entity, tail_entity::Entity, text::String)
    # Extract text between entities
    start_pos = min(head_entity.position.start, tail_entity.position.start)
    end_pos = max(head_entity.position.stop, tail_entity.position.stop)
    between_text = text[start_pos:end_pos]
    
    # Check for relation keywords
    relation_keywords = Dict(
        "TREATS" => ["treats", "cures", "medication"],
        "CAUSES" => ["causes", "leads to", "results in"],
    )
    
    for (relation_type, keywords) in relation_keywords
        for keyword in keywords
            if occursin(keyword, lowercase(between_text))
                return relation_type
            end
        end
    end
    
    return nothing  # or "ASSOCIATED_WITH" as default
end
```

### Pattern 3: Confidence Calculation Based on Multiple Factors

```julia
function calculate_confidence(entity_text::String, entity_type::String, context::Dict{String, Any})
    confidence = 0.5  # Base confidence
    
    # Factor 1: Text length
    if length(entity_text) > 5
        confidence += 0.1
    end
    
    # Factor 2: Capitalization
    if isuppercase(entity_text[1])
        confidence += 0.1
    end
    
    # Factor 3: Context clues
    if haskey(context, "is_in_list") && context["is_in_list"]
        confidence += 0.1
    end
    
    # Factor 4: Pattern match quality
    if haskey(context, "pattern_match_score")
        confidence += context["pattern_match_score"] * 0.2
    end
    
    return min(confidence, 1.0)
end
```

### Pattern 4: Knowledge Base Integration

```julia
function link_entity_to_kb(domain::YourDomain, entity_text::String, kb_client)
    if domain.kb_client === nothing
        return nothing
    end
    
    # Query KB
    results = query_kb(kb_client, entity_text)
    
    if isempty(results)
        return nothing
    end
    
    # Convert to expected format
    candidates = []
    for result in results
        push!(candidates, Dict(
            :kb_id => result.id,
            :name => result.name,
            :score => result.similarity,
            :source => "YourKB",
        ))
    end
    
    return Dict(:candidates => candidates)
end
```

## Testing Your Domain

### Basic Test Structure

```julia
using Test
using GraphMERT

include("GraphMERT/src/domains/yourdomain/yourdomain.jl")

@testset "YourDomain Tests" begin
    # Load domain
    domain = load_yourdomain_domain()
    register_domain!("yourdomain", domain)
    
    # Test entity extraction
    @testset "Entity Extraction" begin
        text = "Sample text with entities."
        options = ProcessingOptions(domain="yourdomain")
        entities = extract_entities(domain, text, options)
        
        @test isa(entities, Vector{Entity})
        @test length(entities) > 0
    end
    
    # Test relation extraction
    @testset "Relation Extraction" begin
        text = "Entity1 relates to Entity2."
        options = ProcessingOptions(domain="yourdomain")
        entities = extract_entities(domain, text, options)
        relations = extract_relations(domain, entities, text, options)
        
        @test isa(relations, Vector{Relation})
    end
    
    # Test validation
    @testset "Validation" begin
        @test validate_entity(domain, "ValidEntity", "ENTITY_TYPE", Dict{String, Any}())
        @test !validate_entity(domain, "", "ENTITY_TYPE", Dict{String, Any}())
    end
    
    # Test confidence calculation
    @testset "Confidence Calculation" begin
        confidence = calculate_entity_confidence(domain, "Entity", "ENTITY_TYPE", Dict{String, Any}())
        @test 0.0 <= confidence <= 1.0
    end
end
```

## Troubleshooting

### Problem: "Domain 'yourdomain' is not registered"

**Solution**: Make sure you've registered your domain:
```julia
register_domain!("yourdomain", your_domain_instance)
```

### Problem: Entity extraction returns empty results

**Check**:
- Are your patterns correct?
- Is the text matching your patterns?
- Are entity types registered correctly?
- Try printing intermediate results for debugging

### Problem: Relations not extracted

**Check**:
- Are entities being extracted correctly?
- Is relation classification logic working?
- Are validation rules too strict?
- Check entity positions in text

### Problem: Validation always returns false

**Check**:
- Are validation rules too strict?
- Try making validation more lenient for testing
- Check if entity types match registered types

### Problem: Confidence scores seem incorrect

**Check**:
- Are you returning values between 0.0 and 1.0?
- Are you handling edge cases correctly?
- Try normalizing confidence scores

## Example: Complete Minimal Domain

Here's a complete minimal domain implementation:

```julia
# domain.jl
using Dates
using Logging

mutable struct MinimalDomain <: DomainProvider
    config::DomainConfig
    entity_types::Dict{String, Dict{String, Any}}
    relation_types::Dict{String, Dict{String, Any}}
    
    function MinimalDomain()
        entity_types = Dict("ENTITY" => Dict("domain" => "minimal"))
        relation_types = Dict("RELATES_TO" => Dict("domain" => "minimal"))
        config = DomainConfig("minimal"; entity_types=["ENTITY"], relation_types=["RELATES_TO"])
        new(config, entity_types, relation_types)
    end
end

function register_entity_types(domain::MinimalDomain)
    return domain.entity_types
end

function register_relation_types(domain::MinimalDomain)
    return domain.relation_types
end

function extract_entities(domain::MinimalDomain, text::String, config::Any)
    entities = Vector{Entity}()
    # Simple extraction: find capitalized words
    for m in eachmatch(r"\b[A-Z][a-z]+\b", text)
        entity = Entity(
            "entity_$(hash(m.match))",
            m.match,
            "ENTITY",
            "ENTITY",
            "minimal",
            Dict{String, Any}(),
            TextPosition(m.offset, m.offset + length(m.match) - 1, 1, m.offset),
            0.7,
            text,
        )
        push!(entities, entity)
    end
    return entities
end

function extract_relations(domain::MinimalDomain, entities::Vector{Any}, text::String, config::Any)
    relations = Vector{Relation}()
    for i in 1:length(entities), j in (i+1):length(entities)
        rel = Relation(entities[i].id, entities[j].id, "RELATES_TO", 0.7, Dict{String, Any}(), now())
        push!(relations, rel)
    end
    return relations
end

function validate_entity(domain::MinimalDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return !isempty(entity_text)
end

function validate_relation(domain::MinimalDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return !isempty(head) && !isempty(tail)
end

function calculate_entity_confidence(domain::MinimalDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return 0.7
end

function calculate_relation_confidence(domain::MinimalDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return 0.7
end

function get_domain_name(domain::MinimalDomain)
    return "minimal"
end

function get_domain_config(domain::MinimalDomain)
    return domain.config
end
```

## Reference Implementations

For complete reference implementations, see:

- **Biomedical Domain**: `GraphMERT/src/domains/biomedical/`
- **Wikipedia Domain**: `GraphMERT/src/domains/wikipedia/`

These implementations demonstrate:
- Complex entity/relation extraction
- Pattern-based validation
- Knowledge base integration (UMLS, Wikidata)
- Domain-specific evaluation metrics
- LLM prompt generation

## Additional Resources

- **Domain Interface**: `GraphMERT/src/domains/interface.jl`
- **Domain Registry**: `GraphMERT/src/domains/registry.jl`
- **Usage Guide**: `DOMAIN_USAGE_GUIDE.md`
- **Examples**: `examples/biomedical/`, `examples/wikipedia/`

## Getting Help

If you encounter issues or need help:

1. Check the reference implementations (biomedical, Wikipedia)
2. Review the domain interface documentation
3. Look at existing test files for examples
4. Check error messages for hints
5. Ensure all required methods are implemented

Happy domain developing!
