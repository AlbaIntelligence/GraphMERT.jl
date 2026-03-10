"""
Domain Interface for GraphMERT.jl

This module defines the abstract interface that all domain implementations must provide.
Domains are pluggable modules that define domain-specific behavior for entity extraction,
relation extraction, validation, confidence calculation, and other domain-specific operations.
"""

using Dates
using Logging

# ============================================================================
# Domain Provider Abstract Type
# ============================================================================

"""
    DomainProvider

Abstract type that all domain implementations must subtype.
"""
abstract type DomainProvider end

"""
    DomainConfig

Configuration for a domain provider.
"""
struct DomainConfig
    name::String
    entity_types::Vector{String}
    relation_types::Vector{String}
    validation_rules::Dict{String, Any}
    extraction_patterns::Dict{String, Any}
    confidence_strategies::Dict{String, Any}
    
    function DomainConfig(
        name::String;
        entity_types::Vector{String} = String[],
        relation_types::Vector{String} = String[],
        validation_rules::Dict{String, Any} = Dict{String, Any}(),
        extraction_patterns::Dict{String, Any} = Dict{String, Any}(),
        confidence_strategies::Dict{String, Any} = Dict{String, Any}(),
    )
        new(name, entity_types, relation_types, validation_rules, extraction_patterns, confidence_strategies)
    end
end

# ============================================================================
# Required Methods (must be implemented by all domains)
# ============================================================================

"""
    register_entity_types(domain::DomainProvider)

Register entity types for this domain. Must return a Dict mapping entity type names
to their metadata.
"""
function register_entity_types(domain::DomainProvider)
    error("register_entity_types must be implemented by domain: $(typeof(domain))")
end

"""
    register_relation_types(domain::DomainProvider)

Register relation types for this domain. Must return a Dict mapping relation type names
to their metadata.
"""
function register_relation_types(domain::DomainProvider)
    error("register_relation_types must be implemented by domain: $(typeof(domain))")
end

"""
    extract_entities(domain::DomainProvider, text::String, config::ProcessingOptions)

Extract entities from text using domain-specific patterns and rules.
Returns a Vector of Entity objects.

Default generic implementation uses simple regex patterns for common entity types.
"""
function extract_entities(domain::DomainProvider, text::String, config::Any)
    # Generic fallback implementation using regex patterns
    entities = Entity[]
    entity_id = 1
    
    # Common entity patterns (language-agnostic)
    patterns = [
        ("PERSON", r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"),
        ("LOCATION", r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:City|Town|Village|State|Country|Kingdom|Empire|Palace|Castle)\b"),
        ("ORGANIZATION", r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:University|College|Corporation|Company|Foundation|Institute|Museum)\b"),
        ("DATE", r"\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"),
        ("NUMBER", r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:years?|days?|months?|centuries?)?\b"),
    ]
    
    for (entity_type, pattern) in patterns
        for match in eachmatch(pattern, text)
            entity_text = String(match.match)
            
            # Skip if too short
            length(entity_text) < 2 && continue
            
            # Calculate confidence based on pattern specificity
            confidence = 0.5 + 0.3 * rand() # 0.5-0.8 range
            
            # Check if entity already found
            if !any(e -> e.text == entity_text, entities)
                push!(entities, Entity(
                    "entity_$entity_id",
                    entity_text,
                    entity_text,  # label
                    entity_type,
                    "generic",
                    Dict{String, Any}(),
                    TextPosition(0, 0, 0, 0),
                    confidence,
                    text
                ))
                entity_id += 1
            end
        end
    end
    
    return entities
end

"""
    extract_relations(domain::DomainProvider, entities::Vector{Entity}, text::String, config::ProcessingOptions)

Extract relations between entities using domain-specific patterns and rules.
Returns a Vector of Relation objects.

Default generic implementation uses co-occurrence and simple pattern matching.
"""
function extract_relations(domain::DomainProvider, entities::Vector{Entity}, text::String, config::Any)
    relations = Relation[]
    relation_id = 1
    
    # Simple relation patterns
    relation_patterns = [
        (r"(\w+)\s+(?:was|is)\s+(?:the\s+)?(\w+)\s+of\s+(\w+)", "IS_OF"),
        (r"(\w+)\s+(?:married|married\s+to)\s+(\w+)", "MARRIED_TO"),
        (r"(\w+)\s+(?:was|is)\s+(?:born\s+in|died\s+in)\s+(\w+)", "LOCATED_IN"),
        (r"(\w+)\s+(?:reigned\s+(?:from|until)|ruled)\s+(\w+)", "REIGNED"),
        (r"(\w+)\s+(?:was|is)\s+(?:the\s+)?(?:father|mother|son|daughter|parent)\s+of\s+(\w+)", "PARENT_OF"),
    ]
    
    # Co-occurrence based relations (entities appearing in same sentence)
    sentences = split(text, r"[.!?]")
    
    for sentence in sentences
        sentence_entities = filter(e -> occursin(e.text, sentence), entities)
        
        if length(sentence_entities) >= 2
            # Create co-occurrence relation between first two entities
            head = sentence_entities[1]
            tail = sentence_entities[2]
            
            # Check if relation already exists
            if !any(r -> r.head == head.id && r.tail == tail.id, relations)
                # Try to find a relation pattern
                relation_type = "ASSOCIATED_WITH"
                confidence = 0.5
                
                for (pattern, rel_type) in relation_patterns
                    m = match(pattern, sentence)
                    if m !== nothing
                        relation_type = rel_type
                        confidence = 0.7
                        break
                    end
                end
                
                push!(relations, Relation(
                    head.id,
                    tail.id,
                    relation_type,
                    confidence,
                    "generic",
                    String(sentence),  # Convert SubString to String
                    "",
                    Dict{String, Any}(),
                    "relation_$relation_id"
                ))
                relation_id += 1
            end
        end
    end
    
    return relations
end

"""
    validate_entity(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict)

Validate that an entity text matches its claimed type according to domain rules.
Returns a Bool indicating validity.

Default generic implementation - always returns true.
"""
function validate_entity(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Generic validation - accept all entities
    return length(entity_text) >= 2
end

"""
    validate_relation(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict)

Validate that a relation is valid according to domain rules.
Returns a Bool indicating validity.

Default generic implementation - accepts common relation types.
"""
function validate_relation(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    # Generic validation
    valid_types = ["ASSOCIATED_WITH", "MARRIED_TO", "PARENT_OF", "LOCATED_IN", "REIGNED", "IS_OF"]
    return relation_type in valid_types || startswith(relation_type, "RELATED")
end

"""
    calculate_entity_confidence(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict)

Calculate confidence score for an entity based on domain-specific rules.
Returns a Float64 between 0.0 and 1.0.

Default generic implementation returns a reasonable default.
"""
function calculate_entity_confidence(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Generic confidence calculation
    base_confidence = 0.6
    
    # Boost confidence for longer entities (more specific)
    if length(entity_text) > 10
        base_confidence += 0.1
    end
    
    # Boost for title case words
    if all(c -> isuppercase(c) || islowercase(c) || isspace(c), entity_text)
        base_confidence += 0.1
    end
    
    return min(0.95, base_confidence)
end

"""
    calculate_relation_confidence(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict)

Calculate confidence score for a relation based on domain-specific rules.
Returns a Float64 between 0.0 and 1.0.

Default generic implementation returns a reasonable default.
"""
function calculate_relation_confidence(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    # Generic confidence calculation for relations
    base_confidence = 0.5
    
    # Boost confidence for well-known relation types
    known_types = ["PARENT_OF", "SPOUSE_OF", "MARRIED_TO", "REIGNED", "BORN_IN", "DIED_IN"]
    if relation_type in known_types
        base_confidence += 0.2
    end
    
    # Boost if both head and tail are non-empty
    if length(head) > 0 && length(tail) > 0
        base_confidence += 0.1
    end
    
    return min(0.95, base_confidence)
end

# ============================================================================
# Optional Methods (domains can implement if needed)
# ============================================================================

"""
    link_entity(domain::DomainProvider, entity_text::String, config::Any)

Link entity to external knowledge base (e.g., UMLS for biomedical, Wikidata for Wikipedia).
Returns a Dict with linking information or nothing if not supported.
"""
function link_entity(domain::DomainProvider, entity_text::String, config::Any)
    return nothing  # Default: not supported
end

"""
    create_seed_triples(domain::DomainProvider, entity_text::String, config::Any)

Create seed KG triples for an entity from domain-specific knowledge base.
Returns a Vector of SemanticTriple objects or empty vector if not supported.
"""
function create_seed_triples(domain::DomainProvider, entity_text::String, config::Any)
    return Vector{Any}()  # Default: not supported
end

"""
    create_evaluation_metrics(domain::DomainProvider, kg::KnowledgeGraph)

Create domain-specific evaluation metrics.
Returns a Dict with metric names and values.
"""
function create_evaluation_metrics(domain::DomainProvider, kg::Any)
    return Dict{String, Any}()  # Default: no domain-specific metrics
end

"""
    create_prompt(domain::DomainProvider, task_type::Symbol, context::Dict)

Generate LLM prompt for domain-specific task.
Task types: :entity_discovery, :relation_matching, :tail_formation
Returns a String prompt.
"""
function create_prompt(domain::DomainProvider, task_type::Symbol, context::Dict{String, Any})
    error("create_prompt must be implemented if LLM prompts are needed for domain: $(typeof(domain))")
end

"""
    get_domain_name(domain::DomainProvider)

Get the name/identifier of this domain.
"""
function get_domain_name(domain::DomainProvider)
    error("get_domain_name must be implemented by domain: $(typeof(domain))")
end

"""
    get_domain_config(domain::DomainProvider)

Get the configuration for this domain.
"""
function get_domain_config(domain::DomainProvider)
    error("get_domain_config must be implemented by domain: $(typeof(domain))")
end

# Export the interface
export DomainProvider, DomainConfig
export register_entity_types, register_relation_types
export extract_entities, extract_relations
export validate_entity, validate_relation
export calculate_entity_confidence, calculate_relation_confidence
export link_entity, create_seed_triples, create_evaluation_metrics, create_prompt
export get_domain_name, get_domain_config
