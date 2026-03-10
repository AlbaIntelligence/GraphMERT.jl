"""
Domain Interface for GraphMERT.jl

This module defines the abstract interface that all domain implementations must provide.
Domains are pluggable modules that define domain-specific behavior for entity extraction,
relation extraction, validation, confidence calculation, and other domain-specific operations.
"""

using Dates
using Logging

abstract type DomainProvider end

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

# Required methods - domains must implement these
function register_entity_types(domain::DomainProvider)
    error("register_entity_types must be implemented by domain: $(typeof(domain))")
end

function register_relation_types(domain::DomainProvider)
    error("register_relation_types must be implemented by domain: $(typeof(domain))")
end

# Default implementations with fallback - used when domain-specific methods fail
function extract_entities(domain::DomainProvider, text::String, config::Any)
    entities = Entity[]
    entity_id = 1
    
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
            length(entity_text) < 2 && continue
            
            confidence = 0.5 + 0.3 * rand()
            if !any(e -> e.text == entity_text, entities)
                push!(entities, Entity(
                    "entity_$entity_id",
                    entity_text,
                    entity_text,
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

function extract_relations(domain::DomainProvider, entities::Vector{Entity}, text::String, config::Any)
    relations = Relation[]
    relation_id = 1
    
    relation_patterns = [
        (r"(\w+)\s+(?:was|is)\s+(?:the\s+)?(\w+)\s+of\s+(\w+)", "IS_OF"),
        (r"(\w+)\s+(?:married|married\s+to)\s+(\w+)", "MARRIED_TO"),
        (r"(\w+)\s+(?:was|is)\s+(?:born\s+in|died\s+in)\s+(\w+)", "LOCATED_IN"),
        (r"(\w+)\s+(?:reigned\s+(?:from|until)|ruled)\s+(\w+)", "REIGNED"),
        (r"(\w+)\s+(?:was|is)\s+(?:the\s+)?(?:father|mother|son|daughter|parent)\s+of\s+(\w+)", "PARENT_OF"),
    ]
    
    sentences = split(text, r"[.!?]")
    for sentence in sentences
        sentence_entities = filter(e -> occursin(e.text, sentence), entities)
        if length(sentence_entities) >= 2
            head = sentence_entities[1]
            tail = sentence_entities[2]
            if !any(r -> r.head == head.id && r.tail == tail.id, relations)
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
                    head.id, tail.id, relation_type, confidence, "generic",
                    String(sentence), "", Dict{String, Any}(), "relation_$relation_id"
                ))
                relation_id += 1
            end
        end
    end
    return relations
end

function validate_entity(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return length(entity_text) >= 2
end

function validate_relation(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    valid_types = ["ASSOCIATED_WITH", "MARRIED_TO", "PARENT_OF", "LOCATED_IN", "REIGNED", "IS_OF"]
    return relation_type in valid_types || startswith(relation_type, "RELATED")
end

function calculate_entity_confidence(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    base = 0.6
    length(entity_text) > 10 && (base += 0.1)
    all(c -> isuppercase(c) || islowercase(c) || isspace(c), entity_text) && (base += 0.1)
    return min(0.95, base)
end

function calculate_relation_confidence(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    base = 0.5
    known_types = ["PARENT_OF", "SPOUSE_OF", "MARRIED_TO", "REIGNED", "BORN_IN", "DIED_IN"]
    relation_type in known_types && (base += 0.2)
    length(head) > 0 && length(tail) > 0 && (base += 0.1)
    return min(0.95, base)
end

# Optional methods - have default implementations
function link_entity(domain::DomainProvider, entity_text::String, config::Any)
    return nothing
end

function create_seed_triples(domain::DomainProvider, entity_text::String, config::Any)
    return Vector{Any}()
end

function create_evaluation_metrics(domain::DomainProvider, kg::Any)
    return Dict{String, Any}()
end

function create_prompt(domain::DomainProvider, task_type::Symbol, context::Dict{String, Any})
    error("create_prompt must be implemented for domain: $(typeof(domain))")
end

function get_domain_name(domain::DomainProvider)
    error("get_domain_name must be implemented by domain: $(typeof(domain))")
end

function get_domain_config(domain::DomainProvider)
    error("get_domain_config must be implemented by domain: $(typeof(domain))")
end

# Export
export DomainProvider, DomainConfig
export register_entity_types, register_relation_types
export extract_entities, extract_relations
export validate_entity, validate_relation
export calculate_entity_confidence, calculate_relation_confidence
export link_entity, create_seed_triples, create_evaluation_metrics, create_prompt
export get_domain_name, get_domain_config
