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
"""
function extract_entities(domain::DomainProvider, text::String, config::Any)
    error("extract_entities must be implemented by domain: $(typeof(domain))")
end

"""
    extract_relations(domain::DomainProvider, entities::Vector{Entity}, text::String, config::ProcessingOptions)

Extract relations between entities using domain-specific patterns and rules.
Returns a Vector of Relation objects.
"""
function extract_relations(domain::DomainProvider, entities::Vector{Any}, text::String, config::Any)
    error("extract_relations must be implemented by domain: $(typeof(domain))")
end

"""
    validate_entity(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict)

Validate that an entity text matches its claimed type according to domain rules.
Returns a Bool indicating validity.
"""
function validate_entity(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    error("validate_entity must be implemented by domain: $(typeof(domain))")
end

"""
    validate_relation(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict)

Validate that a relation is valid according to domain rules.
Returns a Bool indicating validity.
"""
function validate_relation(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    error("validate_relation must be implemented by domain: $(typeof(domain))")
end

"""
    calculate_entity_confidence(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict)

Calculate confidence score for an entity based on domain-specific rules.
Returns a Float64 between 0.0 and 1.0.
"""
function calculate_entity_confidence(domain::DomainProvider, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    error("calculate_entity_confidence must be implemented by domain: $(typeof(domain))")
end

"""
    calculate_relation_confidence(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict)

Calculate confidence score for a relation based on domain-specific rules.
Returns a Float64 between 0.0 and 1.0.
"""
function calculate_relation_confidence(domain::DomainProvider, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    error("calculate_relation_confidence must be implemented by domain: $(typeof(domain))")
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
