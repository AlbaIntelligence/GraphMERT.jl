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

function extract_entities end
function extract_relations end
function validate_entity end
function validate_relation end
function calculate_entity_confidence end
function calculate_relation_confidence end

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
