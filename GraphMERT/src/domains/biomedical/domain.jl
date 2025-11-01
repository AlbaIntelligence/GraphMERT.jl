"""
Biomedical Domain Provider for GraphMERT.jl

This module implements the DomainProvider interface for the biomedical domain,
providing biomedical-specific entity extraction, relation extraction, validation,
confidence calculation, UMLS linking, and LLM prompt generation.
"""

using Dates
using Logging

# Import domain interface (these will be available from the main module)
# Note: When this module is included from GraphMERT.jl, these will already be loaded

# Import biomedical submodules
include("entities.jl")
include("relations.jl")
include("umls.jl")
include("prompts.jl")

"""
    BiomedicalDomain

Domain provider for biomedical knowledge graph extraction.
"""
mutable struct BiomedicalDomain <: DomainProvider
    config::DomainConfig
    entity_types::Dict{String, Dict{String, Any}}
    relation_types::Dict{String, Dict{String, Any}}
    umls_client::Union{Any, Nothing}  # UMLSClient when available
    
    function BiomedicalDomain(umls_client::Union{Any, Nothing} = nothing)
        # Initialize entity types
        entity_types = Dict{String, Dict{String, Any}}(
            "DISEASE" => Dict("domain" => "biomedical", "category" => "pathology", "umls_group" => "DISO"),
            "DRUG" => Dict("domain" => "biomedical", "category" => "chemical", "umls_group" => "CHEM"),
            "PROTEIN" => Dict("domain" => "biomedical", "category" => "molecule", "umls_group" => "CHEM"),
            "GENE" => Dict("domain" => "biomedical", "category" => "genetic", "umls_group" => "GENE"),
            "ANATOMY" => Dict("domain" => "biomedical", "category" => "anatomy", "umls_group" => "ANAT"),
            "SYMPTOM" => Dict("domain" => "biomedical", "category" => "phenotype", "umls_group" => "SIGN"),
            "PROCEDURE" => Dict("domain" => "biomedical", "category" => "procedure", "umls_group" => "PROC"),
            "ORGANISM" => Dict("domain" => "biomedical", "category" => "organism", "umls_group" => "LIVB"),
            "CHEMICAL" => Dict("domain" => "biomedical", "category" => "chemical", "umls_group" => "CHEM"),
            "CELL_LINE" => Dict("domain" => "biomedical", "category" => "cell", "umls_group" => "CELL"),
            "CELL_TYPE" => Dict("domain" => "biomedical", "category" => "cell", "umls_group" => "CELL"),
            "MOLECULAR_FUNCTION" => Dict("domain" => "biomedical", "category" => "function", "umls_group" => "FUNC"),
            "BIOLOGICAL_PROCESS" => Dict("domain" => "biomedical", "category" => "process", "umls_group" => "PROC"),
            "CELLULAR_COMPONENT" => Dict("domain" => "biomedical", "category" => "component", "umls_group" => "COMP"),
        )
        
        # Initialize relation types
        relation_types = Dict{String, Dict{String, Any}}(
            "TREATS" => Dict("domain" => "biomedical", "category" => "therapeutic"),
            "CAUSES" => Dict("domain" => "biomedical", "category" => "causal"),
            "ASSOCIATED_WITH" => Dict("domain" => "biomedical", "category" => "association"),
            "PREVENTS" => Dict("domain" => "biomedical", "category" => "therapeutic"),
            "INHIBITS" => Dict("domain" => "biomedical", "category" => "mechanistic"),
            "ACTIVATES" => Dict("domain" => "biomedical", "category" => "mechanistic"),
            "BINDS_TO" => Dict("domain" => "biomedical", "category" => "interaction"),
            "INTERACTS_WITH" => Dict("domain" => "biomedical", "category" => "interaction"),
            "REGULATES" => Dict("domain" => "biomedical", "category" => "regulatory"),
            "EXPRESSES" => Dict("domain" => "biomedical", "category" => "expression"),
            "LOCATED_IN" => Dict("domain" => "biomedical", "category" => "spatial"),
            "PART_OF" => Dict("domain" => "biomedical", "category" => "spatial"),
            "DERIVED_FROM" => Dict("domain" => "biomedical", "category" => "derivation"),
            "SYNONYMOUS_WITH" => Dict("domain" => "biomedical", "category" => "equivalence"),
            "CONTRAINDICATED_WITH" => Dict("domain" => "biomedical", "category" => "contraindication"),
            "INDICATES" => Dict("domain" => "biomedical", "category" => "diagnostic"),
            "MANIFESTS_AS" => Dict("domain" => "biomedical", "category" => "manifestation"),
            "ADMINISTERED_FOR" => Dict("domain" => "biomedical", "category" => "therapeutic"),
            "TARGETS" => Dict("domain" => "biomedical", "category" => "targeting"),
            "METABOLIZED_BY" => Dict("domain" => "biomedical", "category" => "metabolism"),
            "TRANSPORTED_BY" => Dict("domain" => "biomedical", "category" => "transport"),
            "SECRETED_BY" => Dict("domain" => "biomedical", "category" => "secretion"),
            "PRODUCED_BY" => Dict("domain" => "biomedical", "category" => "production"),
            "CONTAINS" => Dict("domain" => "biomedical", "category" => "composition"),
            "COMPONENT_OF" => Dict("domain" => "biomedical", "category" => "composition"),
        )
        
        config = DomainConfig(
            "biomedical";
            entity_types=collect(keys(entity_types)),
            relation_types=collect(keys(relation_types)),
            validation_rules=Dict{String, Any}(),
            extraction_patterns=Dict{String, Any}(),
            confidence_strategies=Dict{String, Any}(),
        )
        
        new(config, entity_types, relation_types, umls_client)
    end
end

# ============================================================================
# Required DomainProvider Methods
# ============================================================================

"""
    register_entity_types(domain::BiomedicalDomain)

Register biomedical entity types.
"""
function register_entity_types(domain::BiomedicalDomain)
    return domain.entity_types
end

"""
    register_relation_types(domain::BiomedicalDomain)

Register biomedical relation types.
"""
function register_relation_types(domain::BiomedicalDomain)
    return domain.relation_types
end

"""
    extract_entities(domain::BiomedicalDomain, text::String, config::ProcessingOptions)

Extract biomedical entities from text.
"""
function extract_entities(domain::BiomedicalDomain, text::String, config::Any)
    # Delegate to biomedical entities module
    return extract_biomedical_entities(text, config, domain)
end

"""
    extract_relations(domain::BiomedicalDomain, entities::Vector{Entity}, text::String, config::ProcessingOptions)

Extract biomedical relations between entities.
"""
function extract_relations(domain::BiomedicalDomain, entities::Vector{Any}, text::String, config::Any)
    # Delegate to biomedical relations module
    return extract_biomedical_relations(entities, text, config, domain)
end

"""
    validate_entity(domain::BiomedicalDomain, entity_text::String, entity_type::String, context::Dict)

Validate a biomedical entity.
"""
function validate_entity(domain::BiomedicalDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return validate_biomedical_entity(entity_text, entity_type, context)
end

"""
    validate_relation(domain::BiomedicalDomain, head::String, relation_type::String, tail::String, context::Dict)

Validate a biomedical relation.
"""
function validate_relation(domain::BiomedicalDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return validate_biomedical_relation(head, relation_type, tail, context)
end

"""
    calculate_entity_confidence(domain::BiomedicalDomain, entity_text::String, entity_type::String, context::Dict)

Calculate confidence for a biomedical entity.
"""
function calculate_entity_confidence(domain::BiomedicalDomain, entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    return calculate_biomedical_entity_confidence(entity_text, entity_type, context)
end

"""
    calculate_relation_confidence(domain::BiomedicalDomain, head::String, relation_type::String, tail::String, context::Dict)

Calculate confidence for a biomedical relation.
"""
function calculate_relation_confidence(domain::BiomedicalDomain, head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return calculate_biomedical_relation_confidence(head, relation_type, tail, context)
end

# ============================================================================
# Optional DomainProvider Methods
# ============================================================================

"""
    link_entity(domain::BiomedicalDomain, entity_text::String, config::Any)

Link entity to UMLS using biomedical domain's UMLS integration.
"""
function link_entity(domain::BiomedicalDomain, entity_text::String, config::Any)
    if domain.umls_client === nothing
        return nothing
    end
    
    # Delegate to UMLS linking
    linking_result = link_entity_to_umls(entity_text, domain.umls_client)
    
    # Convert to domain interface format
    if linking_result === nothing
        return nothing
    end
    
    # Return in the format expected by seed_injection.jl
    # Format: Dict with :candidates or :candidate key
    if isa(linking_result, Vector)
        candidates = []
        for result in linking_result
            if isa(result, EntityLinkingResult)
                push!(candidates, Dict(
                    :kb_id => result.cui,
                    :name => result.preferred_name,
                    :types => result.semantic_types,
                    :score => result.similarity_score,
                    :source => result.source,
                ))
            end
        end
        return Dict(:candidates => candidates)
    elseif isa(linking_result, EntityLinkingResult)
        return Dict(:candidate => Dict(
            :kb_id => linking_result.cui,
            :name => linking_result.preferred_name,
            :types => linking_result.semantic_types,
            :score => linking_result.similarity_score,
            :source => linking_result.source,
        ))
    end
    
    return nothing
end

"""
    create_seed_triples(domain::BiomedicalDomain, entity_text::String, config::Any)

Create seed KG triples from UMLS for a biomedical entity.
"""
function create_seed_triples(domain::BiomedicalDomain, entity_text::String, config::Any)
    if domain.umls_client === nothing
        return Vector{Any}()
    end
    
    # Get UMLS linking result
    linking_result = link_entity(domain, entity_text, config)
    if linking_result === nothing
        return Vector{Any}()
    end
    
    # Convert to SemanticTriple format
    # This is a simplified implementation - full version would query UMLS for triples
    triples = Vector{Any}()
    # TODO: Implement full triple retrieval from UMLS
    
    return triples
end

"""
    create_prompt(domain::BiomedicalDomain, task_type::Symbol, context::Dict)

Generate LLM prompt for biomedical domain tasks.
"""
function create_prompt(domain::BiomedicalDomain, task_type::Symbol, context::Dict{String, Any})
    return create_biomedical_prompt(task_type, context)
end

"""
    get_domain_name(domain::BiomedicalDomain)

Get the name of this domain.
"""
function get_domain_name(domain::BiomedicalDomain)
    return "biomedical"
end

"""
    get_domain_config(domain::BiomedicalDomain)

Get the configuration for this domain.
"""
function get_domain_config(domain::BiomedicalDomain)
    return domain.config
end

# Export
export BiomedicalDomain
