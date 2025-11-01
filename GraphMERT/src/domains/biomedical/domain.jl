"""
Biomedical Domain Provider for GraphMERT.jl

This module implements the DomainProvider interface for the biomedical domain,
providing biomedical-specific entity extraction, relation extraction, validation,
confidence calculation, UMLS linking, and LLM prompt generation.
"""

using Dates
using Logging

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

This function queries UMLS for relations involving the entity and converts them
to SemanticTriple format for seed injection.

# Arguments
- `domain::BiomedicalDomain`: Domain provider instance
- `entity_text::String`: Entity text or CUI (if entity_text is a CUI, it will be used directly)
- `config::Any`: Configuration (can be SeedInjectionConfig or ProcessingOptions)

# Returns
- `Vector{Any}`: Vector of SemanticTriple objects or Dicts that can be converted to SemanticTriple
"""
function create_seed_triples(domain::BiomedicalDomain, entity_text::String, config::Any)
    if domain.umls_client === nothing
        return Vector{Any}()
    end
    
    # Get CUI from entity text (or use entity_text directly if it's already a CUI)
    # First try to link the entity to get its CUI
    linking_result = link_entity(domain, entity_text, config)
    
    cui = nothing
    entity_name = entity_text
    
    if linking_result !== nothing && isa(linking_result, Dict)
        if haskey(linking_result, :candidate) && isa(linking_result[:candidate], Dict)
            cui = get(linking_result[:candidate], :kb_id, nothing)
            entity_name = get(linking_result[:candidate], :name, entity_text)
        elseif haskey(linking_result, :candidates) && isa(linking_result[:candidates], Vector) && !isempty(linking_result[:candidates])
            first_candidate = linking_result[:candidates][1]
            if isa(first_candidate, Dict)
                cui = get(first_candidate, :kb_id, nothing)
                entity_name = get(first_candidate, :name, entity_text)
            end
        end
    end
    
    # If we couldn't get a CUI, try using entity_text as CUI directly (in case it's already a CUI)
    if cui === nothing && length(entity_text) == 8 && all(c -> isdigit(c) || isuppercase(c), entity_text)
        # Looks like a CUI (format: C0000000)
        cui = entity_text
    end
    
    if cui === nothing
        return Vector{Any}()
    end
    
    # Query UMLS for relations
    relations_response = get_relations(domain.umls_client, cui)
    
    if !relations_response.success || !haskey(relations_response.data, "result")
        return Vector{Any}()
    end
    
    relations_data = relations_response.data["result"]
    relations_list = get(relations_data, "relations", Vector{Any}())
    
    if isempty(relations_list)
        return Vector{Any}()
    end
    
    # Convert UMLS relations to SemanticTriple format
    triples = Vector{Any}()
    
    # Get SemanticTriple type from main module
    SemanticTripleType = Main.GraphMERT.SemanticTriple
    
    for relation in relations_list
        if !isa(relation, Dict)
            continue
        end
        
        # Extract relation information
        target_cui = get(relation, "targetCUI", "")
        relation_name = get(relation, "relationName", "")
        relation_type = get(relation, "relationType", "")
        
        if isempty(target_cui) || isempty(relation_name)
            continue
        end
        
        # Map UMLS relation to our relation type
        # Use the relation name or type, defaulting to relation_name
        mapped_relation = map_umls_relation_to_biomedical_type(relation_name)
        
        # Get target entity name (would need to query UMLS, but for now use CUI)
        target_entity_name = target_cui  # In full implementation, would query UMLS for preferred name
        
        # Create token IDs for tail entity (simplified - would tokenize properly)
        # For now, create placeholder tokens
        tail_tokens = [hash(target_entity_name) % 10000]  # Placeholder token IDs
        
        # Calculate confidence score (simplified)
        confidence = 0.7  # Base confidence for UMLS relations
        
        # Create SemanticTriple
        # Note: head_cui is the CUI we queried, tail_cui is the target CUI
        try
            triple = SemanticTripleType(
                entity_name,      # head
                cui,              # head_cui
                mapped_relation,  # relation
                target_entity_name,  # tail
                tail_tokens,      # tail_tokens
                confidence,        # score
                "UMLS",           # source
            )
            push!(triples, triple)
        catch e
            # If SemanticTriple construction fails, create Dict format instead
            push!(triples, Dict(
                :head => entity_name,
                :head_kb_id => cui,
                :relation => mapped_relation,
                :tail => target_entity_name,
                :tail_tokens => tail_tokens,
                :score => confidence,
                :source => "UMLS",
            ))
        end
    end
    
    return triples
end

"""
    map_umls_relation_to_biomedical_type(umls_relation::String)

Map UMLS relation name to biomedical relation type string.
"""
function map_umls_relation_to_biomedical_type(umls_relation::String)
    relation_lower = lowercase(umls_relation)
    
    # Map common UMLS relations to biomedical relation types
    if occursin("treats", relation_lower) || occursin("therapeutic", relation_lower)
        return "TREATS"
    elseif occursin("causes", relation_lower) || occursin("etiology", relation_lower)
        return "CAUSES"
    elseif occursin("associated", relation_lower) || occursin("related", relation_lower)
        return "ASSOCIATED_WITH"
    elseif occursin("prevents", relation_lower) || occursin("prophylaxis", relation_lower)
        return "PREVENTS"
    elseif occursin("inhibits", relation_lower) || occursin("suppresses", relation_lower)
        return "INHIBITS"
    elseif occursin("activates", relation_lower) || occursin("stimulates", relation_lower)
        return "ACTIVATES"
    elseif occursin("binds", relation_lower) || occursin("interacts", relation_lower)
        return "BINDS_TO"
    elseif occursin("regulates", relation_lower) || occursin("modulates", relation_lower)
        return "REGULATES"
    elseif occursin("expresses", relation_lower) || occursin("produces", relation_lower)
        return "EXPRESSES"
    elseif occursin("located", relation_lower) || occursin("found", relation_lower)
        return "LOCATED_IN"
    elseif occursin("part", relation_lower) || occursin("component", relation_lower)
        return "PART_OF"
    elseif occursin("derived", relation_lower) || occursin("originates", relation_lower)
        return "DERIVED_FROM"
    elseif occursin("synonymous", relation_lower) || occursin("equivalent", relation_lower)
        return "SYNONYMOUS_WITH"
    elseif occursin("contraindicated", relation_lower)
        return "CONTRAINDICATED_WITH"
    elseif occursin("indicates", relation_lower) || occursin("suggests", relation_lower)
        return "INDICATES"
    elseif occursin("manifests", relation_lower) || occursin("presents", relation_lower)
        return "MANIFESTS_AS"
    elseif occursin("administered", relation_lower) || occursin("used", relation_lower)
        return "ADMINISTERED_FOR"
    elseif occursin("targets", relation_lower) || occursin("acts", relation_lower)
        return "TARGETS"
    elseif occursin("metabolized", relation_lower) || occursin("metabolism", relation_lower)
        return "METABOLIZED_BY"
    elseif occursin("transported", relation_lower) || occursin("transport", relation_lower)
        return "TRANSPORTED_BY"
    elseif occursin("secreted", relation_lower) || occursin("secretion", relation_lower)
        return "SECRETED_BY"
    elseif occursin("produced", relation_lower) || occursin("production", relation_lower)
        return "PRODUCED_BY"
    elseif occursin("contains", relation_lower) || occursin("includes", relation_lower)
        return "CONTAINS"
    elseif occursin("component", relation_lower) || occursin("constituent", relation_lower)
        return "COMPONENT_OF"
    else
        # Default to ASSOCIATED_WITH for unknown relations
        return "ASSOCIATED_WITH"
    end
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
