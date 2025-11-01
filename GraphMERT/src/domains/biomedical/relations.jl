"""
Biomedical Relation Extraction for GraphMERT.jl

This module provides biomedical-specific relation extraction functionality.
It bridges to the existing biomedical/relations.jl implementation.
"""

# Import the existing biomedical relations module
# Note: This will be refactored to be fully domain-independent later
include("../../biomedical/relations.jl")

# Forward declarations (will be available when module is loaded)
# Entity, Relation, ProcessingOptions are defined in GraphMERT.types.jl

"""
    extract_biomedical_relations(entities::Vector{Entity}, text::String, config::ProcessingOptions, domain::BiomedicalDomain)

Extract biomedical relations between entities.

# Arguments
- `entities::Vector{Entity}`: Extracted entities
- `text::String`: Original text
- `config::ProcessingOptions`: Processing options
- `domain::BiomedicalDomain`: Domain provider instance

# Returns
- `Vector{Relation}`: Extracted relations
"""
function extract_biomedical_relations(entities::Vector{Any}, text::String, config::Any, domain::Any)
    # Use GraphMERT.Relation from Main scope
    RelationType = Main.GraphMERT.Relation
    
    relations = Vector{RelationType}()
    
    if length(entities) < 2
        return relations
    end
    
    # Extract relations between entity pairs
    for i in 1:(length(entities)-1)
        for j in (i+1):length(entities)
            head_entity = entities[i]
            tail_entity = entities[j]
            
            # Find context around entities
            head_pos = head_entity.position.start
            tail_pos = tail_entity.position.start
            context_start = max(1, min(head_pos, tail_pos) - 50)
            context_end = min(length(text), max(head_pos + length(head_entity.text), tail_pos + length(tail_entity.text)) + 50)
            context = text[context_start:context_end]
            
            # Classify relation
            relation_type_enum = classify_by_rules(head_entity.text, tail_entity.text, context)
            relation_type = string(relation_type_enum)
            
            # Calculate confidence
            confidence = calculate_biomedical_relation_confidence(head_entity.text, relation_type, tail_entity.text, Dict{String, Any}("context" => context))
            
            # Create relation
            relation = RelationType(
                head_entity.id,
                tail_entity.id,
                relation_type,
                0.7,  # confidence
                "biomedical",
                text,
                context,
                Dict{String, Any}(
                    "relation_type_enum" => relation_type_enum,
                    "context" => context,
                ),
                "relation_$(i)_$(j)_$(hash(relation_type))",  # id
            )
            push!(relations, relation)
        end
    end
    
    return relations
end

"""
    validate_biomedical_relation(head::String, relation_type::String, tail::String, context::Dict)

Validate a biomedical relation.
"""
function validate_biomedical_relation(head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    # Basic validation: check that all fields are non-empty
    if isempty(head) || isempty(tail) || isempty(relation_type)
        return false
    end
    
    # Parse relation type string to enum if possible
    try
        relation_type_enum = parse_relation_type(relation_type)
        if relation_type_enum != UNKNOWN_RELATION
            return true
        else
            return false
        end
    catch
        # If parsing fails, basic validation: check that relation type is valid string
        # Check if relation type is in the registered relation types
        return true  # For now, accept any non-empty relation type
    end
end

"""
    calculate_biomedical_relation_confidence(head::String, relation_type::String, tail::String, context::Dict)

Calculate confidence for a biomedical relation.
"""
function calculate_biomedical_relation_confidence(head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    # Parse relation type string to enum if possible
    try
        relation_type_enum = parse_relation_type(relation_type)
        if relation_type_enum != UNKNOWN_RELATION
            # Basic confidence based on relation type validity
            return 0.7
        else
            return 0.3
        end
    catch
        # Basic confidence calculation
        return 0.5
    end
end

# Export
export extract_biomedical_relations
export validate_biomedical_relation
export calculate_biomedical_relation_confidence
