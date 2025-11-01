"""
Biomedical Entity Extraction for GraphMERT.jl

This module provides biomedical-specific entity extraction functionality.
It bridges to the existing biomedical/entities.jl implementation.
"""

# Import the existing biomedical entities module
# Note: This will be refactored to be fully domain-independent later
# The path is relative to this file: domains/biomedical/entities.jl -> biomedical/entities.jl
include("../../biomedical/entities.jl")

# Import types from main module (they'll be available when this is included)
# Note: Cannot use `using ..GraphMERT` here because this file is included from domain.jl
# which is included from biomedical.jl, not directly from GraphMERT.jl
# Types will be available in the parent scope

# Forward declarations (will be available when module is loaded)
# Entity, TextPosition, ProcessingOptions are defined in GraphMERT.types.jl

"""
    extract_biomedical_entities(text::String, config::ProcessingOptions, domain::BiomedicalDomain)

Extract biomedical entities from text.

# Arguments
- `text::String`: Input text
- `config::ProcessingOptions`: Processing options
- `domain::BiomedicalDomain`: Domain provider instance

# Returns
- `Vector{Entity}`: Extracted entities
"""
function extract_biomedical_entities(text::String, config::Any, domain::Any)
    # Use the existing extraction function
    # Convert to Entity objects
    entities = Vector{Entity}()
    
    # Simple extraction using patterns
    # Get all biomedical entity types
    entity_types = [
        DISEASE, DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM,
        PROCEDURE, ORGANISM, CHEMICAL, CELL_TYPE
    ]
    
    extracted = extract_entities_from_text(text; entity_types=entity_types)
    
    for (i, (entity_text, entity_type, confidence)) in enumerate(extracted)
        # Find position in text
        pos = findfirst(entity_text, text)
        start_pos = pos !== nothing ? first(pos) : 1
        end_pos = pos !== nothing ? last(pos) : length(entity_text)
        
        entity = Entity(
            "entity_$(i)_$(hash(entity_text))",
            entity_text,
            entity_text,
            string(entity_type),
            "biomedical",
            Dict{String, Any}(
                "entity_type_enum" => entity_type,
                "confidence" => confidence,
            ),
            TextPosition(start_pos, end_pos, 1, 1),
            confidence,
            text,
        )
        push!(entities, entity)
    end
    
    return entities
end

"""
    validate_biomedical_entity(entity_text::String, entity_type::String, context::Dict)

Validate a biomedical entity.
"""
function validate_biomedical_entity(entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Parse entity type string to enum if possible
    try
        entity_type_enum = parse_entity_type(entity_type)
        if entity_type_enum isa BiomedicalEntityType
            return validate_biomedical_entity(entity_text, entity_type_enum)
        end
    catch
        # If parsing fails, use basic validation
        return length(entity_text) > 2 && length(entity_text) < 100
    end
    
    return true
end

"""
    calculate_biomedical_entity_confidence(entity_text::String, entity_type::String, context::Dict)

Calculate confidence for a biomedical entity.
"""
function calculate_biomedical_entity_confidence(entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Parse entity type string to enum if possible
    try
        entity_type_enum = parse_entity_type(entity_type)
        if entity_type_enum isa BiomedicalEntityType
            return calculate_entity_confidence(entity_text, entity_type_enum)
        end
    catch
        # Basic confidence calculation
        return 0.5
    end
    
    return 0.5
end

# Export
export extract_biomedical_entities
export validate_biomedical_entity
export calculate_biomedical_entity_confidence
