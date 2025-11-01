"""
Biomedical Entity Extraction for GraphMERT.jl

This module provides biomedical-specific entity extraction functionality.
It is self-contained and includes all entity types, patterns, validation, and extraction logic.
"""

# Import types from main module (they'll be available when this is included)
# Note: Cannot use `using ..GraphMERT` here because this file is included from domain.jl
# which is included from biomedical.jl, not directly from GraphMERT.jl
# Types will be available in the parent scope

# Forward declarations (will be available when module is loaded)
# Entity, TextPosition, ProcessingOptions are defined in GraphMERT.types.jl

# ============================================================================
# Biomedical Entity Type Enum (Legacy Compatibility)
# ============================================================================

"""
    BiomedicalEntityType

Legacy enum for backward compatibility. Use string-based types for the generic system.
"""
@enum BiomedicalEntityType begin
    DISEASE
    DRUG
    PROTEIN
    GENE
    ANATOMY
    SYMPTOM
    PROCEDURE
    ORGANISM
    CHEMICAL
    CELL_LINE
    CELL_TYPE
    MOLECULAR_FUNCTION
    BIOLOGICAL_PROCESS
    CELLULAR_COMPONENT
    PERSON
    ORGANIZATION
    LOCATION
    CONCEPT
    EVENT
    TECHNOLOGY
    ARTWORK
    PERIOD
    THEORY
    METHOD
    INSTITUTION
    COUNTRY
    UNKNOWN
end

# ============================================================================
# Entity Type Parsing and Utilities
# ============================================================================

"""
    parse_entity_type(type_name::String)

Parse a string to a biomedical entity type enum (legacy compatibility).
"""
function parse_entity_type(type_name::String)
    type_name_upper = uppercase(type_name)

    if type_name_upper == "DISEASE"
        return DISEASE
    elseif type_name_upper == "DRUG"
        return DRUG
    elseif type_name_upper == "PROTEIN"
        return PROTEIN
    elseif type_name_upper == "GENE"
        return GENE
    elseif type_name_upper == "ANATOMY"
        return ANATOMY
    elseif type_name_upper == "SYMPTOM"
        return SYMPTOM
    elseif type_name_upper == "PROCEDURE"
        return PROCEDURE
    elseif type_name_upper == "ORGANISM"
        return ORGANISM
    elseif type_name_upper == "CHEMICAL"
        return CHEMICAL
    elseif type_name_upper == "CELL_LINE"
        return CELL_LINE
    elseif type_name_upper == "CELL_TYPE"
        return CELL_TYPE
    elseif type_name_upper == "MOLECULAR_FUNCTION"
        return MOLECULAR_FUNCTION
    elseif type_name_upper == "BIOLOGICAL_PROCESS"
        return BIOLOGICAL_PROCESS
    elseif type_name_upper == "CELLULAR_COMPONENT"
        return CELLULAR_COMPONENT
    elseif type_name_upper == "PERSON"
        return PERSON
    elseif type_name_upper == "ORGANIZATION"
        return ORGANIZATION
    elseif type_name_upper == "LOCATION"
        return LOCATION
    elseif type_name_upper == "CONCEPT"
        return CONCEPT
    elseif type_name_upper == "EVENT"
        return EVENT
    elseif type_name_upper == "TECHNOLOGY"
        return TECHNOLOGY
    elseif type_name_upper == "ARTWORK"
        return ARTWORK
    elseif type_name_upper == "PERIOD"
        return PERIOD
    elseif type_name_upper == "THEORY"
        return THEORY
    elseif type_name_upper == "METHOD"
        return METHOD
    elseif type_name_upper == "INSTITUTION"
        return INSTITUTION
    elseif type_name_upper == "COUNTRY"
        return COUNTRY
    else
        return UNKNOWN
    end
end

# ============================================================================
# Entity Classification
# ============================================================================

"""
    classify_by_rules(entity_text_lower::String)

Classify entity using rule-based patterns.
"""
function classify_by_rules(entity_text_lower::String)
    # Disease patterns
    if occursin(
        r"\b(disease|disorder|syndrome|condition|illness|pathology)\b",
        entity_text_lower,
    )
        return DISEASE
    end

    # Drug patterns
    if occursin(
        r"\b(drug|medication|medicine|pharmaceutical|therapeutic|treatment)\b",
        entity_text_lower,
    )
        return DRUG
    end

    # Protein patterns
    if occursin(
        r"\b(protein|enzyme|receptor|antibody|hormone|kinase|phosphatase)\b",
        entity_text_lower,
    )
        return PROTEIN
    end

    # Gene patterns
    if occursin(
        r"\b(gene|genetic|allele|mutation|variant|polymorphism)\b",
        entity_text_lower,
    )
        return GENE
    end

    # Anatomy patterns
    if occursin(
        r"\b(organ|tissue|muscle|bone|nerve|vessel|gland|organelle)\b",
        entity_text_lower,
    )
        return ANATOMY
    end

    # Symptom patterns
    if occursin(
        r"\b(symptom|sign|manifestation|complaint|pain|fever|nausea)\b",
        entity_text_lower,
    )
        return SYMPTOM
    end

    # Procedure patterns
    if occursin(
        r"\b(procedure|surgery|operation|treatment|therapy|intervention)\b",
        entity_text_lower,
    )
        return PROCEDURE
    end

    # Organism patterns
    if occursin(
        r"\b(bacteria|virus|fungus|parasite|microorganism|pathogen)\b",
        entity_text_lower,
    )
        return ORGANISM
    end

    # Chemical patterns
    if occursin(
        r"\b(chemical|compound|molecule|substance|element|ion)\b",
        entity_text_lower,
    )
        return CHEMICAL
    end

    # Cell patterns
    if occursin(r"\b(cell|tissue|line|culture|strain)\b", entity_text_lower)
        return CELL_TYPE
    end

    return UNKNOWN
end

# ============================================================================
# Entity Validation
# ============================================================================

"""
    validate_biomedical_entity(entity_text::String, entity_type::BiomedicalEntityType)

Validate that an entity text matches its claimed type (legacy enum version).
"""
function validate_biomedical_entity(entity_text::String, entity_type::BiomedicalEntityType)
    if isempty(entity_text)
        return false
    end

    # Basic length validation
    if length(entity_text) < 2 || length(entity_text) > 200
        return false
    end

    # Type-specific validation
    entity_text_lower = lowercase(entity_text)

    if entity_type == DISEASE
        return occursin(
            r"\b(disease|disorder|syndrome|condition|illness|pathology)\b",
            entity_text_lower,
        )
    elseif entity_type == DRUG
        return occursin(
            r"\b(drug|medication|medicine|pharmaceutical|therapeutic|treatment)\b",
            entity_text_lower,
        )
    elseif entity_type == PROTEIN
        return occursin(
            r"\b(protein|enzyme|receptor|antibody|hormone|kinase|phosphatase)\b",
            entity_text_lower,
        )
    elseif entity_type == GENE
        return occursin(
            r"\b(gene|genetic|allele|mutation|variant|polymorphism)\b",
            entity_text_lower,
        )
    elseif entity_type == ANATOMY
        return occursin(
            r"\b(organ|tissue|muscle|bone|nerve|vessel|gland|organelle)\b",
            entity_text_lower,
        )
    elseif entity_type == SYMPTOM
        return occursin(
            r"\b(symptom|sign|manifestation|complaint|pain|fever|nausea)\b",
            entity_text_lower,
        )
    elseif entity_type == PROCEDURE
        return occursin(
            r"\b(procedure|surgery|operation|treatment|therapy|intervention)\b",
            entity_text_lower,
        )
    elseif entity_type == ORGANISM
        return occursin(
            r"\b(bacteria|virus|fungus|parasite|microorganism|pathogen)\b",
            entity_text_lower,
        )
    elseif entity_type == CHEMICAL
        return occursin(
            r"\b(chemical|compound|molecule|substance|element|ion)\b",
            entity_text_lower,
        )
    elseif entity_type == CELL_TYPE
        return occursin(r"\b(cell|tissue|line|culture|strain)\b", entity_text_lower)
    else
        return true  # Unknown type is always valid
    end
end

"""
    validate_biomedical_entity(entity_text::String, entity_type::String, context::Dict)

Validate a biomedical entity (string-based version for domain interface).
"""
function validate_biomedical_entity(entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Parse entity type string to enum if possible
    try
        entity_type_enum = parse_entity_type(entity_type)
        if entity_type_enum isa BiomedicalEntityType
            # Call the enum-based validation function
            return validate_biomedical_entity(entity_text, entity_type_enum)
        end
    catch e
        # If parsing fails, use basic validation
        return length(entity_text) > 2 && length(entity_text) < 100
    end

    # Fallback: basic validation
    return length(entity_text) > 2 && length(entity_text) < 100
end

# ============================================================================
# Entity Confidence Scoring
# ============================================================================

"""
    calculate_entity_confidence(entity_text::String, entity_type::BiomedicalEntityType)

Calculate confidence score for entity classification (legacy enum version).
"""
function calculate_entity_confidence(entity_text::String, entity_type::BiomedicalEntityType)
    if !validate_biomedical_entity(entity_text, entity_type)
        return 0.0
    end

    # Base confidence
    confidence = 0.5

    # Length bonus (optimal length range)
    text_length = length(entity_text)
    if 5 <= text_length <= 50
        confidence += 0.2
    elseif 3 <= text_length <= 100
        confidence += 0.1
    end

    # Specificity bonus (more specific terms get higher confidence)
    entity_text_lower = lowercase(entity_text)
    if occursin(r"\b(specific|precise|exact|definitive)\b", entity_text_lower)
        confidence += 0.1
    end

    # Domain-specific terminology bonus
    if entity_type in [
        DISEASE,
        DRUG,
        PROTEIN,
        GENE,
        ANATOMY,
        SYMPTOM,
        PROCEDURE,
        ORGANISM,
        CHEMICAL,
        CELL_LINE,
        CELL_TYPE,
        MOLECULAR_FUNCTION,
        BIOLOGICAL_PROCESS,
        CELLULAR_COMPONENT,
    ]
        # Medical terminology bonus for biomedical entities
        if occursin(r"\b(medical|clinical|biomedical|scientific)\b", entity_text_lower)
            confidence += 0.1
        end
    end

    return min(confidence, 1.0)
end

"""
    calculate_biomedical_entity_confidence(entity_text::String, entity_type::String, context::Dict)

Calculate confidence for a biomedical entity (string-based version for domain interface).
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

# ============================================================================
# Entity Extraction Patterns
# ============================================================================

"""
    get_entity_patterns(entity_type::BiomedicalEntityType)

Get regex patterns for extracting entities of a specific type.
"""
function get_entity_patterns(entity_type::BiomedicalEntityType)
    patterns = Regex[]

    if entity_type == DISEASE
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|disorder|syndrome|condition|illness|pathology)\b",
        )
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:syndrome|disease)\b")
    elseif entity_type == DRUG
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:drug|medication|medicine|pharmaceutical)\b",
        )
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:treatment|therapy)\b")
    elseif entity_type == PROTEIN
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:protein|enzyme|receptor|antibody|hormone)\b",
        )
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:kinase|phosphatase|transferase)\b",
        )
    elseif entity_type == GENE
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:gene|genetic|allele|mutation|variant)\b",
        )
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:polymorphism|SNP|variant)\b",
        )
    elseif entity_type == ANATOMY
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:organ|tissue|muscle|bone|nerve|vessel|gland)\b",
        )
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:organelle|membrane|nucleus|mitochondria)\b",
        )
    elseif entity_type == SYMPTOM
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:symptom|sign|manifestation|complaint)\b",
        )
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:pain|fever|nausea|headache)\b",
        )
    elseif entity_type == PROCEDURE
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:procedure|surgery|operation|treatment)\b",
        )
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:therapy|intervention|therapy)\b",
        )
    elseif entity_type == ORGANISM
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:bacteria|virus|fungus|parasite|microorganism)\b",
        )
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:pathogen|organism|species)\b",
        )
    elseif entity_type == CHEMICAL
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:chemical|compound|molecule|substance)\b",
        )
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:element|ion|salt|acid|base)\b",
        )
    elseif entity_type == CELL_TYPE
        push!(
            patterns,
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:cell|tissue|line|culture|strain)\b",
        )
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:type|kind|form|variant)\b")
    end

    return patterns
end

"""
    extract_entities_from_text(text::String; entity_types::Vector{BiomedicalEntityType}=BiomedicalEntityType[])

Extract biomedical entities from text using pattern matching.
"""
function extract_entities_from_text(
    text::String;
    entity_types::Vector{BiomedicalEntityType}=BiomedicalEntityType[],
)
    if isempty(entity_types)
        entity_types = [
            DISEASE,
            DRUG,
            PROTEIN,
            GENE,
            ANATOMY,
            SYMPTOM,
            PROCEDURE,
            ORGANISM,
            CHEMICAL,
            CELL_TYPE,
        ]
    end

    entities = Vector{Tuple{String,BiomedicalEntityType,Float64}}()

    for entity_type in entity_types
        patterns = get_entity_patterns(entity_type)

        for pattern in patterns
            matches = eachmatch(pattern, text, overlap=false)
            for match in matches
                entity_text = String(match.match)
                confidence = calculate_entity_confidence(entity_text, entity_type)

                if confidence > 0.3  # Minimum confidence threshold
                    push!(entities, (entity_text, entity_type, confidence))
                end
            end
        end
    end

    # Remove duplicates and sort by confidence
    unique_entities = unique(entities)
    sort!(unique_entities, by=x -> x[3], rev=true)

    return unique_entities
end

# ============================================================================
# Domain Interface Functions
# ============================================================================

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
    # Use GraphMERT.Entity from Main scope
    EntityType = Main.GraphMERT.Entity
    TextPositionType = Main.GraphMERT.TextPosition

    entities = Vector{EntityType}()

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

        entity = EntityType(
            "entity_$(i)_$(hash(entity_text))",
            entity_text,
            entity_text,
            string(entity_type),
            "biomedical",
            Dict{String, Any}(
                "entity_type_enum" => entity_type,
                "confidence" => confidence,
            ),
            TextPositionType(start_pos, end_pos, 1, 1),
            confidence,
            text,
        )
        push!(entities, entity)
    end

    return entities
end

# Export
export extract_biomedical_entities
export validate_biomedical_entity
export calculate_biomedical_entity_confidence
export BiomedicalEntityType
export parse_entity_type
export extract_entities_from_text
export get_entity_patterns
export classify_by_rules
export calculate_entity_confidence
