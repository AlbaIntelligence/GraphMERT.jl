"""
Biomedical entity types for GraphMERT.jl

This module defines biomedical entity types and their classification
as specified in the GraphMERT paper for biomedical knowledge graph construction.
"""

using Dates
using Random

# ============================================================================
# Biomedical Entity Types
# ============================================================================

"""
    BiomedicalEntityType

Enumeration of supported biomedical entity types.
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
    UNKNOWN
end

"""
    get_entity_type_name(entity_type::BiomedicalEntityType)

Get the string name of an entity type.
"""
function get_entity_type_name(entity_type::BiomedicalEntityType)
    return string(entity_type)
end

"""
    parse_entity_type(type_name::String)

Parse a string to a biomedical entity type.
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
    else
        return UNKNOWN
    end
end

# ============================================================================
# Entity Classification
# ============================================================================

"""
    classify_entity(entity_text::String; umls_client=nothing)

Classify a biomedical entity text into its appropriate type.
"""
function classify_entity(entity_text::String; umls_client=nothing)
    entity_text_lower = lowercase(entity_text)
    
    # Try UMLS classification first if client is available
    if umls_client !== nothing
        try
            cui = get_entity_cui(umls_client, entity_text)
            if cui !== nothing
                semantic_types = get_entity_semantic_types(umls_client, cui)
                if !isempty(semantic_types)
                    return classify_by_semantic_types(semantic_types)
                end
            end
        catch e
            @warn "UMLS classification failed: $e"
        end
    end
    
    # Fallback to rule-based classification
    return classify_by_rules(entity_text_lower)
end

"""
    classify_by_semantic_types(semantic_types::Vector{String})

Classify entity based on UMLS semantic types.
"""
function classify_by_semantic_types(semantic_types::Vector{String})
    # Map UMLS semantic types to our entity types
    for st in semantic_types
        st_lower = lowercase(st)
        
        if occursin("disease", st_lower) || occursin("disorder", st_lower)
            return DISEASE
        elseif occursin("pharmacologic", st_lower) || occursin("drug", st_lower)
            return DRUG
        elseif occursin("protein", st_lower) || occursin("enzyme", st_lower)
            return PROTEIN
        elseif occursin("gene", st_lower) || occursin("genetic", st_lower)
            return GENE
        elseif occursin("anatomy", st_lower) || occursin("body", st_lower)
            return ANATOMY
        elseif occursin("symptom", st_lower) || occursin("sign", st_lower)
            return SYMPTOM
        elseif occursin("procedure", st_lower) || occursin("surgery", st_lower)
            return PROCEDURE
        elseif occursin("organism", st_lower) || occursin("bacteria", st_lower)
            return ORGANISM
        elseif occursin("chemical", st_lower) || occursin("compound", st_lower)
            return CHEMICAL
        elseif occursin("cell", st_lower)
            return CELL_TYPE
        elseif occursin("function", st_lower)
            return MOLECULAR_FUNCTION
        elseif occursin("process", st_lower)
            return BIOLOGICAL_PROCESS
        elseif occursin("component", st_lower)
            return CELLULAR_COMPONENT
        end
    end
    
    return UNKNOWN
end

"""
    classify_by_rules(entity_text_lower::String)

Classify entity using rule-based patterns.
"""
function classify_by_rules(entity_text_lower::String)
    # Disease patterns
    if occursin(r"\b(disease|disorder|syndrome|condition|illness|pathology)\b", entity_text_lower)
        return DISEASE
    end
    
    # Drug patterns
    if occursin(r"\b(drug|medication|medicine|pharmaceutical|therapeutic|treatment)\b", entity_text_lower)
        return DRUG
    end
    
    # Protein patterns
    if occursin(r"\b(protein|enzyme|receptor|antibody|hormone|kinase|phosphatase)\b", entity_text_lower)
        return PROTEIN
    end
    
    # Gene patterns
    if occursin(r"\b(gene|genetic|allele|mutation|variant|polymorphism)\b", entity_text_lower)
        return GENE
    end
    
    # Anatomy patterns
    if occursin(r"\b(organ|tissue|muscle|bone|nerve|vessel|gland|organelle)\b", entity_text_lower)
        return ANATOMY
    end
    
    # Symptom patterns
    if occursin(r"\b(symptom|sign|manifestation|complaint|pain|fever|nausea)\b", entity_text_lower)
        return SYMPTOM
    end
    
    # Procedure patterns
    if occursin(r"\b(procedure|surgery|operation|treatment|therapy|intervention)\b", entity_text_lower)
        return PROCEDURE
    end
    
    # Organism patterns
    if occursin(r"\b(bacteria|virus|fungus|parasite|microorganism|pathogen)\b", entity_text_lower)
        return ORGANISM
    end
    
    # Chemical patterns
    if occursin(r"\b(chemical|compound|molecule|substance|element|ion)\b", entity_text_lower)
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

Validate that an entity text matches its claimed type.
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
        return occursin(r"\b(disease|disorder|syndrome|condition|illness|pathology)\b", entity_text_lower)
    elseif entity_type == DRUG
        return occursin(r"\b(drug|medication|medicine|pharmaceutical|therapeutic|treatment)\b", entity_text_lower)
    elseif entity_type == PROTEIN
        return occursin(r"\b(protein|enzyme|receptor|antibody|hormone|kinase|phosphatase)\b", entity_text_lower)
    elseif entity_type == GENE
        return occursin(r"\b(gene|genetic|allele|mutation|variant|polymorphism)\b", entity_text_lower)
    elseif entity_type == ANATOMY
        return occursin(r"\b(organ|tissue|muscle|bone|nerve|vessel|gland|organelle)\b", entity_text_lower)
    elseif entity_type == SYMPTOM
        return occursin(r"\b(symptom|sign|manifestation|complaint|pain|fever|nausea)\b", entity_text_lower)
    elseif entity_type == PROCEDURE
        return occursin(r"\b(procedure|surgery|operation|treatment|therapy|intervention)\b", entity_text_lower)
    elseif entity_type == ORGANISM
        return occursin(r"\b(bacteria|virus|fungus|parasite|microorganism|pathogen)\b", entity_text_lower)
    elseif entity_type == CHEMICAL
        return occursin(r"\b(chemical|compound|molecule|substance|element|ion)\b", entity_text_lower)
    elseif entity_type == CELL_TYPE
        return occursin(r"\b(cell|tissue|line|culture|strain)\b", entity_text_lower)
    else
        return true  # Unknown type is always valid
    end
end

# ============================================================================
# Entity Normalization
# ============================================================================

"""
    normalize_entity_text(entity_text::String)

Normalize entity text for consistent processing.
"""
function normalize_entity_text(entity_text::String)
    # Remove extra whitespace
    normalized = strip(entity_text)
    
    # Remove common prefixes/suffixes
    normalized = replace(normalized, r"^(the|a|an)\s+"i => "")
    normalized = replace(normalized, r"\s+(disease|disorder|syndrome|condition)$"i => "")
    
    # Normalize case for common biomedical terms
    normalized = replace(normalized, r"\bDNA\b" => "DNA")
    normalized = replace(normalized, r"\bRNA\b" => "RNA")
    normalized = replace(normalized, r"\bATP\b" => "ATP")
    normalized = replace(normalized, r"\bGTP\b" => "GTP")
    
    return normalized
end

# ============================================================================
# Entity Confidence Scoring
# ============================================================================

"""
    calculate_entity_confidence(entity_text::String, entity_type::BiomedicalEntityType)

Calculate confidence score for entity classification.
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
    
    # Medical terminology bonus
    if occursin(r"\b(medical|clinical|biomedical|scientific)\b", entity_text_lower)
        confidence += 0.1
    end
    
    # UMLS integration bonus (if available)
    # This would be added when UMLS client is available
    
    return min(confidence, 1.0)
end

# ============================================================================
# Entity Extraction Patterns
# ============================================================================

"""
    extract_entities_from_text(text::String; entity_types::Vector{BiomedicalEntityType}=BiomedicalEntityType[])

Extract biomedical entities from text using pattern matching.
"""
function extract_entities_from_text(text::String; entity_types::Vector{BiomedicalEntityType}=BiomedicalEntityType[])
    if isempty(entity_types)
        entity_types = [DISEASE, DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM, PROCEDURE, ORGANISM, CHEMICAL, CELL_TYPE]
    end
    
    entities = Vector{Tuple{String, BiomedicalEntityType, Float64}}()
    
    for entity_type in entity_types
        patterns = get_entity_patterns(entity_type)
        
        for pattern in patterns
            matches = eachmatch(pattern, text, overlap=false)
            for match in matches
                entity_text = match.match
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

"""
    get_entity_patterns(entity_type::BiomedicalEntityType)

Get regex patterns for extracting entities of a specific type.
"""
function get_entity_patterns(entity_type::BiomedicalEntityType)
    patterns = Regex[]
    
    if entity_type == DISEASE
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|disorder|syndrome|condition|illness|pathology)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:syndrome|disease)\b")
    elseif entity_type == DRUG
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:drug|medication|medicine|pharmaceutical)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:treatment|therapy)\b")
    elseif entity_type == PROTEIN
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:protein|enzyme|receptor|antibody|hormone)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:kinase|phosphatase|transferase)\b")
    elseif entity_type == GENE
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:gene|genetic|allele|mutation|variant)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:polymorphism|SNP|variant)\b")
    elseif entity_type == ANATOMY
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:organ|tissue|muscle|bone|nerve|vessel|gland)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:organelle|membrane|nucleus|mitochondria)\b")
    elseif entity_type == SYMPTOM
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:symptom|sign|manifestation|complaint)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:pain|fever|nausea|headache)\b")
    elseif entity_type == PROCEDURE
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:procedure|surgery|operation|treatment)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:therapy|intervention|therapy)\b")
    elseif entity_type == ORGANISM
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:bacteria|virus|fungus|parasite|microorganism)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:pathogen|organism|species)\b")
    elseif entity_type == CHEMICAL
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:chemical|compound|molecule|substance)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:element|ion|salt|acid|base)\b")
    elseif entity_type == CELL_TYPE
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:cell|tissue|line|culture|strain)\b")
        push!(patterns, r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:type|kind|form|variant)\b")
    end
    
    return patterns
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    get_entity_type_description(entity_type::BiomedicalEntityType)

Get a description of an entity type.
"""
function get_entity_type_description(entity_type::BiomedicalEntityType)
    descriptions = Dict{BiomedicalEntityType, String}(
        DISEASE => "Diseases, disorders, syndromes, and medical conditions",
        DRUG => "Pharmaceutical drugs, medications, and therapeutic agents",
        PROTEIN => "Proteins, enzymes, receptors, antibodies, and hormones",
        GENE => "Genes, genetic variants, mutations, and polymorphisms",
        ANATOMY => "Anatomical structures, organs, tissues, and body parts",
        SYMPTOM => "Symptoms, signs, and clinical manifestations",
        PROCEDURE => "Medical procedures, surgeries, and treatments",
        ORGANISM => "Microorganisms, bacteria, viruses, and pathogens",
        CHEMICAL => "Chemical compounds, molecules, and substances",
        CELL_TYPE => "Cell types, cell lines, and cellular structures",
        MOLECULAR_FUNCTION => "Molecular functions and biochemical activities",
        BIOLOGICAL_PROCESS => "Biological processes and pathways",
        CELLULAR_COMPONENT => "Cellular components and organelles",
        UNKNOWN => "Unknown or unclassified entity type"
    )
    
    return get(descriptions, entity_type, "Unknown entity type")
end

"""
    get_supported_entity_types()

Get all supported biomedical entity types.
"""
function get_supported_entity_types()
    return [DISEASE, DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM, PROCEDURE, ORGANISM, CHEMICAL, CELL_TYPE, MOLECULAR_FUNCTION, BIOLOGICAL_PROCESS, CELLULAR_COMPONENT]
end
