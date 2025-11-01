"""
Wikipedia Entity Extraction for GraphMERT.jl

This module provides Wikipedia-specific entity extraction functionality
for general knowledge entities (people, places, organizations, concepts, etc.).
"""

using ..GraphMERT: Entity, TextPosition, ProcessingOptions

"""
    extract_wikipedia_entities(text::String, config::ProcessingOptions, domain::WikipediaDomain)

Extract Wikipedia entities from text using pattern matching.

# Arguments
- `text::String`: Input text
- `config::ProcessingOptions`: Processing options
- `domain::WikipediaDomain`: Domain provider instance

# Returns
- `Vector{Entity}`: Extracted entities
"""
function extract_wikipedia_entities(text::String, config::Any, domain::Any)
    entities = Vector{Entity}()
    
    # Split text into words
    words = split(text)
    
    # Patterns for different entity types
    entity_patterns = Dict(
        "PERSON" => [
            r"\b[A-Z][a-z]+ [A-Z][a-z]+",  # First Last
            r"\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+",  # First Middle Last
            r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.) [A-Z][a-z]+",  # Titles
        ],
        "ORGANIZATION" => [
            r"\b[A-Z][a-z]+ (?:University|College|Corporation|Corp|Inc|Ltd|Company|Organization|Foundation|Institute)",  # Organizations
            r"\b[A-Z][a-z]+ (?:Bank|Hospital|Museum|Library|Theater)",  # Institutions
        ],
        "LOCATION" => [
            r"\b[A-Z][a-z]+ (?:City|Town|Village|State|Province|Country|Island|Mountain|River|Lake|Ocean)",  # Geographic locations
            r"\b(?:North|South|East|West) [A-Z][a-z]+",  # Directions
        ],
        "COUNTRY" => [
            r"\b[A-Z][a-z]+(?:ia|land|stan|onia|ia|istan)\b",  # Country name patterns
        ],
        "EVENT" => [
            r"\b(?:the|The) [A-Z][a-z]+ (?:War|Revolution|Renaissance|Reformation|Renaissance)",  # Historical events
        ],
        "TECHNOLOGY" => [
            r"\b[A-Z][a-z]+ (?:Machine|Computer|System|Technology|Platform|Framework)",  # Technologies
        ],
        "ARTWORK" => [
            r"\b(?:the|The) [A-Z][a-z]+",  # Artworks (often start with "the")
        ],
        "CONCEPT" => [
            r"\b[A-Z][a-z]+(?:ism|ology|ity|tion|ment)\b",  # Concepts ending in common suffixes
        ],
    )
    
    entity_id = 1
    
    # Extract entities using patterns
    for (entity_type, patterns) in entity_patterns
        for pattern in patterns
            matches = eachmatch(pattern, text, overlap=false)
            for match in matches
                entity_text = String(match.match)
                
                # Skip if too short or too long
                if length(entity_text) < 3 || length(entity_text) > 100
                    continue
                end
                
                # Validate entity
                if !validate_wikipedia_entity(entity_text, entity_type, Dict())
                    continue
                end
                
                # Calculate confidence
                confidence = calculate_wikipedia_entity_confidence(entity_text, entity_type, Dict())
                
                # Skip if confidence too low
                if confidence < 0.3
                    continue
                end
                
                # Find position
                pos = findfirst(entity_text, text)
                start_pos = pos !== nothing ? first(pos) : 1
                end_pos = pos !== nothing ? last(pos) : length(entity_text)
                
                entity = Entity(
                    "entity_$(entity_id)_$(hash(entity_text))",
                    entity_text,
                    entity_text,
                    entity_type,
                    "wikipedia",
                    Dict{String, Any}(
                        "confidence" => confidence,
                        "extracted_by" => "pattern_matching",
                    ),
                    TextPosition(start_pos, end_pos, 1, 1),
                    confidence,
                    text,
                )
                push!(entities, entity)
                entity_id += 1
            end
        end
    end
    
    # Also extract capitalized proper nouns (potential entities)
    for (i, word) in enumerate(words)
        if length(word) > 3 && isuppercase(word[1]) && !isuppercase(word[2])
            # Check if it's not already extracted
            if !any(e -> e.text == word, entities)
                confidence = 0.4  # Lower confidence for single words
                
                pos = findfirst(word, text)
                start_pos = pos !== nothing ? first(pos) : 1
                end_pos = pos !== nothing ? last(pos) : length(word)
                
                entity = Entity(
                    "entity_$(entity_id)_$(hash(word))",
                    word,
                    word,
                    "CONCEPT",  # Default to CONCEPT for unknown proper nouns
                    "wikipedia",
                    Dict{String, Any}(
                        "confidence" => confidence,
                        "extracted_by" => "capitalization",
                    ),
                    TextPosition(start_pos, end_pos, 1, 1),
                    confidence,
                    text,
                )
                push!(entities, entity)
                entity_id += 1
            end
        end
    end
    
    # Remove duplicates
    unique_entities = Vector{Entity}()
    seen_texts = Set{String}()
    for entity in entities
        if !(entity.text in seen_texts)
            push!(unique_entities, entity)
            push!(seen_texts, entity.text)
        end
    end
    
    # Sort by confidence
    sort!(unique_entities, by = e -> e.confidence, rev = true)
    
    return unique_entities
end

"""
    validate_wikipedia_entity(entity_text::String, entity_type::String, context::Dict)

Validate a Wikipedia entity.
"""
function validate_wikipedia_entity(entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Basic validation
    if length(entity_text) < 2 || length(entity_text) > 100
        return false
    end
    
    # Type-specific validation
    if entity_type == "PERSON"
        # Person names should be capitalized
        return isuppercase(entity_text[1])
    elseif entity_type == "ORGANIZATION"
        # Organizations often have specific suffixes
        return occursin(r"(?:University|College|Corp|Inc|Ltd|Company|Organization|Foundation|Institute|Bank|Hospital|Museum|Library|Theater)", entity_text)
    elseif entity_type == "LOCATION" || entity_type == "COUNTRY"
        # Locations should be capitalized
        return isuppercase(entity_text[1])
    elseif entity_type == "CONCEPT"
        # Concepts often have specific suffixes
        return occursin(r"(?:ism|ology|ity|tion|ment)$", entity_text)
    end
    
    return true
end

"""
    calculate_wikipedia_entity_confidence(entity_text::String, entity_type::String, context::Dict)

Calculate confidence for a Wikipedia entity.
"""
function calculate_wikipedia_entity_confidence(entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    confidence = 0.5
    
    # Length bonus
    text_length = length(entity_text)
    if 3 <= text_length <= 50
        confidence += 0.2
    elseif 50 < text_length <= 100
        confidence += 0.1
    end
    
    # Capitalization bonus
    if isuppercase(entity_text[1])
        confidence += 0.1
    end
    
    # Type-specific bonuses
    if entity_type == "PERSON"
        # Multi-word names get higher confidence
        if count(c -> c == ' ', entity_text) >= 1
            confidence += 0.1
        end
    elseif entity_type == "ORGANIZATION"
        # Known organization suffixes
        if occursin(r"(?:University|College|Corp|Inc|Ltd|Company)", entity_text)
            confidence += 0.15
        end
    elseif entity_type == "LOCATION" || entity_type == "COUNTRY"
        # Geographic indicators
        if occursin(r"(?:City|Town|Country|Island|Mountain|River)", entity_text)
            confidence += 0.15
        end
    end
    
    return min(confidence, 1.0)
end

# Export
export extract_wikipedia_entities
export validate_wikipedia_entity
export calculate_wikipedia_entity_confidence
