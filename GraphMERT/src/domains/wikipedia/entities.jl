"""
Wikipedia Entity Extraction for GraphMERT.jl

This module provides Wikipedia-specific entity extraction functionality
for general knowledge entities (people, places, organizations, concepts, etc.).
"""

# Types assumed to be available in global scope when included

"""
    extract_wikipedia_entities(text::String, config::ProcessingOptions, domain::WikipediaDomain, llm_client::Any=nothing)

Extract Wikipedia entities from text using pattern matching or LocalLLM.
If llm_client is provided, uses LocalLLM for entity discovery instead of hardcoded list.

# Arguments
- `text::String`: Input text
- `config::ProcessingOptions`: Processing options
- `domain::WikipediaDomain`: Domain provider instance
- `llm_client::Any`: Optional LocalLLM client for LLM-based extraction

# Returns
- `Vector{Entity}`: Extracted entities
"""
function extract_wikipedia_entities(text::String, config::ProcessingOptions, domain::WikipediaDomain, llm_client::Any=nothing)
    # If llm_client is provided, use LLM for entity discovery (LocalLLM / llama-cpp)
    if llm_client !== nothing && config.use_local
        return _extract_entities_with_llm(text, config, domain, llm_client)
    end
    
    # Default: use hardcoded entity list
    entities = Vector{Entity}()
    
    # Known entities for French monarchy domain - high confidence
    known_entities = Dict{String, String}(
        # French Kings
        "Louis XIV" => "PERSON",
        "Louis XV" => "PERSON",
        "Louis XVI" => "PERSON",
        "Louis XVII" => "PERSON",
        "Louis XIII" => "PERSON",
        "Louis XII" => "PERSON",
        "Louis XI" => "PERSON",
        "Louis X" => "PERSON",
        "Louis IX" => "PERSON",
        "Henry IV" => "PERSON",
        "Henry III" => "PERSON",
        "Henry II" => "PERSON",
        "Charles V" => "PERSON",
        "Charles VI" => "PERSON",
        "Charles VII" => "PERSON",
        "Charles VIII" => "PERSON",
        "Charles IX" => "PERSON",
        "Francis I" => "PERSON",
        "Francis II" => "PERSON",
        "Philip II" => "PERSON",
        # French Queens & Consorts
        "Marie Antoinette" => "PERSON",
        "Maria Theresa of Spain" => "PERSON",
        "Maria Theresa of Austria" => "PERSON",
        "Marie de' Medici" => "PERSON",
        "Margaret of Valois" => "PERSON",
        "Anne of Austria" => "PERSON",
        "Catherine de' Medici" => "PERSON",
        "Marie Leszczyńska" => "PERSON",
        # Other royalty
        "Louis, Grand Dauphin" => "PERSON",
        "Grand Dauphin" => "PERSON",
        "Marie-Thérèse Charlotte" => "PERSON",
        "Louis-Charles" => "PERSON",
        "Louis-Joseph" => "PERSON",
        # Locations
        "France" => "COUNTRY",
        "Paris" => "CITY",
        "Versailles" => "CITY",
        "Palace of Versailles" => "LOCATION",
        "Château de Saint-Germain-en-Laye" => "LOCATION",
        "Saint-Germain-en-Laye" => "CITY",
        "Pau" => "CITY",
        "Reims" => "CITY",
        "Blois" => "CITY",
        "Fontainebleau" => "CITY",
        "Louvre" => "LOCATION",
        # Dynasties
        "Bourbon" => "ORGANIZATION",
        "House of Bourbon" => "ORGANIZATION",
        "House of Valois" => "ORGANIZATION",
        "Valois" => "ORGANIZATION",
        "Capetian" => "ORGANIZATION",
        "Orléans" => "ORGANIZATION",
        # Events
        "French Revolution" => "EVENT",
        "War of the Spanish Succession" => "EVENT",
    )
    
    entity_id = 1
    
    # First pass: Extract known entities with high confidence
    for (entity_text, entity_type) in known_entities
        if occursin(entity_text, text)
            @info "Found known entity: $entity_text"
            pos = findfirst(entity_text, text)
            start_pos = pos !== nothing ? first(pos) : 1
            end_pos = pos !== nothing ? last(pos) : length(entity_text)
            
            confidence = 0.95  # High confidence for known entities
            
            entity = Entity(
                "entity_$(entity_id)_$(hash(entity_text))",
                entity_text,
                entity_text,
                entity_type,
                "wikipedia",
                Dict{String, Any}(
                    "confidence" => confidence,
                    "extracted_by" => "known_entity",
                ),
                TextPosition(start_pos, end_pos, 1, 1),
                confidence,
                text,
            )
            push!(entities, entity)
            entity_id += 1
        end
    end
    
    # Split text into words
    words = split(text)
    
    # Patterns for different entity types
    entity_patterns = Dict(
        "PERSON" => [
            r"\b[A-Z][a-z]+ [A-Z][a-z]+",  # First Last
            r"\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+",  # First Middle Last
            r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.) [A-Z][a-z]+",  # Titles
            r"\b(?:Louis|Henry|Charles|Francis|Philip) (?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII)\b",  # French kings with Roman numerals
            r"\b[A-Z][a-z]+ (?=of\b)",  # Names followed by "of" (e.g., Louis of)
            r"\b(?:Marie|Catherine|Anne|Margaret|Maria) [A-Z][a-z]+",  # French queens/consorts
            r"\b[A-Z][a-z]+ de [A-Z][a-z]+",  # Names with "de" (e.g., Marie de' Medici)
            r"\b[A-Z][a-z]+' [A-Z][a-z]+",  # Names with apostrophe
        ],
        "TITLE" => [
            r"\b(?:King|Queen|Kaiser|Emperor|Empress) of [A-Z][a-z]+",  # Royal titles
            r"\b(?:the )?(?:Sun King|Henry the Great|King of France|Queen of France)\b",  # Known titles
        ],
        "ORGANIZATION" => [
            r"\b[A-Z][a-z]+ (?:University|College|Corporation|Corp|Inc|Ltd|Company|Organization|Foundation|Institute)",  # Organizations
            r"\b[A-Z][a-z]+ (?:Bank|Hospital|Museum|Library|Theater)",  # Institutions
            r"\b(?:House of|Bourbon|Valois|Capetian|Orléans)\b",  # Dynasties as organizations
        ],
        "LOCATION" => [
            r"\b[A-Z][a-z]+ (?:City|Town|Village|State|Province|Country|Island|Mountain|River|Lake|Ocean)",  # Geographic locations
            r"\b(?:North|South|East|West) [A-Z][a-z]+",  # Directions
            r"\b(?:Château|Palace) de [A-Z][a-z]+",  # French palaces
            r"\b[A-Z][a-z]+-en-[A-Z][a-z]+",  # French place names (e.g., Saint-Germain-en-Laye)
            r"\b(?:Saint|St) [A-Z][a-z]+",  # Saint- places
        ],
        "COUNTRY" => [
            r"\b(?:France|Austria|Spain|Portugal|England|Scotland|Ireland|Germany|Italy|Poland|Sweden|Norway|Denmark|Belgium|Netherlands|Switzerland)\b",  # Major countries
            r"\b[A-Z][a-z]+(?:ia|land|stan|onia|ia|istan)\b",  # Country name patterns
        ],
        "CITY" => [
            r"\b(?:Paris|Versailles|London|Rome|Madrid|Berlin|Vienna|Augsburg|Pau|Blois|Reims|Fontainebleau|Louvre|Saint-Germain-en-Laye)\b",  # Major cities
        ],
        "EVENT" => [
            r"\b(?:the |The )?(?:French Revolution|War of the Spanish Succession|Hundred Years'|Thirty Years'|Seven Years')\b",  # Major historical events
            r"\b(?:War|Revolution|Renaissance|Reformation) [A-Z][a-z]+\b",  # General historical events
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
                entity_text = string(match.match)  # Ensure it's a String, not SubString
                
                # Skip if too short or too long
                if length(entity_text) < 3 || length(entity_text) > 100
                    continue
                end
                
                # Validate entity
                if !validate_wikipedia_entity(entity_text, entity_type, Dict{String, Any}())
                    continue
                end
                
                # Calculate confidence
                confidence = calculate_wikipedia_entity_confidence(entity_text, entity_type, Dict{String, Any}())
                
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
        word_str = string(word)  # Ensure it's a String, not SubString
        
        # Skip numbers, Roman numerals, short words
        if occursin(r"^\d+$", word_str)
            continue
        end
        if occursin(r"^(I|V|X|L|C|D|M)+$", word_str)  # Roman numerals
            continue
        end
        
        if length(word_str) > 3 && isuppercase(word_str[1]) && !isuppercase(word_str[2])
            # Check if it's not already extracted
            if !any(e -> e.text == word_str, entities)
                confidence = 0.4  # Lower confidence for single words
                
                pos = findfirst(word_str, text)
                start_pos = pos !== nothing ? first(pos) : 1
                end_pos = pos !== nothing ? last(pos) : length(word_str)
                
                # Try to determine better type for single words
                entity_type = "CONCEPT"
                if word_str in ["France", "Austria", "Spain"]
                    entity_type = "COUNTRY"
                elseif word_str in ["Paris", "Versailles", "London"]
                    entity_type = "CITY"
                end
                
                entity = Entity(
                    "entity_$(entity_id)_$(hash(word_str))",
                    word_str,
                    word_str,
                    entity_type,
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
    
    # Skip if it's just a number or Roman numeral alone
    if occursin(r"^\d+$", entity_text)
        return false
    end
    
    # Skip short words unless they're well-known place names
    if length(entity_text) <= 2 && entity_type ∉ ["CITY", "COUNTRY"]
        return false
    end
    
    # Type-specific validation
    if entity_type == "PERSON"
        # Person names should be capitalized
        return isuppercase(entity_text[1])
    elseif entity_type == "TITLE"
        # Titles contain keywords
        return occursin(r"(?:King|Queen|Emperor|Empress|the Great|Sun King)", entity_text)
    elseif entity_type == "ORGANIZATION"
        # Organizations often have specific suffixes or are dynasties
        return occursin(r"(?:University|College|Corp|Inc|Ltd|Company|Organization|Foundation|Institute|Bank|Hospital|Museum|Library|Theater|House of|Bourbon|Valois|Capetian|Orléans)", entity_text)
    elseif entity_type == "LOCATION" || entity_type == "COUNTRY" || entity_type == "CITY"
        # Locations should be capitalized
        return isuppercase(entity_text[1])
    elseif entity_type == "EVENT"
        # Events have keywords
        return occursin(r"(?:War|Revolution|Renaissance|Reformation|Succession)", entity_text)
    elseif entity_type == "CONCEPT"
        # Concepts often have specific suffixes
        return occursin(r"(?:ism|ology|ity|tion|ment)$", entity_text)
    end
    
    return true
end

"""
    _extract_entities_with_llm(text, config, domain, llm_client)

Extract entities using LocalLLM (llama-cpp / GGUF) client.
"""
function _extract_entities_with_llm(text::String, config::ProcessingOptions, domain::WikipediaDomain, llm_client::Any)
    entities = Vector{Entity}()
    
    try
        if isa(llm_client, GraphMERT.LocalLLM.LocalLLMClient)
            entity_texts = GraphMERT.LocalLLM.discover_entities(llm_client, text, "wikipedia")
        else
            throw(ArgumentError("Unknown LLM client type: $(typeof(llm_client))"))
        end
        
        @info "LLM discovered $(length(entity_texts)) entities"
        
        # Convert to Entity objects
        entity_id = 1
        for entity_text in entity_texts
            # Try to classify the entity type using patterns
            entity_type = _classify_entity_type(entity_text)
            
            pos = findfirst(entity_text, text)
            start_pos = pos !== nothing ? first(pos) : 1
            end_pos = pos !== nothing ? last(pos) : length(entity_text)
            
            # Use slightly lower confidence for LLM-extracted entities
            confidence = 0.7
            
            entity = Entity(
                "entity_llm_$(entity_id)_$(hash(entity_text))",
                entity_text,
                entity_text,
                entity_type,
                "wikipedia",
                Dict{String, Any}(
                    "confidence" => confidence,
                    "extracted_by" => "llm",
                ),
                TextPosition(start_pos, end_pos, 1, 1),
                confidence,
                text,
            )
            push!(entities, entity)
            entity_id += 1
        end
    catch e
        @warn "LLM entity extraction failed: $e. Falling back to hardcoded list."
        # Fall back to hardcoded entities
        return extract_wikipedia_entities(text, config, domain, nothing)
    end
    
    return entities
end

"""
    _classify_entity_type(entity_text::String)

Classify entity type based on patterns.
"""
function _classify_entity_type(entity_text::String)
    # Check for person names (multi-word, capitalized)
    if occursin(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$", entity_text)
        # Check for royal names
        if occursin(r"^(Louis|Henry|Charles|Francis|Philip|Marie|Anne|Catherine|Margaret)\b", entity_text)
            return "PERSON"
        end
        return "PERSON"
    end
    
    # Check for known locations
    known_cities = ["Paris", "Versailles", "London", "Rome", "Madrid", "Berlin", "Vienna", "Pau", "Blois", "Reims", "Fontainebleau", "Saint-Germain-en-Laye"]
    if entity_text in known_cities
        return "CITY"
    end
    
    # Check for countries
    known_countries = ["France", "Spain", "Austria", "England", "Germany", "Italy"]
    if entity_text in known_countries
        return "COUNTRY"
    end
    
    # Check for organizations/dynasties
    known_orgs = ["Bourbon", "Valois", "Capetian", "Orléans", "House of Bourbon", "House of Valois"]
    if entity_text in known_orgs || occursin(r"(University|College|Corp|Inc)$", entity_text)
        return "ORGANIZATION"
    end
    
    # Check for events
    known_events = ["French Revolution", "War of the Spanish Succession"]
    if entity_text in known_events
        return "EVENT"
    end
    
    # Check for titles
    if occursin(r"\bKing|Queen|Queen|King|Dauphin\b", entity_text)
        return "TITLE"
    end
    
    return "CONCEPT"
end

"""
    calculate_wikipedia_entity_confidence(entity_text::String, entity_type::String, context::Dict)

Calculate confidence for a Wikipedia entity.
"""
function calculate_wikipedia_entity_confidence(entity_text::String, entity_type::String, context::Dict{String, Any} = Dict{String, Any}())
    # Handle empty string
    if isempty(entity_text)
        return 0.0
    end
    
    # High-confidence types: PERSON, CITY, COUNTRY, EVENT, ORGANIZATION are well-extracted
    # Low-confidence type: CONCEPT is often false positive
    if entity_type == "CONCEPT"
        return 0.35  # Low confidence for generic concepts
    end
    
    confidence = 0.55  # Start higher
    
    # Length bonus - prefer medium-length names
    text_length = length(entity_text)
    if 3 <= text_length <= 30
        confidence += 0.15
    elseif 30 < text_length <= 60
        confidence += 0.1
    elseif 60 < text_length <= 100
        confidence += 0.05
    end
    
    # Capitalization bonus
    if isuppercase(entity_text[1])
        confidence += 0.1
    end
    
    # Type-specific bonuses
    if entity_type == "PERSON"
        # Multi-word names get higher confidence
        if count(c -> c == ' ', entity_text) >= 1
            confidence += 0.15
        end
        # Royal titles are very reliable
        if occursin(r"(Louis|Henry|Charles|Francis|Philip|Marie|Anne|Margaret)\b", entity_text)
            confidence += 0.1
        end
    elseif entity_type == "ORGANIZATION"
        # Known organization suffixes
        if occursin(r"(University|College|Corp|Inc|Ltd|Company|Bourbon|Valois|Capetian)", entity_text)
            confidence += 0.15
        end
    elseif entity_type == "LOCATION" || entity_type == "COUNTRY" || entity_type == "CITY"
        # Geographic indicators
        if occursin(r"(City|Town|Country|Island|Mountain|River|Palace|Château)", entity_text)
            confidence += 0.15
        end
        # Known cities and countries
        if entity_text in ["France", "Paris", "Versailles", "London", "Spain", "Austria"]
            confidence += 0.15
        end
    elseif entity_type == "EVENT"
        confidence += 0.15  # Events are usually well-identified
    elseif entity_type == "TITLE"
        confidence += 0.15
    end
    
    return min(confidence, 1.0)
end

# Export
export extract_wikipedia_entities
export validate_wikipedia_entity
export calculate_wikipedia_entity_confidence
