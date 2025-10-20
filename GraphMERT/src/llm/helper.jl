"""
Helper LLM integration for entity discovery and relation matching.
"""

# Placeholder implementations without HTTP dependencies

"""
    HelperLLMConfig

Configuration for helper LLM client.
"""
struct HelperLLMConfig
    api_key::String
    base_url::String
    model::String
    timeout::Int
    max_retries::Int
    temperature::Float64
    max_tokens::Int
end

"""
    HelperLLMResponse

Response from helper LLM.
"""
struct HelperLLMResponse
    success::Bool
    content::String
    error::Union{String, Nothing}
    usage::Dict{String, Any}
end

"""
    HelperLLMCache

Local cache for LLM responses.
"""
struct HelperLLMCache
    responses::Dict{String, HelperLLMResponse}
    max_size::Int
end

"""
    HelperLLMClient

Helper LLM client with rate limiting and caching.
"""
struct HelperLLMClient
    config::HelperLLMConfig
    cache::HelperLLMCache
    last_request_time::Float64
    request_count::Int
end

"""
    create_helper_llm_client(api_key::String; kwargs...)

Create a new helper LLM client.
"""
function create_helper_llm_client(api_key::String; 
                                 base_url::String = "https://api.openai.com/v1",
                                 model::String = "gpt-3.5-turbo",
                                 timeout::Int = 30,
                                 max_retries::Int = 3,
                                 temperature::Float64 = 0.7,
                                 max_tokens::Int = 1000)
    config = HelperLLMConfig(api_key, base_url, model, timeout, max_retries, temperature, max_tokens)
    cache = HelperLLMCache(Dict{String, HelperLLMResponse}(), 1000)
    return HelperLLMClient(config, cache, 0.0, 0)
end

"""
    make_llm_request(client::HelperLLMClient, prompt::String)

Make a request to the helper LLM.
"""
function make_llm_request(client::HelperLLMClient, prompt::String)
    @warn "LLM requests not available - HTTP.jl not loaded"
    return HelperLLMResponse(false, "", "HTTP.jl not available", Dict{String, Any}())
end

"""
    discover_entities(client::HelperLLMClient, text::String)

Discover entities in text using LLM.
"""
function discover_entities(client::HelperLLMClient, text::String)
    @warn "Entity discovery not available - HTTP.jl not loaded"
    return String[]
end

"""
    match_relations(client::HelperLLMClient, entities::Vector{String}, text::String)

Match relations between entities using LLM.
"""
function match_relations(client::HelperLLMClient, entities::Vector{String}, text::String)
    @warn "Relation matching not available - HTTP.jl not loaded"
    return Dict{String, Any}()
end

"""
    create_entity_discovery_prompt(text::String)

Create a prompt for entity discovery.
"""
function create_entity_discovery_prompt(text::String)
    return """
    Extract biomedical entities from the following text. Return only the entity names, one per line:
    
    Text: $text
    
    Entities:
    """
end

"""
    create_relation_matching_prompt(entities::Vector{String}, text::String)

Create a prompt for relation matching.
"""
function create_relation_matching_prompt(entities::Vector{String}, text::String)
    entities_str = join(entities, ", ")
    return """
    Find relations between these entities in the text. Return in format: entity1 -> relation -> entity2
    
    Entities: $entities_str
    Text: $text
    
    Relations:
    """
end

"""
    parse_entity_response(response::String)

Parse entity discovery response.
"""
function parse_entity_response(response::String)
    entities = String[]
    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) && !startswith(line, "#")
            push!(entities, line)
        end
    end
    return entities
end

"""
    parse_relation_response(response::String)

Parse relation matching response.
"""
function parse_relation_response(response::String)
    relations = Dict{String, Any}()
    for line in split(response, '\n')
        line = strip(line)
        if !isempty(line) && !startswith(line, "#") && occursin("->", line)
            parts = split(line, "->")
            if length(parts) == 3
                entity1 = strip(parts[1])
                relation = strip(parts[2])
                entity2 = strip(parts[3])
                relations["$entity1-$entity2"] = Dict(
                    "entity1" => entity1,
                    "relation" => relation,
                    "entity2" => entity2
                )
            end
        end
    end
    return relations
end

"""
    discover_entities_batch(client::HelperLLMClient, texts::Vector{String})

Discover entities in multiple texts.
"""
function discover_entities_batch(client::HelperLLMClient, texts::Vector{String})
    @warn "Batch entity discovery not available - HTTP.jl not loaded"
    return Vector{String}[]
end

"""
    match_relations_batch(client::HelperLLMClient, entity_lists::Vector{Vector{String}}, texts::Vector{String})

Match relations in multiple texts.
"""
function match_relations_batch(client::HelperLLMClient, entity_lists::Vector{Vector{String}}, texts::Vector{String})
    @warn "Batch relation matching not available - HTTP.jl not loaded"
    return Vector{Dict{String, Any}}[]
end

"""
    fallback_entity_discovery(text::String)

Fallback entity discovery when LLM is not available.
"""
function fallback_entity_discovery(text::String)
    # Simple regex-based entity discovery
    entities = String[]
    
    # Look for common biomedical patterns
    patterns = [
        r"\b[A-Z][a-z]+(?:'s)?\s+(?:disease|syndrome|disorder|condition)\b",
        r"\b[A-Z][a-z]+(?:'s)?\s+(?:cancer|carcinoma|tumor|neoplasm)\b",
        r"\b[A-Z][a-z]+(?:'s)?\s+(?:virus|bacteria|infection)\b",
        r"\b[A-Z][a-z]+(?:'s)?\s+(?:protein|enzyme|receptor)\b",
        r"\b[A-Z][a-z]+(?:'s)?\s+(?:drug|medication|therapy)\b"
    ]
    
    for pattern in patterns
        matches = eachmatch(pattern, text)
        for match in matches
            push!(entities, match.match)
        end
    end
    
    return entities
end

"""
    fallback_relation_matching(entities::Vector{String}, text::String)

Fallback relation matching when LLM is not available.
"""
function fallback_relation_matching(entities::Vector{String}, text::String)
    relations = Dict{String, Any}()
    
    # Simple co-occurrence based relation detection
    for i in 1:length(entities)
        for j in (i+1):length(entities)
            entity1 = entities[i]
            entity2 = entities[j]
            
            # Check if both entities appear in the same sentence
            sentences = split(text, r"[\.\!\?]+")
            for sentence in sentences
                if occursin(entity1, sentence) && occursin(entity2, sentence)
                    # Simple relation detection based on keywords
                    if occursin("treats", lowercase(sentence)) || occursin("cures", lowercase(sentence))
                        relation = "TREATS"
                    elseif occursin("causes", lowercase(sentence)) || occursin("leads to", lowercase(sentence))
                        relation = "CAUSES"
                    elseif occursin("associated with", lowercase(sentence)) || occursin("related to", lowercase(sentence))
                        relation = "ASSOCIATED_WITH"
                    else
                        relation = "RELATED_TO"
                    end
                    
                    relations["$entity1-$entity2"] = Dict(
                        "entity1" => entity1,
                        "relation" => relation,
                        "entity2" => entity2
                    )
                end
            end
        end
    end
    
    return relations
end
