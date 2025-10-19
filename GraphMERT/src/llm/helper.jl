"""
Helper LLM integration for GraphMERT.jl

This module implements helper LLM integration for entity discovery and relation matching
as specified in the GraphMERT paper for biomedical knowledge graph construction.
"""

using HTTP
using JSON
using Logging
using Random

# ============================================================================
# Helper LLM Configuration
# ============================================================================

"""
    HelperLLMConfig

Configuration for helper LLM integration.
"""
struct HelperLLMConfig
    api_key::String
    base_url::String
    model::String
    max_tokens::Int
    temperature::Float64
    rate_limit::Int
    timeout::Int
    retry_attempts::Int
    cache_enabled::Bool
    cache_ttl::Int

    function HelperLLMConfig(;
        api_key::String="",
        base_url::String="https://api.openai.com/v1",
        model::String="gpt-4",
        max_tokens::Int=1000,
        temperature::Float64=0.1,
        rate_limit::Int=10000,  # tokens per minute
        timeout::Int=60,
        retry_attempts::Int=3,
        cache_enabled::Bool=true,
        cache_ttl::Int=3600  # 1 hour
    )
        @assert !isempty(api_key) "Helper LLM API key is required"
        @assert max_tokens > 0 "Max tokens must be positive"
        @assert 0.0 <= temperature <= 2.0 "Temperature must be between 0.0 and 2.0"
        @assert rate_limit > 0 "Rate limit must be positive"
        @assert timeout > 0 "Timeout must be positive"
        @assert retry_attempts > 0 "Retry attempts must be positive"
        @assert cache_ttl > 0 "Cache TTL must be positive"

        new(api_key, base_url, model, max_tokens, temperature, rate_limit, timeout, retry_attempts, cache_enabled, cache_ttl)
    end
end

"""
    HelperLLMResponse

Response from helper LLM API.
"""
struct HelperLLMResponse
    success::Bool
    content::String
    usage::Dict{String,Any}
    error::String
    status_code::Int
    timestamp::DateTime

    function HelperLLMResponse(success::Bool, content::String, usage::Dict{String,Any}, error::String="", status_code::Int=200)
        new(success, content, usage, error, status_code, now())
    end
end

"""
    HelperLLMCache

Simple in-memory cache for helper LLM responses.
"""
mutable struct HelperLLMCache
    data::Dict{String,Any}
    timestamps::Dict{String,DateTime}
    ttl::Int

    function HelperLLMCache(ttl::Int=3600)
        new(Dict{String,Any}(), Dict{String,DateTime}(), ttl)
    end
end

# ============================================================================
# Helper LLM Client
# ============================================================================

"""
    HelperLLMClient

Client for helper LLM integration with rate limiting and caching.
"""
mutable struct HelperLLMClient
    config::HelperLLMConfig
    cache::HelperLLMCache
    last_request_time::DateTime
    token_count::Int
    rate_limit_window::DateTime

    function HelperLLMClient(config::HelperLLMConfig)
        cache = HelperLLMCache(config.cache_ttl)
        new(config, cache, now(), 0, now())
    end
end

"""
    create_helper_llm_client(api_key::String; kwargs...)

Create a new helper LLM client with the given API key.
"""
function create_helper_llm_client(api_key::String; kwargs...)
    config = HelperLLMConfig(; api_key=api_key, kwargs...)
    return HelperLLMClient(config)
end

# ============================================================================
# Rate Limiting
# ============================================================================

"""
    check_rate_limit(client::HelperLLMClient, estimated_tokens::Int)

Check if we can make a request without exceeding rate limits.
"""
function check_rate_limit(client::HelperLLMClient, estimated_tokens::Int)
    current_time = now()
    
    # Reset counter if we're in a new minute
    if current_time - client.rate_limit_window >= Minute(1)
        client.token_count = 0
        client.rate_limit_window = current_time
    end
    
    return (client.token_count + estimated_tokens) < client.config.rate_limit
end

"""
    wait_for_rate_limit(client::HelperLLMClient, estimated_tokens::Int)

Wait until we can make a request without exceeding rate limits.
"""
function wait_for_rate_limit(client::HelperLLMClient, estimated_tokens::Int)
    while !check_rate_limit(client, estimated_tokens)
        sleep(1)  # Wait 1 second
    end
end

# ============================================================================
# Caching
# ============================================================================

"""
    get_from_cache(cache::HelperLLMCache, key::String)

Get a value from the cache if it exists and hasn't expired.
"""
function get_from_cache(cache::HelperLLMCache, key::String)
    if !haskey(cache.data, key)
        return nothing
    end
    
    if haskey(cache.timestamps, key)
        age = now() - cache.timestamps[key]
        if age >= Second(cache.ttl)
            delete!(cache.data, key)
            delete!(cache.timestamps, key)
            return nothing
        end
    end
    
    return cache.data[key]
end

"""
    set_in_cache(cache::HelperLLMCache, key::String, value::Any)

Set a value in the cache with current timestamp.
"""
function set_in_cache(cache::HelperLLMCache, key::String, value::Any)
    cache.data[key] = value
    cache.timestamps[key] = now()
end

# ============================================================================
# HTTP Requests
# ============================================================================

"""
    make_llm_request(client::HelperLLMClient, messages::Vector{Dict{String,Any}})

Make a request to the helper LLM API with rate limiting and error handling.
"""
function make_llm_request(client::HelperLLMClient, messages::Vector{Dict{String,Any}})
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = sum(length(msg["content"]) for msg in messages) ÷ 4
    
    # Check rate limit
    wait_for_rate_limit(client, estimated_tokens)
    
    # Check cache first
    cache_key = "llm:$(hash(messages))"
    if client.config.cache_enabled
        cached_result = get_from_cache(client.cache, cache_key)
        if cached_result !== nothing
            @debug "Helper LLM cache hit"
            return HelperLLMResponse(true, cached_result["content"], cached_result["usage"])
        end
    end
    
    # Make request with retries
    for attempt in 1:client.config.retry_attempts
        try
            @debug "Making helper LLM request (attempt $attempt)"
            
            headers = [
                "Authorization" => "Bearer $(client.config.api_key)",
                "Content-Type" => "application/json"
            ]
            
            body = Dict{String,Any}(
                "model" => client.config.model,
                "messages" => messages,
                "max_tokens" => client.config.max_tokens,
                "temperature" => client.config.temperature
            )
            
            response = HTTP.post(
                "$(client.config.base_url)/chat/completions",
                headers,
                JSON.json(body);
                timeout=client.config.timeout
            )
            
            if response.status == 200
                data = JSON.parse(String(response.body))
                
                content = get(data, "choices", [Dict{String,Any}()])[1]["message"]["content"]
                usage = get(data, "usage", Dict{String,Any}())
                
                # Cache successful response
                if client.config.cache_enabled
                    set_in_cache(client.cache, cache_key, Dict("content" => content, "usage" => usage))
                end
                
                client.token_count += get(usage, "total_tokens", estimated_tokens)
                client.last_request_time = now()
                
                return HelperLLMResponse(true, content, usage)
            else
                error_msg = "Helper LLM API returned status $(response.status)"
                @warn error_msg
                
                if attempt < client.config.retry_attempts
                    sleep(2^attempt)  # Exponential backoff
                else
                    return HelperLLMResponse(false, "", Dict{String,Any}(), error_msg, response.status)
                end
            end
            
        catch e
            error_msg = "Helper LLM request failed: $(e)"
            @warn error_msg
            
            if attempt < client.config.retry_attempts
                sleep(2^attempt)  # Exponential backoff
            else
                return HelperLLMResponse(false, "", Dict{String,Any}(), error_msg, 0)
            end
        end
    end
    
    return HelperLLMResponse(false, "", Dict{String,Any}(), "All retry attempts failed", 0)
end

# ============================================================================
# Entity Discovery
# ============================================================================

"""
    discover_entities(client::HelperLLMClient, text::String; entity_types::Vector{String}=String[])

Discover biomedical entities in text using helper LLM.
"""
function discover_entities(client::HelperLLMClient, text::String; entity_types::Vector{String}=String[])
    if isempty(entity_types)
        entity_types = ["DISEASE", "DRUG", "PROTEIN", "GENE", "ANATOMY", "SYMPTOM", "PROCEDURE", "ORGANISM", "CHEMICAL", "CELL_TYPE"]
    end
    
    prompt = create_entity_discovery_prompt(text, entity_types)
    messages = [Dict("role" => "user", "content" => prompt)]
    
    response = make_llm_request(client, messages)
    
    if !response.success
        @warn "Entity discovery failed: $(response.error)"
        return Vector{Dict{String,Any}}()
    end
    
    return parse_entity_discovery_response(response.content)
end

"""
    create_entity_discovery_prompt(text::String, entity_types::Vector{String})

Create a prompt for entity discovery.
"""
function create_entity_discovery_prompt(text::String, entity_types::Vector{String})
    entity_types_str = join(entity_types, ", ")
    
    return """
You are a biomedical entity extraction expert. Extract all biomedical entities from the following text and classify them into the specified types.

Entity types to extract: $entity_types_str

Text: "$text"

Please return your response as a JSON array of objects, where each object has the following structure:
{
    "text": "extracted entity text",
    "type": "entity type",
    "confidence": 0.0-1.0,
    "start_pos": character_start_position,
    "end_pos": character_end_position
}

Only extract entities that are clearly biomedical in nature. Be precise and conservative in your extractions.
"""
end

"""
    parse_entity_discovery_response(response::String)

Parse the response from entity discovery.
"""
function parse_entity_discovery_response(response::String)
    try
        # Try to extract JSON from response
        json_start = findfirst("{", response)
        json_end = findlast("}", response)
        
        if json_start === nothing || json_end === nothing
            return Vector{Dict{String,Any}}()
        end
        
        json_str = response[json_start:json_end]
        entities = JSON.parse(json_str)
        
        if isa(entities, Vector)
            return entities
        else
            return Vector{Dict{String,Any}}()
        end
        
    catch e
        @warn "Failed to parse entity discovery response: $e"
        return Vector{Dict{String,Any}}()
    end
end

# ============================================================================
# Relation Matching
# ============================================================================

"""
    match_relations(client::HelperLLMClient, entities::Vector{Dict{String,Any}}, text::String)

Match relations between entities using helper LLM.
"""
function match_relations(client::HelperLLMClient, entities::Vector{Dict{String,Any}}, text::String)
    if length(entities) < 2
        return Vector{Dict{String,Any}}()
    end
    
    prompt = create_relation_matching_prompt(entities, text)
    messages = [Dict("role" => "user", "content" => prompt)]
    
    response = make_llm_request(client, messages)
    
    if !response.success
        @warn "Relation matching failed: $(response.error)"
        return Vector{Dict{String,Any}}()
    end
    
    return parse_relation_matching_response(response.content)
end

"""
    create_relation_matching_prompt(entities::Vector{Dict{String,Any}}, text::String)

Create a prompt for relation matching.
"""
function create_relation_matching_prompt(entities::Vector{Dict{String,Any}}, text::String)
    entity_list = join(["- $(e["text"]) (Type: $(e["type"]))" for e in entities], "\n")
    
    return """
You are a biomedical relation extraction expert. Identify all meaningful relations between the extracted entities in the following text.

Entities:
$entity_list

Text: "$text"

Please return your response as a JSON array of objects, where each object has the following structure:
{
    "head_entity": "source entity text",
    "tail_entity": "target entity text",
    "relation_type": "relation type",
    "confidence": 0.0-1.0,
    "context": "supporting text context"
}

Use these relation types: TREATS, CAUSES, ASSOCIATED_WITH, PREVENTS, INHIBITS, ACTIVATES, BINDS_TO, INTERACTS_WITH, REGULATES, EXPRESSES, LOCATED_IN, PART_OF, DERIVED_FROM, SYNONYMOUS_WITH, CONTRAINDICATED_WITH, INDICATES, MANIFESTS_AS, ADMINISTERED_FOR, TARGETS, METABOLIZED_BY, TRANSPORTED_BY, SECRETED_BY, PRODUCED_BY, CONTAINS, COMPONENT_OF

Only extract relations that are clearly supported by the text. Be precise and conservative in your extractions.
"""
end

"""
    parse_relation_matching_response(response::String)

Parse the response from relation matching.
"""
function parse_relation_matching_response(response::String)
    try
        # Try to extract JSON from response
        json_start = findfirst("{", response)
        json_end = findlast("}", response)
        
        if json_start === nothing || json_end === nothing
            return Vector{Dict{String,Any}}()
        end
        
        json_str = response[json_start:json_end]
        relations = JSON.parse(json_str)
        
        if isa(relations, Vector)
            return relations
        else
            return Vector{Dict{String,Any}}()
        end
        
    catch e
        @warn "Failed to parse relation matching response: $e"
        return Vector{Dict{String,Any}}()
    end
end

# ============================================================================
# Batch Processing
# ============================================================================

"""
    discover_entities_batch(client::HelperLLMClient, texts::Vector{String}; entity_types::Vector{String}=String[])

Discover entities in multiple texts in batch.
"""
function discover_entities_batch(client::HelperLLMClient, texts::Vector{String}; entity_types::Vector{String}=String[])
    results = Vector{Vector{Dict{String,Any}}}()
    
    for text in texts
        @debug "Discovering entities in text: $(text[1:min(50, length(text))])..."
        
        entities = discover_entities(client, text; entity_types=entity_types)
        push!(results, entities)
        
        # Small delay to respect rate limits
        sleep(0.1)
    end
    
    return results
end

"""
    match_relations_batch(client::HelperLLMClient, entity_text_pairs::Vector{Tuple{Vector{Dict{String,Any}}, String})

Match relations in multiple entity-text pairs in batch.
"""
function match_relations_batch(client::HelperLLMClient, entity_text_pairs::Vector{Tuple{Vector{Dict{String,Any}}, String}})
    results = Vector{Vector{Dict{String,Any}}}()
    
    for (entities, text) in entity_text_pairs
        @debug "Matching relations for $(length(entities)) entities..."
        
        relations = match_relations(client, entities, text)
        push!(results, relations)
        
        # Small delay to respect rate limits
        sleep(0.1)
    end
    
    return results
end

# ============================================================================
# Fallback Methods
# ============================================================================

"""
    fallback_entity_discovery(text::String; entity_types::Vector{String}=String[])

Fallback entity discovery when helper LLM is unavailable.
"""
function fallback_entity_discovery(text::String; entity_types::Vector{String}=String[])
    # Simple rule-based entity discovery
    entities = Vector{Dict{String,Any}}()
    
    # Common biomedical entity patterns
    patterns = [
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|disorder|syndrome|condition|illness)\b", "DISEASE"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:drug|medication|medicine|pharmaceutical)\b", "DRUG"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:protein|enzyme|receptor|antibody|hormone)\b", "PROTEIN"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:gene|genetic|allele|mutation|variant)\b", "GENE"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:organ|tissue|muscle|bone|nerve|vessel|gland)\b", "ANATOMY"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:symptom|sign|manifestation|complaint)\b", "SYMPTOM"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:procedure|surgery|operation|treatment)\b", "PROCEDURE"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:bacteria|virus|fungus|parasite|microorganism)\b", "ORGANISM"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:chemical|compound|molecule|substance)\b", "CHEMICAL"),
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:cell|tissue|line|culture|strain)\b", "CELL_TYPE")
    ]
    
    for (pattern, entity_type) in patterns
        if isempty(entity_types) || entity_type in entity_types
            matches = eachmatch(pattern, text, overlap=false)
            for match in matches
                push!(entities, Dict(
                    "text" => match.match,
                    "type" => entity_type,
                    "confidence" => 0.5,  # Lower confidence for fallback
                    "start_pos" => match.offset,
                    "end_pos" => match.offset + length(match.match)
                ))
            end
        end
    end
    
    return entities
end

"""
    fallback_relation_matching(entities::Vector{Dict{String,Any}}, text::String)

Fallback relation matching when helper LLM is unavailable.
"""
function fallback_relation_matching(entities::Vector{Dict{String,Any}}, text::String)
    # Simple rule-based relation matching
    relations = Vector{Dict{String,Any}}()
    
    if length(entities) < 2
        return relations
    end
    
    # Look for common relation patterns
    relation_patterns = [
        (r"\b(treats?|therapy|therapeutic|medication|drug|medicine)\b", "TREATS"),
        (r"\b(causes?|leads? to|results? in|induces?|triggers?)\b", "CAUSES"),
        (r"\b(associated with|related to|linked to|correlated with)\b", "ASSOCIATED_WITH"),
        (r"\b(prevents?|protects? against|reduces? risk|avoids?)\b", "PREVENTS"),
        (r"\b(inhibits?|suppresses?|blocks?|reduces?)\b", "INHIBITS"),
        (r"\b(activates?|stimulates?|enhances?|promotes?)\b", "ACTIVATES"),
        (r"\b(binds? to|interacts? with|attaches? to|connects? to)\b", "BINDS_TO"),
        (r"\b(regulates?|modulates?|controls?|influences?)\b", "REGULATES"),
        (r"\b(expresses?|produces?|synthesizes?|generates?)\b", "EXPRESSES"),
        (r"\b(located in|found in|present in|exists in)\b", "LOCATED_IN"),
        (r"\b(part of|component of|constituent of|element of)\b", "PART_OF"),
        (r"\b(derived from|originates from|comes from|stems from)\b", "DERIVED_FROM")
    ]
    
    for (pattern, relation_type) in relation_patterns
        matches = eachmatch(pattern, text, overlap=false)
        for match in matches
            # Find entities near this relation
            for i in 1:length(entities)
                for j in (i+1):length(entities)
                    head_entity = entities[i]
                    tail_entity = entities[j]
                    
                    # Check if entities are near the relation
                    if (head_entity["start_pos"] < match.offset < tail_entity["end_pos"]) ||
                       (tail_entity["start_pos"] < match.offset < head_entity["end_pos"])
                        
                        push!(relations, Dict(
                            "head_entity" => head_entity["text"],
                            "tail_entity" => tail_entity["text"],
                            "relation_type" => relation_type,
                            "confidence" => 0.5,  # Lower confidence for fallback
                            "context" => match.match
                        ))
                    end
                end
            end
        end
    end
    
    return relations
end

# ============================================================================
# Error Handling
# ============================================================================

"""
    handle_llm_error(response::HelperLLMResponse, context::String="")

Handle helper LLM API errors with appropriate logging and fallback.
"""
function handle_llm_error(response::HelperLLMResponse, context::String="")
    if response.success
        return
    end
    
    error_msg = "Helper LLM API error"
    if !isempty(context)
        error_msg *= " in $context"
    end
    
    if !isempty(response.error)
        error_msg *= ": $(response.error)"
    end
    
    if response.status_code > 0
        error_msg *= " (status: $(response.status_code))"
    end
    
    @error error_msg
    
    # Return fallback response
    return HelperLLMResponse(false, "", Dict{String,Any}(), error_msg, response.status_code)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    is_llm_available(client::HelperLLMClient)

Check if helper LLM API is available.
"""
function is_llm_available(client::HelperLLMClient)
    test_messages = [Dict("role" => "user", "content" => "Test")]
    response = make_llm_request(client, test_messages)
    return response.success
end

"""
    get_llm_status(client::HelperLLMClient)

Get the current status of the helper LLM client.
"""
function get_llm_status(client::HelperLLMClient)
    return Dict{String,Any}(
        "api_available" => is_llm_available(client),
        "rate_limit_remaining" => client.config.rate_limit - client.token_count,
        "cache_enabled" => client.config.cache_enabled,
        "cache_size" => length(client.cache.data),
        "last_request_time" => client.last_request_time
    )
end
