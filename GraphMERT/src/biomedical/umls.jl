"""
UMLS integration for GraphMERT.jl

This module implements UMLS (Unified Medical Language System) integration
for biomedical entity linking and validation as specified in the GraphMERT paper.
"""

using HTTP
using JSON
using Logging
using Random

# ============================================================================
# UMLS Configuration
# ============================================================================

"""
    UMLSConfig

Configuration for UMLS API integration.
"""
struct UMLSConfig
    api_key::String
    base_url::String
    rate_limit::Int
    timeout::Int
    retry_attempts::Int
    cache_enabled::Bool
    cache_ttl::Int

    function UMLSConfig(;
        api_key::String="",
        base_url::String="https://uts-ws.nlm.nih.gov/rest",
        rate_limit::Int=100,  # requests per minute
        timeout::Int=30,
        retry_attempts::Int=3,
        cache_enabled::Bool=true,
        cache_ttl::Int=3600  # 1 hour
    )
        @assert !isempty(api_key) "UMLS API key is required"
        @assert rate_limit > 0 "Rate limit must be positive"
        @assert timeout > 0 "Timeout must be positive"
        @assert retry_attempts > 0 "Retry attempts must be positive"
        @assert cache_ttl > 0 "Cache TTL must be positive"

        new(api_key, base_url, rate_limit, timeout, retry_attempts, cache_enabled, cache_ttl)
    end
end

"""
    UMLSResponse

Response from UMLS API.
"""
struct UMLSResponse
    success::Bool
    data::Dict{String,Any}
    error::String
    status_code::Int
    timestamp::DateTime

    function UMLSResponse(success::Bool, data::Dict{String,Any}, error::String="", status_code::Int=200)
        new(success, data, error, status_code, now())
    end
end

"""
    UMLSCache

Simple in-memory cache for UMLS responses.
"""
mutable struct UMLSCache
    data::Dict{String,Any}
    timestamps::Dict{String,DateTime}
    ttl::Int

    function UMLSCache(ttl::Int=3600)
        new(Dict{String,Any}(), Dict{String,DateTime}(), ttl)
    end
end

# ============================================================================
# UMLS Client
# ============================================================================

"""
    UMLSClient

Client for UMLS API integration with rate limiting and caching.
"""
mutable struct UMLSClient
    config::UMLSConfig
    cache::UMLSCache
    last_request_time::DateTime
    request_count::Int
    rate_limit_window::DateTime

    function UMLSClient(config::UMLSConfig)
        cache = UMLSCache(config.cache_ttl)
        new(config, cache, now(), 0, now())
    end
end

"""
    create_umls_client(api_key::String; kwargs...)

Create a new UMLS client with the given API key.
"""
function create_umls_client(api_key::String; kwargs...)
    config = UMLSConfig(; api_key=api_key, kwargs...)
    return UMLSClient(config)
end

# ============================================================================
# Rate Limiting
# ============================================================================

"""
    check_rate_limit(client::UMLSClient)

Check if we can make a request without exceeding rate limits.
"""
function check_rate_limit(client::UMLSClient)
    current_time = now()
    
    # Reset counter if we're in a new minute
    if current_time - client.rate_limit_window >= Minute(1)
        client.request_count = 0
        client.rate_limit_window = current_time
    end
    
    return client.request_count < client.config.rate_limit
end

"""
    wait_for_rate_limit(client::UMLSClient)

Wait until we can make a request without exceeding rate limits.
"""
function wait_for_rate_limit(client::UMLSClient)
    while !check_rate_limit(client)
        sleep(1)  # Wait 1 second
    end
end

# ============================================================================
# Caching
# ============================================================================

"""
    get_from_cache(cache::UMLSCache, key::String)

Get a value from the cache if it exists and hasn't expired.
"""
function get_from_cache(cache::UMLSCache, key::String)
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
    set_in_cache(cache::UMLSCache, key::String, value::Any)

Set a value in the cache with current timestamp.
"""
function set_in_cache(cache::UMLSCache, key::String, value::Any)
    cache.data[key] = value
    cache.timestamps[key] = now()
end

# ============================================================================
# HTTP Requests
# ============================================================================

"""
    make_umls_request(client::UMLSClient, endpoint::String, params::Dict{String,Any})

Make a request to the UMLS API with rate limiting and error handling.
"""
function make_umls_request(client::UMLSClient, endpoint::String, params::Dict{String,Any})
    # Check rate limit
    wait_for_rate_limit(client)
    
    # Build URL
    url = "$(client.config.base_url)/$endpoint"
    
    # Add API key to parameters
    params["apiKey"] = client.config.api_key
    
    # Check cache first
    cache_key = "$endpoint:$(hash(params))"
    if client.config.cache_enabled
        cached_result = get_from_cache(client.cache, cache_key)
        if cached_result !== nothing
            @debug "UMLS cache hit for $endpoint"
            return UMLSResponse(true, cached_result)
        end
    end
    
    # Make request with retries
    for attempt in 1:client.config.retry_attempts
        try
            @debug "Making UMLS request to $endpoint (attempt $attempt)"
            
            response = HTTP.get(url; query=params, timeout=client.config.timeout)
            
            if response.status == 200
                data = JSON.parse(String(response.body))
                
                # Cache successful response
                if client.config.cache_enabled
                    set_in_cache(client.cache, cache_key, data)
                end
                
                client.request_count += 1
                client.last_request_time = now()
                
                return UMLSResponse(true, data)
            else
                error_msg = "UMLS API returned status $(response.status)"
                @warn error_msg
                
                if attempt < client.config.retry_attempts
                    sleep(2^attempt)  # Exponential backoff
                else
                    return UMLSResponse(false, Dict{String,Any}(), error_msg, response.status)
                end
            end
            
        catch e
            error_msg = "UMLS request failed: $(e)"
            @warn error_msg
            
            if attempt < client.config.retry_attempts
                sleep(2^attempt)  # Exponential backoff
            else
                return UMLSResponse(false, Dict{String,Any}(), error_msg, 0)
            end
        end
    end
    
    return UMLSResponse(false, Dict{String,Any}(), "All retry attempts failed", 0)
end

# ============================================================================
# UMLS API Methods
# ============================================================================

"""
    search_concepts(client::UMLSClient, query::String; kwargs...)

Search for concepts in UMLS.
"""
function search_concepts(client::UMLSClient, query::String;
    search_type::String="exact",
    sabs::String="",
    page_size::Int=25,
    page_number::Int=1)
    
    params = Dict{String,Any}(
        "string" => query,
        "searchType" => search_type,
        "pageSize" => page_size,
        "pageNumber" => page_number
    )
    
    if !isempty(sabs)
        params["sabs"] = sabs
    end
    
    return make_umls_request(client, "search/current", params)
end

"""
    get_concept_details(client::UMLSClient, cui::String)

Get detailed information about a concept by CUI.
"""
function get_concept_details(client::UMLSClient, cui::String)
    params = Dict{String,Any}("cui" => cui)
    return make_umls_request(client, "content/current/CUI/$cui", params)
end

"""
    get_atoms(client::UMLSClient, cui::String; sabs::String="")

Get atoms (terms) for a concept by CUI.
"""
function get_atoms(client::UMLSClient, cui::String; sabs::String="")
    params = Dict{String,Any}("cui" => cui)
    
    if !isempty(sabs)
        params["sabs"] = sabs
    end
    
    return make_umls_request(client, "content/current/CUI/$cui/atoms", params)
end

"""
    get_relations(client::UMLSClient, cui::String; relation_type::String="")

Get relations for a concept by CUI.
"""
function get_relations(client::UMLSClient, cui::String; relation_type::String="")
    params = Dict{String,Any}("cui" => cui)
    
    if !isempty(relation_type)
        params["relationType"] = relation_type
    end
    
    return make_umls_request(client, "content/current/CUI/$cui/relations", params)
end

# ============================================================================
# Entity Linking
# ============================================================================

"""
    link_entity(client::UMLSClient, entity_text::String; threshold::Float64=0.8)

Link a biomedical entity to UMLS concepts.
"""
function link_entity(client::UMLSClient, entity_text::String; threshold::Float64=0.8)
    @assert 0.0 <= threshold <= 1.0 "Threshold must be between 0.0 and 1.0"
    
    # Search for exact matches first
    response = search_concepts(client, entity_text; search_type="exact")
    
    if !response.success
        @warn "UMLS search failed for entity: $entity_text"
        return nothing
    end
    
    results = get(response.data, "result", Dict{String,Any}())
    concepts = get(results, "results", Vector{Any}())
    
    if isempty(concepts)
        # Try approximate search
        response = search_concepts(client, entity_text; search_type="approximate")
        
        if !response.success
            return nothing
        end
        
        results = get(response.data, "result", Dict{String,Any}())
        concepts = get(results, "results", Vector{Any}())
    end
    
    if isempty(concepts)
        return nothing
    end
    
    # Find best match above threshold
    best_match = nothing
    best_score = 0.0
    
    for concept in concepts
        score = get(concept, "score", 0.0)
        if score >= threshold && score > best_score
            best_match = concept
            best_score = score
        end
    end
    
    return best_match
end

"""
    get_entity_cui(client::UMLSClient, entity_text::String; threshold::Float64=0.8)

Get the CUI (Concept Unique Identifier) for an entity.
"""
function get_entity_cui(client::UMLSClient, entity_text::String; threshold::Float64=0.8)
    match = link_entity(client, entity_text; threshold=threshold)
    
    if match === nothing
        return nothing
    end
    
    return get(match, "ui", nothing)
end

"""
    get_entity_semantic_types(client::UMLSClient, cui::String)

Get semantic types for a concept by CUI.
"""
function get_entity_semantic_types(client::UMLSClient, cui::String)
    response = get_concept_details(client, cui)
    
    if !response.success
        return String[]
    end
    
    result = get(response.data, "result", Dict{String,Any}())
    semantic_types = get(result, "semanticTypes", Vector{Any}())
    
    return [get(st, "name", "") for st in semantic_types]
end

# ============================================================================
# Batch Processing
# ============================================================================

"""
    link_entities_batch(client::UMLSClient, entities::Vector{String}; threshold::Float64=0.8)

Link multiple entities to UMLS concepts in batch.
"""
function link_entities_batch(client::UMLSClient, entities::Vector{String}; threshold::Float64=0.8)
    results = Dict{String,Any}()
    
    for entity in entities
        @debug "Linking entity: $entity"
        
        match = link_entity(client, entity; threshold=threshold)
        if match !== nothing
            results[entity] = match
        else
            results[entity] = nothing
        end
        
        # Small delay to respect rate limits
        sleep(0.1)
    end
    
    return results
end

# ============================================================================
# Fallback Entity Recognition
# ============================================================================

"""
    fallback_entity_recognition(entity_text::String)

Fallback entity recognition when UMLS is unavailable.
"""
function fallback_entity_recognition(entity_text::String)
    # Simple rule-based entity recognition
    entity_text_lower = lowercase(entity_text)
    
    # Common biomedical entity patterns
    if occursin(r"\b(disease|disorder|syndrome|condition)\b", entity_text_lower)
        return "DISEASE"
    elseif occursin(r"\b(drug|medication|medicine|pharmaceutical)\b", entity_text_lower)
        return "DRUG"
    elseif occursin(r"\b(protein|gene|enzyme|receptor)\b", entity_text_lower)
        return "PROTEIN"
    elseif occursin(r"\b(organ|tissue|cell|anatomy)\b", entity_text_lower)
        return "ANATOMY"
    elseif occursin(r"\b(symptom|sign|manifestation)\b", entity_text_lower)
        return "SYMPTOM"
    else
        return "UNKNOWN"
    end
end

# ============================================================================
# Error Handling
# ============================================================================

"""
    handle_umls_error(response::UMLSResponse, context::String="")

Handle UMLS API errors with appropriate logging and fallback.
"""
function handle_umls_error(response::UMLSResponse, context::String="")
    if response.success
        return
    end
    
    error_msg = "UMLS API error"
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
    return UMLSResponse(false, Dict{String,Any}(), error_msg, response.status_code)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    is_umls_available(client::UMLSClient)

Check if UMLS API is available.
"""
function is_umls_available(client::UMLSClient)
    response = search_concepts(client, "test"; page_size=1)
    return response.success
end

"""
    get_umls_status(client::UMLSClient)

Get the current status of the UMLS client.
"""
function get_umls_status(client::UMLSClient)
    return Dict{String,Any}(
        "api_available" => is_umls_available(client),
        "rate_limit_remaining" => client.config.rate_limit - client.request_count,
        "cache_enabled" => client.config.cache_enabled,
        "cache_size" => length(client.cache.data),
        "last_request_time" => client.last_request_time
    )
end
