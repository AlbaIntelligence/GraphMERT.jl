"""
UMLS (Unified Medical Language System) integration for biomedical entity linking.

This module provides a complete UMLS API client with authentication, rate limiting,
caching, and error handling for biomedical entity linking and validation.
"""

using Dates
using Logging
using HTTP
using JSON

# EntityLinkingResult is defined in the main GraphMERT module

"""
    UMLSConfig

Configuration for UMLS API client.
"""
struct UMLSConfig
    api_key::String
    base_url::String
    timeout::Int
    max_retries::Int
    semantic_networks::Vector{String}
    rate_limit::Int  # requests per minute
    mock_mode::Bool
end

"""
    UMLSResponse

Response from UMLS API.
"""
struct UMLSResponse
    success::Bool
    data::Dict{String,Any}
    error::Union{String,Nothing}
    http_status::Union{Int,Nothing}
end

"""
    UMLSCache

Local cache for UMLS responses with TTL.
"""
mutable struct UMLSCache
    concepts::Dict{String,Tuple{Any,DateTime}}
    relations::Dict{String,Tuple{Any,DateTime}}
    max_size::Int
    ttl_seconds::Int

    function UMLSCache(max_size::Int = 1000, ttl_seconds::Int = 3600)
        new(
            Dict{String,Tuple{Any,DateTime}}(),
            Dict{String,Tuple{Any,DateTime}}(),
            max_size,
            ttl_seconds,
        )
    end
end

"""
    UMLSClient

UMLS API client with rate limiting and caching.
"""
mutable struct UMLSClient
    config::UMLSConfig
    cache::UMLSCache
    last_request_time::Float64
    request_count::Int
    rate_limit_window_start::Float64
end

"""
    create_umls_client(api_key::String; kwargs...)

Create a new UMLS client with authentication and rate limiting.
If `api_key` is "mock", `mock_mode` defaults to true.
"""
function create_umls_client(
    api_key::String;
    base_url::String = "https://uts-ws.nlm.nih.gov/rest",
    timeout::Int = 30,
    max_retries::Int = 3,
    semantic_networks::Vector{String} = ["SNOMEDCT_US", "MSH", "RXNORM"],
    rate_limit::Int = 100,
    cache_ttl::Int = 3600,
    mock_mode::Bool = (api_key == "mock"),
)
    config = UMLSConfig(
        api_key,
        base_url,
        timeout,
        max_retries,
        semantic_networks,
        rate_limit,
        mock_mode,
    )
    cache = UMLSCache(1000, cache_ttl)
    return UMLSClient(config, cache, 0.0, 0, time())
end

"""
    _make_request(client::UMLSClient, endpoint::String, params::Dict{String,String})

Make a request to the UMLS API with rate limiting and retries.
"""
function _make_request(client::UMLSClient, endpoint::String, params::Dict{String,String})
    # 1. Check mock mode
    if client.config.mock_mode
        return UMLSResponse(true, Dict{String,Any}(), nothing, 200)
    end

    # 2. Rate limiting
    current_time = time()
    if current_time - client.rate_limit_window_start > 60.0
        client.rate_limit_window_start = current_time
        client.request_count = 0
    end

    if client.request_count >= client.config.rate_limit
        sleep_time = 60.0 - (current_time - client.rate_limit_window_start)
        if sleep_time > 0
            @debug "Rate limit reached, sleeping for $sleep_time seconds"
            sleep(sleep_time)
        end
        client.rate_limit_window_start = time()
        client.request_count = 0
    end

    client.request_count += 1
    client.last_request_time = time()

    # 3. Prepare request
    url = "$(client.config.base_url)$endpoint"
    query_params = copy(params)
    query_params["apiKey"] = client.config.api_key
    query_params["version"] = "current"

    # 4. Execute with retries
    retries = 0
    while retries <= client.config.max_retries
        try
            response = HTTP.get(
                url,
                query=query_params,
                readtimeout=client.config.timeout,
                connecttimeout=client.config.timeout
            )
            
            if response.status == 200
                data = JSON.parse(String(response.body))
                return UMLSResponse(true, data, nothing, 200)
            else
                return UMLSResponse(false, Dict{String,Any}(), "HTTP $(response.status)", response.status)
            end
        catch e
            retries += 1
            if retries > client.config.max_retries
                return UMLSResponse(false, Dict{String,Any}(), string(e), nothing)
            end
            sleep(2^retries) # Exponential backoff
        end
    end
    
    return UMLSResponse(false, Dict{String,Any}(), "Max retries exceeded", nothing)
end

"""
    link_entity_to_umls(entity_text::String, client::UMLSClient)

Link a biomedical entity to UMLS concepts.
"""
function link_entity_to_umls(entity_text::String, client::UMLSClient)
    # 1. Mock mode
    if client.config.mock_mode
        # Deterministic mock response based on text
        if isempty(entity_text)
            return nothing
        end
        
        # Simple mock logic
        cui = "C" * lpad(string(hash(entity_text) % 1000000), 7, '0')
        return EntityLinkingResult(
            entity_text,
            cui,
            uppercasefirst(entity_text), # Preferred name
            ["T047"], # Disease or Syndrome
            1.0,
            "UMLS-MOCK"
        )
    end

    # 2. Check cache (TODO: implement caching for linking results specifically if needed, 
    # currently we cache concepts/relations)
    
    # 3. Search API
    # https://documentation.uts.nlm.nih.gov/rest/search/index.html
    params = Dict(
        "string" => entity_text,
        "searchType" => "approximate"
    )
    
    # Add semantic network filter if configured
    # (Note: API parameter might be 'sabs' for source vocabularies, not semantic types directly in search)
    
    response = _make_request(client, "/search/current", params)
    
    if !response.success || !haskey(response.data, "result") || isempty(response.data["result"]["results"])
        return nothing
    end
    
    # 4. Process best match
    results = response.data["result"]["results"]
    best_match = results[1]
    
    if best_match["ui"] == "NONE"
        return nothing
    end
    
    cui = best_match["ui"]
    name = best_match["name"]
    
    # Get semantic types to filter
    semantic_types = get_entity_semantic_types(client, cui)
    
    return EntityLinkingResult(
        entity_text,
        cui,
        name,
        semantic_types,
        1.0, # Placeholder score, search results don't always give score
        "UMLS"
    )
end

"""
    get_entity_cui(client::UMLSClient, entity_text::String)

Get the CUI for an entity.
"""
function get_entity_cui(client::UMLSClient, entity_text::String)
    linking_result = link_entity_to_umls(entity_text, client)
    return linking_result !== nothing ? linking_result.cui : nothing
end

"""
    link_entities_batch(client::UMLSClient, entities::Vector{String})

Link multiple entities to UMLS concepts.
"""
function link_entities_batch(client::UMLSClient, entities::Vector{String})
    results = Dict{String,EntityLinkingResult}()

    for entity in entities
        linking_result = link_entity_to_umls(entity, client)
        if linking_result !== nothing
            results[entity] = linking_result
        end
    end

    return results
end

"""
    get_relations(client::UMLSClient, cui::String)

Get relations for a UMLS CUI.
"""
function get_relations(client::UMLSClient, cui::String)
    # Check cache first
    cache_key = "relations:$cui"
    if haskey(client.cache.relations, cache_key)
        cached_data, cached_time = client.cache.relations[cache_key]
        if now() - cached_time < Second(client.cache.ttl_seconds)
            return UMLSResponse(true, cached_data, nothing, 200)
        end
    end

    if client.config.mock_mode
        # Return dummy relations
        mock_data = Dict{String,Any}(
            "result" => [
                Dict("ui" => "R1", "relatedId" => "C12345", "relationLabel" => "TREATS"),
                Dict("ui" => "R2", "relatedId" => "C67890", "relationLabel" => "CAUSES")
            ]
        )
         # Cache the mock response
        if length(client.cache.relations) < client.cache.max_size
            client.cache.relations[cache_key] = (mock_data, now())
        end
        return UMLSResponse(true, mock_data, nothing, 200)
    end

    # Query API
    # https://documentation.uts.nlm.nih.gov/rest/content/views/2.0/concept/CUI/relations
    response = _make_request(client, "/content/current/CUI/$cui/relations", Dict{String,String}())
    
    if response.success
        if length(client.cache.relations) < client.cache.max_size
            client.cache.relations[cache_key] = (response.data, now())
        end
    end
    
    return response
end

"""
    UMLSTriple

Represents a triple retrieved from UMLS.
"""
struct UMLSTriple
    cui::String
    relation_label::String
    related_cui::String
    related_name::String
    source::String
end

"""
    retrieve_triples(client::UMLSClient, cui::String, allowed_relations::Vector{String}=String[])

Retrieve triples from UMLS for a given CUI, optionally filtering by relation types.
Returns a list of `UMLSTriple`s.
"""
function retrieve_triples(client::UMLSClient, cui::String, allowed_relations::Vector{String}=String[])
    response = get_relations(client, cui)
    triples = UMLSTriple[]
    
    if !response.success || !haskey(response.data, "result")
        return triples
    end
    
    # Check if result is array or dict with "results"
    # Based on mock: "result" => [ ... ]
    # Based on API: "result" might be array directly if using /relations endpoint?
    # Let's handle both.
    
    results = response.data["result"]
    if isa(results, Dict) && haskey(results, "results")
        results = results["results"]
    elseif !isa(results, Vector)
        # Handle unexpected format gracefully
        if isa(results, Dict)
             # Maybe single object?
             results = [results]
        else
             return triples
        end
    end
    
    allowed_set = Set(allowed_relations)
    
    for item in results
        # Item structure: { "ui": "R...", "relatedId": "C...", "relationLabel": "..." }
        # Note: API field names might vary slightly, need to check docs.
        # usually: "relatedId" (url), "relatedIdName" (name), "relationLabel" (REL)
        
        rel_label = get(item, "relationLabel", "")
        if isempty(rel_label)
            rel_label = get(item, "additionalRelationLabel", "") # legacy
        end
        
        if isempty(rel_label)
            continue
        end
        
        if !isempty(allowed_set) && !(rel_label in allowed_set)
            continue
        end
        
        related_cui = ""
        related_id_url = get(item, "relatedId", "")
        # Extract CUI from URL: .../C0012345
        m = match(r"C[0-9]+$", related_id_url)
        if m !== nothing
            related_cui = m.match
        else
            # Try to find it in other fields if needed
            continue 
        end
        
        related_name = get(item, "relatedIdName", "")
        source = get(item, "rootSource", "UMLS")
        
        push!(triples, UMLSTriple(cui, rel_label, related_cui, related_name, source))
    end
    
    return triples
end

"""
    get_entity_semantic_types(client::UMLSClient, cui::String)

Get semantic types for a UMLS CUI.
"""
function get_entity_semantic_types(client::UMLSClient, cui::String)
    # Check cache first
    cache_key = "semantic_types:$cui"
    if haskey(client.cache.concepts, cache_key)
        cached_data, cached_time = client.cache.concepts[cache_key]
        if now() - cached_time < Dates.Second(client.cache.ttl_seconds)
            if isa(cached_data, Dict) && haskey(cached_data, "semantic_types")
                return cached_data["semantic_types"]
            end
        end
    end

    if client.config.mock_mode
        # Deterministic mock types
        mock_types = ["T047"] # Disease or Syndrome
        if length(client.cache.concepts) < client.cache.max_size
            client.cache.concepts[cache_key] = (Dict("semantic_types" => mock_types), now())
        end
        return mock_types
    end

    # Query API
    # https://documentation.uts.nlm.nih.gov/rest/content/views/2.0/concept/CUI
    response = _make_request(client, "/content/current/CUI/$cui", Dict{String,String}())
    
    semantic_types = String[]
    
    if response.success && haskey(response.data, "result") && haskey(response.data["result"], "semanticTypes")
        for st in response.data["result"]["semanticTypes"]
             # Extract TUI (e.g., "T047") or name
             # The API returns objects with 'name' and 'uri'. 
             # We'll use 'name' or try to parse TUI from URI if available.
             # For now, let's use 'name'.
             if haskey(st, "name")
                 push!(semantic_types, st["name"])
             end
        end
    end
    
    # Cache the result
    if length(client.cache.concepts) < client.cache.max_size
        client.cache.concepts[cache_key] = (Dict("semantic_types" => semantic_types), now())
    end
    
    return semantic_types
end
