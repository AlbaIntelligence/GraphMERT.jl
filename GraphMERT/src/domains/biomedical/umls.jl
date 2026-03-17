"""
UMLS (Unified Medical Language System) integration for biomedical entity linking.

This module provides a complete UMLS API client with authentication, rate limiting,
caching, and error handling for biomedical entity linking and validation.
"""

using Dates
using Logging
using HTTP
using JSON
using SQLite
using DBInterface

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
    AbstractUMLSCache

Abstract base type for UMLS caches.
"""
abstract type AbstractUMLSCache end

"""
    InMemoryUMLSCache <: AbstractUMLSCache

Local in-memory cache for UMLS responses with TTL.
"""
mutable struct InMemoryUMLSCache <: AbstractUMLSCache
    concepts::Dict{String,Tuple{Any,DateTime}}
    relations::Dict{String,Tuple{Any,DateTime}}
    max_size::Int
    ttl_seconds::Int

    function InMemoryUMLSCache(max_size::Int = 1000, ttl_seconds::Int = 3600)
        new(
            Dict{String,Tuple{Any,DateTime}}(),
            Dict{String,Tuple{Any,DateTime}}(),
            max_size,
            ttl_seconds,
        )
    end
end

"""
    SQLiteUMLSCache <: AbstractUMLSCache

SQLite-backed persistent cache for UMLS responses with TTL.
"""
struct SQLiteUMLSCache <: AbstractUMLSCache
    db::SQLite.DB
    ttl_seconds::Int

    function SQLiteUMLSCache(path::String, ttl_seconds::Int = 86400 * 7) # Default 1 week for disk cache
        db = SQLite.DB(path)
        
        # Create tables if not exist
        DBInterface.execute(db, """
            CREATE TABLE IF NOT EXISTS umls_cache (
                key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """)
        
        # Create index on category for cleanup if needed
        DBInterface.execute(db, "CREATE INDEX IF NOT EXISTS idx_category ON umls_cache(category)")
        
        new(db, ttl_seconds)
    end
end

# ============================================================================
# Cache Interface Methods
# ============================================================================

function get_cached_item(cache::InMemoryUMLSCache, category::String, key::String)
    store = category == "relations" ? cache.relations : cache.concepts
    
    if haskey(store, key)
        data, timestamp = store[key]
        if now() - timestamp < Second(cache.ttl_seconds)
            return data
        else
            delete!(store, key) # Expired
        end
    end
    return nothing
end

function set_cached_item(cache::InMemoryUMLSCache, category::String, key::String, data::Any)
    store = category == "relations" ? cache.relations : cache.concepts
    
    # Simple LRU-like eviction (random deletion if full)
    if length(store) >= cache.max_size
        # Delete a random key (not true LRU but sufficient for simple memory cache)
        pop!(store) 
    end
    
    store[key] = (data, now())
end

function get_cached_item(cache::SQLiteUMLSCache, category::String, key::String)
    # Composite key for DB lookup to avoid collisions between categories if key logic overlaps
    # But here we use (key, category) in WHERE clause
    
    # Note: SQLite.jl returns a DataFrame-like iterator
    # We select data and timestamp
    row = DBInterface.execute(cache.db, "SELECT data, timestamp FROM umls_cache WHERE key = ? AND category = ?", [key, category])
    
    for r in row
        # Parse timestamp (SQLite stores as string typically, need to check format)
        # SQLite.jl handles DateTime conversion usually if column type is detected
        # Assuming r.timestamp is compatible or string
        
        # Check TTL
        # If r.timestamp is a string, parse it. If DateTime, use it.
        ts = r.timestamp isa AbstractString ? DateTime(r.timestamp) : r.timestamp
        
        if now() - ts < Second(cache.ttl_seconds)
            return JSON.parse(r.data)
        else
            # Delete expired
            DBInterface.execute(cache.db, "DELETE FROM umls_cache WHERE key = ? AND category = ?", [key, category])
            return nothing
        end
    end
    
    return nothing
end

function set_cached_item(cache::SQLiteUMLSCache, category::String, key::String, data::Any)
    json_data = JSON.json(data)
    ts = now()
    
    # Upsert
    DBInterface.execute(cache.db, """
        INSERT INTO umls_cache (key, category, data, timestamp) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            data = excluded.data,
            timestamp = excluded.timestamp,
            category = excluded.category
    """, [key, category, json_data, ts])
end


"""
    UMLSClient

UMLS API client with rate limiting and caching.
"""
mutable struct UMLSClient
    config::UMLSConfig
    cache::AbstractUMLSCache
    last_request_time::Float64
    request_count::Int
    rate_limit_window_start::Float64
end

"""
    create_umls_client(api_key::String; kwargs...)

Create a new UMLS client with authentication and rate limiting.
If `api_key` is "mock", `mock_mode` defaults to true.

# Arguments
- `cache_path::String`: Path to SQLite cache file (optional). If provided, uses persistent cache.
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
    cache_path::Union{String, Nothing} = nothing
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
    
    local cache
    if cache_path !== nothing && !isempty(cache_path)
        # Use persistent cache (default 7 days if not specified, but here we use passed cache_ttl)
        # Maybe allow separate TTL for disk? For now use same or default to longer.
        # Let's use cache_ttl * 24 for disk to make it worth it, or just 1 week default logic
        disk_ttl = cache_ttl == 3600 ? 86400 * 7 : cache_ttl
        cache = SQLiteUMLSCache(cache_path, disk_ttl)
    else
        cache = InMemoryUMLSCache(1000, cache_ttl)
    end

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
    cached_data = get_cached_item(client.cache, "relations", cache_key)
    
    if cached_data !== nothing
        return UMLSResponse(true, cached_data, nothing, 200)
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
        set_cached_item(client.cache, "relations", cache_key, mock_data)
        return UMLSResponse(true, mock_data, nothing, 200)
    end

    # Query API
    # https://documentation.uts.nlm.nih.gov/rest/content/views/2.0/concept/CUI/relations
    response = _make_request(client, "/content/current/CUI/$cui/relations", Dict{String,String}())
    
    if response.success
        set_cached_item(client.cache, "relations", cache_key, response.data)
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
    cached_data = get_cached_item(client.cache, "concepts", cache_key)
    
    if cached_data !== nothing && isa(cached_data, Dict) && haskey(cached_data, "semantic_types")
        return cached_data["semantic_types"]
    end

    if client.config.mock_mode
        # Deterministic mock types
        mock_types = ["T047"] # Disease or Syndrome
        set_cached_item(client.cache, "concepts", cache_key, Dict("semantic_types" => mock_types))
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
    set_cached_item(client.cache, "concepts", cache_key, Dict("semantic_types" => semantic_types))
    
    return semantic_types
end
