"""
PubMed text processing for GraphMERT.jl

This module implements PubMed text processing for biomedical document parsing
as specified in the GraphMERT paper for biomedical knowledge graph construction.
"""

using HTTP
using JSON
using Logging
using Random
using Dates

# ============================================================================
# PubMed Configuration
# ============================================================================

"""
    PubMedConfig

Configuration for PubMed API integration.
"""
struct PubMedConfig
    base_url::String
    rate_limit::Int
    timeout::Int
    retry_attempts::Int
    cache_enabled::Bool
    cache_ttl::Int
    email::String

    function PubMedConfig(;
        base_url::String="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        rate_limit::Int=3,  # requests per second
        timeout::Int=30,
        retry_attempts::Int=3,
        cache_enabled::Bool=true,
        cache_ttl::Int=3600,  # 1 hour
        email::String=""
    )
        @assert rate_limit > 0 "Rate limit must be positive"
        @assert timeout > 0 "Timeout must be positive"
        @assert retry_attempts > 0 "Retry attempts must be positive"
        @assert cache_ttl > 0 "Cache TTL must be positive"

        new(base_url, rate_limit, timeout, retry_attempts, cache_enabled, cache_ttl, email)
    end
end

"""
    PubMedResponse

Response from PubMed API.
"""
struct PubMedResponse
    success::Bool
    data::Dict{String,Any}
    error::String
    status_code::Int
    timestamp::DateTime

    function PubMedResponse(success::Bool, data::Dict{String,Any}, error::String="", status_code::Int=200)
        new(success, data, error, status_code, now())
    end
end

"""
    PubMedCache

Simple in-memory cache for PubMed responses.
"""
mutable struct PubMedCache
    data::Dict{String,Any}
    timestamps::Dict{String,DateTime}
    ttl::Int

    function PubMedCache(ttl::Int=3600)
        new(Dict{String,Any}(), Dict{String,DateTime}(), ttl)
    end
end

# ============================================================================
# PubMed Client
# ============================================================================

"""
    PubMedClient

Client for PubMed API integration with rate limiting and caching.
"""
mutable struct PubMedClient
    config::PubMedConfig
    cache::PubMedCache
    last_request_time::DateTime
    request_count::Int
    rate_limit_window::DateTime

    function PubMedClient(config::PubMedConfig)
        cache = PubMedCache(config.cache_ttl)
        new(config, cache, now(), 0, now())
    end
end

"""
    create_pubmed_client(; kwargs...)

Create a new PubMed client.
"""
function create_pubmed_client(; kwargs...)
    config = PubMedConfig(; kwargs...)
    return PubMedClient(config)
end

# ============================================================================
# Rate Limiting
# ============================================================================

"""
    check_rate_limit(client::PubMedClient)

Check if we can make a request without exceeding rate limits.
"""
function check_rate_limit(client::PubMedClient)
    current_time = now()
    
    # Reset counter if we're in a new second
    if current_time - client.rate_limit_window >= Second(1)
        client.request_count = 0
        client.rate_limit_window = current_time
    end
    
    return client.request_count < client.config.rate_limit
end

"""
    wait_for_rate_limit(client::PubMedClient)

Wait until we can make a request without exceeding rate limits.
"""
function wait_for_rate_limit(client::PubMedClient)
    while !check_rate_limit(client)
        sleep(0.1)  # Wait 100ms
    end
end

# ============================================================================
# Caching
# ============================================================================

"""
    get_from_cache(cache::PubMedCache, key::String)

Get a value from the cache if it exists and hasn't expired.
"""
function get_from_cache(cache::PubMedCache, key::String)
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
    set_in_cache(cache::PubMedCache, key::String, value::Any)

Set a value in the cache with current timestamp.
"""
function set_in_cache(cache::PubMedCache, key::String, value::Any)
    cache.data[key] = value
    cache.timestamps[key] = now()
end

# ============================================================================
# HTTP Requests
# ============================================================================

"""
    make_pubmed_request(client::PubMedClient, endpoint::String, params::Dict{String,Any})

Make a request to the PubMed API with rate limiting and error handling.
"""
function make_pubmed_request(client::PubMedClient, endpoint::String, params::Dict{String,Any})
    # Check rate limit
    wait_for_rate_limit(client)
    
    # Build URL
    url = "$(client.config.base_url)/$endpoint"
    
    # Add email to parameters if provided
    if !isempty(client.config.email)
        params["email"] = client.config.email
    end
    
    # Check cache first
    cache_key = "$endpoint:$(hash(params))"
    if client.config.cache_enabled
        cached_result = get_from_cache(client.cache, cache_key)
        if cached_result !== nothing
            @debug "PubMed cache hit for $endpoint"
            return PubMedResponse(true, cached_result)
        end
    end
    
    # Make request with retries
    for attempt in 1:client.config.retry_attempts
        try
            @debug "Making PubMed request to $endpoint (attempt $attempt)"
            
            response = HTTP.get(url; query=params, timeout=client.config.timeout)
            
            if response.status == 200
                data = JSON.parse(String(response.body))
                
                # Cache successful response
                if client.config.cache_enabled
                    set_in_cache(client.cache, cache_key, data)
                end
                
                client.request_count += 1
                client.last_request_time = now()
                
                return PubMedResponse(true, data)
            else
                error_msg = "PubMed API returned status $(response.status)"
                @warn error_msg
                
                if attempt < client.config.retry_attempts
                    sleep(2^attempt)  # Exponential backoff
                else
                    return PubMedResponse(false, Dict{String,Any}(), error_msg, response.status)
                end
            end
            
        catch e
            error_msg = "PubMed request failed: $(e)"
            @warn error_msg
            
            if attempt < client.config.retry_attempts
                sleep(2^attempt)  # Exponential backoff
            else
                return PubMedResponse(false, Dict{String,Any}(), error_msg, 0)
            end
        end
    end
    
    return PubMedResponse(false, Dict{String,Any}(), "All retry attempts failed", 0)
end

# ============================================================================
# PubMed API Methods
# ============================================================================

"""
    search_pubmed(client::PubMedClient, query::String; kwargs...)

Search for articles in PubMed.
"""
function search_pubmed(client::PubMedClient, query::String;
    max_results::Int=100,
    retstart::Int=0,
    retmax::Int=20,
    sort::String="relevance",
    mindate::String="",
    maxdate::String="")
    
    params = Dict{String,Any}(
        "db" => "pubmed",
        "term" => query,
        "retstart" => retstart,
        "retmax" => min(retmax, max_results),
        "sort" => sort,
        "retmode" => "json"
    )
    
    if !isempty(mindate)
        params["mindate"] = mindate
    end
    
    if !isempty(maxdate)
        params["maxdate"] = maxdate
    end
    
    return make_pubmed_request(client, "esearch.fcgi", params)
end

"""
    fetch_pubmed_articles(client::PubMedClient, pmids::Vector{String})

Fetch full article details from PubMed.
"""
function fetch_pubmed_articles(client::PubMedClient, pmids::Vector{String})
    if isempty(pmids)
        return PubMedResponse(true, Dict{String,Any}())
    end
    
    params = Dict{String,Any}(
        "db" => "pubmed",
        "id" => join(pmids, ","),
        "retmode" => "json",
        "rettype" => "abstract"
    )
    
    return make_pubmed_request(client, "efetch.fcgi", params)
end

"""
    get_pubmed_article(client::PubMedClient, pmid::String)

Get a single article from PubMed.
"""
function get_pubmed_article(client::PubMedClient, pmid::String)
    return fetch_pubmed_articles(client, [pmid])
end

# ============================================================================
# Text Processing
# ============================================================================

"""
    process_pubmed_article(article_data::Dict{String,Any})

Process a PubMed article and extract relevant text.
"""
function process_pubmed_article(article_data::Dict{String,Any})
    # Extract article information
    article_info = Dict{String,Any}()
    
    # Get PMID
    if haskey(article_data, "uid")
        article_info["pmid"] = string(article_data["uid"])
    end
    
    # Get title
    if haskey(article_data, "title")
        article_info["title"] = article_data["title"]
    end
    
    # Get abstract
    if haskey(article_data, "abstract")
        article_info["abstract"] = article_data["abstract"]
    end
    
    # Get authors
    if haskey(article_data, "authors")
        article_info["authors"] = article_data["authors"]
    end
    
    # Get journal
    if haskey(article_data, "journal")
        article_info["journal"] = article_data["journal"]
    end
    
    # Get publication date
    if haskey(article_data, "pubdate")
        article_info["pubdate"] = article_data["pubdate"]
    end
    
    # Get MeSH terms
    if haskey(article_data, "mesh_terms")
        article_info["mesh_terms"] = article_data["mesh_terms"]
    end
    
    # Get keywords
    if haskey(article_data, "keywords")
        article_info["keywords"] = article_data["keywords"]
    end
    
    # Combine title and abstract for processing
    full_text = ""
    if haskey(article_info, "title")
        full_text *= article_info["title"] * " "
    end
    if haskey(article_info, "abstract")
        full_text *= article_info["abstract"]
    end
    
    article_info["full_text"] = strip(full_text)
    
    return article_info
end

"""
    extract_biomedical_text(article_info::Dict{String,Any})

Extract biomedical text from article information.
"""
function extract_biomedical_text(article_info::Dict{String,Any})
    text_parts = String[]
    
    # Add title if available
    if haskey(article_info, "title") && !isempty(article_info["title"])
        push!(text_parts, article_info["title"])
    end
    
    # Add abstract if available
    if haskey(article_info, "abstract") && !isempty(article_info["abstract"])
        push!(text_parts, article_info["abstract"])
    end
    
    # Add MeSH terms if available
    if haskey(article_info, "mesh_terms") && !isempty(article_info["mesh_terms"])
        mesh_text = join(article_info["mesh_terms"], ", ")
        push!(text_parts, "MeSH Terms: $mesh_text")
    end
    
    # Add keywords if available
    if haskey(article_info, "keywords") && !isempty(article_info["keywords"])
        keywords_text = join(article_info["keywords"], ", ")
        push!(text_parts, "Keywords: $keywords_text")
    end
    
    return join(text_parts, " ")
end

# ============================================================================
# Batch Processing
# ============================================================================

"""
    search_and_process_pubmed(client::PubMedClient, query::String; max_results::Int=100)

Search PubMed and process all results.
"""
function search_and_process_pubmed(client::PubMedClient, query::String; max_results::Int=100)
    # Search for articles
    search_response = search_pubmed(client, query; max_results=max_results)
    
    if !search_response.success
        @warn "PubMed search failed: $(search_response.error)"
        return Vector{Dict{String,Any}}()
    end
    
    # Get PMIDs
    search_data = search_response.data
    pmids = get(search_data, "esearchresult", Dict{String,Any}())
    id_list = get(pmids, "idlist", String[])
    
    if isempty(id_list)
        return Vector{Dict{String,Any}}()
    end
    
    # Fetch article details
    fetch_response = fetch_pubmed_articles(client, id_list)
    
    if !fetch_response.success
        @warn "PubMed fetch failed: $(fetch_response.error)"
        return Vector{Dict{String,Any}}()
    end
    
    # Process articles
    articles = Vector{Dict{String,Any}}()
    fetch_data = fetch_response.data
    
    if haskey(fetch_data, "result")
        result_data = fetch_data["result"]
        
        for (pmid, article_data) in result_data
            if isa(article_data, Dict)
                processed_article = process_pubmed_article(article_data)
                push!(articles, processed_article)
            end
        end
    end
    
    return articles
end

"""
    process_pubmed_batch(client::PubMedClient, queries::Vector{String}; max_results_per_query::Int=50)

Process multiple PubMed queries in batch.
"""
function process_pubmed_batch(client::PubMedClient, queries::Vector{String}; max_results_per_query::Int=50)
    all_articles = Vector{Dict{String,Any}}()
    
    for query in queries
        @debug "Processing PubMed query: $query"
        
        articles = search_and_process_pubmed(client, query; max_results=max_results_per_query)
        append!(all_articles, articles)
        
        # Small delay to respect rate limits
        sleep(0.5)
    end
    
    return all_articles
end

# ============================================================================
# Text Preprocessing
# ============================================================================

"""
    preprocess_pubmed_text(text::String)

Preprocess PubMed text for biomedical entity extraction.
"""
function preprocess_pubmed_text(text::String)
    # Remove HTML tags
    text = replace(text, r"<[^>]*>" => "")
    
    # Remove extra whitespace
    text = replace(text, r"\s+" => " ")
    
    # Remove special characters but keep biomedical terms
    text = replace(text, r"[^\w\s\-\.\(\)\[\]]" => " ")
    
    # Normalize case for common biomedical terms
    text = replace(text, r"\bDNA\b" => "DNA")
    text = replace(text, r"\bRNA\b" => "RNA")
    text = replace(text, r"\bATP\b" => "ATP")
    text = replace(text, r"\bGTP\b" => "GTP")
    text = replace(text, r"\bADP\b" => "ADP")
    text = replace(text, r"\bGDP\b" => "GDP")
    
    # Remove common stop words but keep biomedical terms
    stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
    words = split(text, " ")
    filtered_words = filter(word -> !(lowercase(word) in stop_words), words)
    
    return join(filtered_words, " ")
end

"""
    extract_sentences(text::String)

Extract sentences from text for processing.
"""
function extract_sentences(text::String)
    # Simple sentence splitting
    sentences = split(text, r"[.!?]+")
    
    # Filter out empty sentences
    sentences = filter(s -> !isempty(strip(s)), sentences)
    
    # Clean up sentences
    sentences = map(s -> strip(s), sentences)
    
    return sentences
end

# ============================================================================
# Error Handling
# ============================================================================

"""
    handle_pubmed_error(response::PubMedResponse, context::String="")

Handle PubMed API errors with appropriate logging and fallback.
"""
function handle_pubmed_error(response::PubMedResponse, context::String="")
    if response.success
        return
    end
    
    error_msg = "PubMed API error"
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
    return PubMedResponse(false, Dict{String,Any}(), error_msg, response.status_code)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    is_pubmed_available(client::PubMedClient)

Check if PubMed API is available.
"""
function is_pubmed_available(client::PubMedClient)
    response = search_pubmed(client, "test"; max_results=1)
    return response.success
end

"""
    get_pubmed_status(client::PubMedClient)

Get the current status of the PubMed client.
"""
function get_pubmed_status(client::PubMedClient)
    return Dict{String,Any}(
        "api_available" => is_pubmed_available(client),
        "rate_limit_remaining" => client.config.rate_limit - client.request_count,
        "cache_enabled" => client.config.cache_enabled,
        "cache_size" => length(client.cache.data),
        "last_request_time" => client.last_request_time
    )
end
