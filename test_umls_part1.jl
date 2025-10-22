"""
UMLS (Unified Medical Language System) integration for biomedical entity linking.

This module provides a complete UMLS API client with authentication, rate limiting,
caching, and error handling for biomedical entity linking and validation.
"""

# HTTP and JSON3 dependencies for UMLS API (commented out for now)
# using HTTP
# using JSON3
using Dates

# EntityLinkingResult is defined in the main GraphMERT module
# No import needed - it's available directly

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

  function UMLSCache(max_size::Int=1000, ttl_seconds::Int=3600)
    new(Dict{String,Tuple{Any,DateTime}}(),
      Dict{String,Tuple{Any,DateTime}}(),
      max_size, ttl_seconds)
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

# Arguments
- `api_key::String`: UMLS API key from NLM UTS
- `base_url::String`: UMLS REST API base URL (default: "https://uts-ws.nlm.nih.gov/rest")
- `timeout::Int`: Request timeout in seconds (default: 30)
- `max_retries::Int`: Maximum retry attempts (default: 3)
- `semantic_networks::Vector{String}`: Semantic networks to search (default: ["SNOMEDCT_US", "MSH", "RXNORM"])
- `rate_limit::Int`: Requests per minute limit (default: 100)
- `cache_ttl::Int`: Cache TTL in seconds (default: 3600)

# Returns
- `UMLSClient`: Configured UMLS client

# Example
# ```julia
# client = create_umls_client(ENV["UMLS_API_KEY"])
# linking_result = link_entity_to_umls("diabetes mellitus", client)
# if linking_result !== nothing
#     println("CUI: $(linking_result.cui), Confidence: $(linking_result.similarity_score)")
# end
# ```
"""
function create_umls_client(api_key::String;
  base_url::String="https://uts-ws.nlm.nih.gov/rest",