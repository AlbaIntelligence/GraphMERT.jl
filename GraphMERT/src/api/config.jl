"""
API Configuration module for GraphMERT.jl

This module provides configuration management for the GraphMERT API,
including endpoint configuration, authentication, and request handling.
"""

using JSON
using HTTP
# using DocStringExtensions  # Temporarily disabled

# ============================================================================
# API Configuration Types
# ============================================================================

"""
    APIConfig

Configuration for GraphMERT API endpoints.

"""
struct APIConfig
  base_url::String
  timeout::Int
  max_retries::Int
  retry_delay::Float64
  api_key::Union{String,Nothing}

  function APIConfig(;
    base_url::String="https://api.graphmert.ai",
    timeout::Int=30,
    max_retries::Int=3,
    retry_delay::Float64=1.0,
    api_key::Union{String,Nothing}=nothing
  )
    @assert timeout > 0 "Timeout must be positive"
    @assert max_retries >= 0 "Max retries must be non-negative"
    @assert retry_delay > 0 "Retry delay must be positive"

    new(base_url, timeout, max_retries, retry_delay, api_key)
  end
end

"""
    RequestConfig

Configuration for individual API requests.

"""
struct RequestConfig
  headers::Dict{String,String}
  timeout::Int
  retry_on_failure::Bool

  function RequestConfig(;
    headers::Dict{String,String}=Dict{String,String}(),
    timeout::Int=30,
    retry_on_failure::Bool=true
  )
    @assert timeout > 0 "Timeout must be positive"
    new(headers, timeout, retry_on_failure)
  end
end

# ============================================================================
# Configuration Functions
# ============================================================================

"""
    default_api_config()

Create default API configuration.

"""
function default_api_config()
  return APIConfig()
end

"""
    create_request_config(api_config::APIConfig, additional_headers::Dict{String, String} = Dict{String, String}())

Create request configuration from API configuration.

"""
function create_request_config(api_config::APIConfig, additional_headers::Dict{String,String}=Dict{String,String}())
  headers = copy(additional_headers)

  # Add authentication header if API key is provided
  if api_config.api_key !== nothing
    headers["Authorization"] = "Bearer $(api_config.api_key)"
  end

  # Add content type
  headers["Content-Type"] = "application/json"

  return RequestConfig(
    headers=headers,
    timeout=api_config.timeout,
    retry_on_failure=true
  )
end

"""
    validate_api_config(config::APIConfig)

Validate API configuration.

"""
function validate_api_config(config::APIConfig)
  errors = String[]

  if !startswith(config.base_url, "http")
    push!(errors, "Base URL must start with http:// or https://")
  end

  if config.timeout <= 0
    push!(errors, "Timeout must be positive")
  end

  if config.max_retries < 0
    push!(errors, "Max retries must be non-negative")
  end

  if config.retry_delay <= 0
    push!(errors, "Retry delay must be positive")
  end

  return errors
end

"""
    load_api_config_from_file(filepath::String)

Load API configuration from JSON file.

"""
function load_api_config_from_file(filepath::String)
  if !isfile(filepath)
    throw(ArgumentError("Configuration file not found: $filepath"))
  end

  config_data = JSON.parsefile(filepath)

  return APIConfig(
    base_url=get(config_data, "base_url", "https://api.graphmert.ai"),
    timeout=get(config_data, "timeout", 30),
    max_retries=get(config_data, "max_retries", 3),
    retry_delay=get(config_data, "retry_delay", 1.0),
    api_key=get(config_data, "api_key", nothing)
  )
end
