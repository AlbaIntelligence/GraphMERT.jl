"""
API Helper functions for GraphMERT.jl

This module provides utility functions for API operations,
including request formatting, response parsing, and error handling.
"""

using JSON
using HTTP
using Dates
# using DocStringExtensions  # Temporarily disabled

# ============================================================================
# Request Helper Functions
# ============================================================================

"""
    format_request_body(data::Dict{String, Any})

Format request body as JSON string.

"""
function format_request_body(data::Dict{String,Any})
  return JSON.json(data)
end

"""
    create_http_headers(api_key::Union{String, Nothing}, content_type::String = "application/json")

Create HTTP headers for API requests.

"""
function create_http_headers(api_key::Union{String,Nothing}, content_type::String="application/json")
  headers = Dict{String,String}()

  headers["Content-Type"] = content_type
  headers["User-Agent"] = "GraphMERT.jl/0.1.0"
  headers["Accept"] = "application/json"

  if api_key !== nothing
    headers["Authorization"] = "Bearer $api_key"
  end

  return headers
end

"""
    parse_response(response::HTTP.Response)

Parse HTTP response and extract JSON data.

"""
function parse_response(response::HTTP.Response)
  if response.status >= 200 && response.status < 300
    try
      return JSON.parse(String(response.body))
    catch e
      throw(ArgumentError("Failed to parse JSON response: $e"))
    end
  else
    error_msg = "HTTP $(response.status): $(String(response.body))"
    throw(HTTPException(response.status, error_msg))
  end
end

# ============================================================================
# Error Handling
# ============================================================================

"""
    HTTPException <: Exception

Exception for HTTP-related errors.

"""
struct HTTPException <: Exception
  status_code::Int
  message::String
end

"""
    handle_api_error(response::HTTP.Response)

Handle API errors and throw appropriate exceptions.

"""
function handle_api_error(response::HTTP.Response)
  if response.status >= 400
    error_data = try
      JSON.parse(String(response.body))
    catch
      Dict{String,Any}("message" => String(response.body))
    end

    error_msg = get(error_data, "message", "Unknown error")
    throw(HTTPException(response.status, error_msg))
  end
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    validate_request_data(data::Dict{String, Any}, required_fields::Vector{String})

Validate request data has required fields.

"""
function validate_request_data(data::Dict{String,Any}, required_fields::Vector{String})
  missing_fields = String[]

  for field in required_fields
    if !haskey(data, field)
      push!(missing_fields, field)
    end
  end

  if !isempty(missing_fields)
    throw(ArgumentError("Missing required fields: $(join(missing_fields, ", "))"))
  end

  return true
end

"""
    create_timestamp()

Create ISO 8601 timestamp string.

"""
function create_timestamp()
  return Dates.format(now(), Dates.ISO8601Format)
end

"""
    sanitize_text(text::String, max_length::Int = 1000)

Sanitize text for API requests.

"""
function sanitize_text(text::String, max_length::Int=1000)
  # Remove control characters
  sanitized = replace(text, r"[\x00-\x1f\x7f-\x9f]" => "")

  # Truncate if too long
  if length(sanitized) > max_length
    sanitized = sanitized[1:max_length]
  end

  return sanitized
end
