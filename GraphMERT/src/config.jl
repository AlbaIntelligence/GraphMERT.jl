"""
Configuration management for GraphMERT.jl

This module handles configuration loading, validation, and management
for the GraphMERT implementation.
"""

using JSON
using Logging
# using DocStringExtensions  # Temporarily disabled
# ProcessingOptions will be available from main module

# ============================================================================
# Default Configuration
# ============================================================================

"""
    get_default_config()

Get the default configuration for GraphMERT.

"""
function get_default_config()
  return Dict{String,Any}(
    "model" => Dict{String,Any}(
      "vocab_size" => 50265,
      "hidden_size" => 768,
      "num_attention_heads" => 12,
      "num_hidden_layers" => 12,
      "max_position_embeddings" => 512,
      "type_vocab_size" => 2,
      "initializer_range" => 0.02,
      "layer_norm_eps" => 1e-12,
    ),
    "processing" => Dict{String,Any}(
      "max_length" => 512,
      "batch_size" => 32,
      "use_umls" => true,
      "use_helper_llm" => true,
      "confidence_threshold" => 0.5,
      "cache_enabled" => true,
      "parallel_processing" => false,
      "verbose" => false,
    ),
    "api" => Dict{String,Any}(
      "umls" => Dict{String,Any}(
        "base_url" => "https://uts-ws.nlm.nih.gov/rest",
        "rate_limit" => 100,
        "timeout" => 30,
        "retry_attempts" => 3,
        "cache_ttl" => 3600,
      ),
      "helper_llm" => Dict{String,Any}(
        "base_url" => "https://api.openai.com/v1",
        "model" => "gpt-4",
        "max_tokens" => 1000,
        "temperature" => 0.1,
        "rate_limit" => 10000,
        "timeout" => 60,
        "retry_attempts" => 3,
        "cache_ttl" => 3600,
      ),
    ),
    "evaluation" => Dict{String,Any}(
      "factscore_threshold" => 0.7,
      "validity_threshold" => 0.8,
      "graphrag_threshold" => 0.75,
    ),
  )
end

# ============================================================================
# Configuration Loading
# ============================================================================

"""
    load_config(config_path::String)

Load configuration from a JSON file.

"""
function load_config(config_path::String)
  try
    if !isfile(config_path)
      @warn "Configuration file not found: $config_path. Using default configuration."
      return get_default_config()
    end

    config_data = JSON.parsefile(config_path)
    return merge_config(get_default_config(), config_data)
  catch e
    @error "Failed to load configuration from $config_path: $e"
    @warn "Using default configuration."
    return get_default_config()
  end
end

"""
    merge_config(default::Dict{String, Any}, user::Dict{String, Any})

Merge user configuration with default configuration.

"""
function merge_config(default::Dict{String,Any}, user::Dict{String,Any})
  result = deepcopy(default)

  for (key, value) in user
    if haskey(result, key) && isa(value, Dict) && isa(result[key], Dict)
      result[key] = merge_config(result[key], value)
    else
      result[key] = value
    end
  end

  return result
end

# ============================================================================
# Configuration Validation
# ============================================================================

"""
    validate_config(config::Dict{String, Any})

Validate configuration parameters.

"""
function validate_config(config::Dict{String,Any})
  errors = String[]

  # Validate model configuration
  if haskey(config, "model")
    model_config = config["model"]

    if haskey(model_config, "vocab_size") && model_config["vocab_size"] <= 0
      push!(errors, "vocab_size must be positive")
    end

    if haskey(model_config, "hidden_size") && model_config["hidden_size"] <= 0
      push!(errors, "hidden_size must be positive")
    end

    if haskey(model_config, "num_attention_heads") &&
       model_config["num_attention_heads"] <= 0
      push!(errors, "num_attention_heads must be positive")
    end

    if haskey(model_config, "num_hidden_layers") &&
       model_config["num_hidden_layers"] <= 0
      push!(errors, "num_hidden_layers must be positive")
    end
  end

  # Validate processing configuration
  if haskey(config, "processing")
    processing_config = config["processing"]

    if haskey(processing_config, "max_length") && processing_config["max_length"] <= 0
      push!(errors, "max_length must be positive")
    end

    if haskey(processing_config, "batch_size") && processing_config["batch_size"] <= 0
      push!(errors, "batch_size must be positive")
    end

    if haskey(processing_config, "confidence_threshold") && (
      processing_config["confidence_threshold"] < 0.0 ||
      processing_config["confidence_threshold"] > 1.0
    )
      push!(errors, "confidence_threshold must be between 0.0 and 1.0")
    end
  end

  # Validate API configuration
  if haskey(config, "api")
    api_config = config["api"]

    if haskey(api_config, "umls")
      umls_config = api_config["umls"]

      if haskey(umls_config, "rate_limit") && umls_config["rate_limit"] <= 0
        push!(errors, "UMLS rate_limit must be positive")
      end

      if haskey(umls_config, "timeout") && umls_config["timeout"] <= 0
        push!(errors, "UMLS timeout must be positive")
      end
    end

    if haskey(api_config, "helper_llm")
      llm_config = api_config["helper_llm"]

      if haskey(llm_config, "rate_limit") && llm_config["rate_limit"] <= 0
        push!(errors, "Helper LLM rate_limit must be positive")
      end

      if haskey(llm_config, "timeout") && llm_config["timeout"] <= 0
        push!(errors, "Helper LLM timeout must be positive")
      end
    end
  end

  if !isempty(errors)
    error("Configuration validation failed: " * join(errors, ", "))
  end

  return true
end

# ============================================================================
# Configuration Utilities
# ============================================================================

"""
    save_config(config::Dict{String, Any}, config_path::String)

Save configuration to a JSON file.

"""
function save_config(config::Dict{String,Any}, config_path::String)
  try
    open(config_path, "w") do io
      JSON.print(io, config, 2)
    end
    @info "Configuration saved to: $config_path"
  catch e
    @error "Failed to save configuration to $config_path: $e"
    throw(e)
  end
end

"""
    get_config_value(config::Dict{String, Any}, key_path::String, default_value::Any=nothing)

Get a configuration value using dot notation (e.g., "model.hidden_size").

"""
function get_config_value(
  config::Dict{String,Any},
  key_path::String,
  default_value::Any=nothing,
)
  keys = split(key_path, ".")
  current = config

  for key in keys
    if haskey(current, key)
      current = current[key]
    else
      return default_value
    end
  end

  return current
end

"""
    set_config_value!(config::Dict{String, Any}, key_path::String, value::Any)

Set a configuration value using dot notation.

"""
function set_config_value!(config::Dict{String,Any}, key_path::String, value::Any)
  keys = split(key_path, ".")
  current = config

  for key in keys[1:(end-1)]
    if !haskey(current, key) || !isa(current[key], Dict)
      current[key] = Dict{String,Any}()
    end
    current = current[key]
  end

  current[keys[end]] = value
end

"""
    default_processing_options(;
        batch_size::Int=32,
        max_length::Int=1024,
        device::Symbol=:cpu,
        use_amp::Bool=false,
        num_workers::Int=1,
        seed::Union{Int, Nothing}=nothing,
        top_k_predictions::Int=20,
        similarity_threshold::Float64=0.8,
        enable_provenance_tracking::Bool=true
    )::ProcessingOptions

Create default processing options for GraphMERT.

"""
function default_processing_options(;
  batch_size::Int=32,
  max_length::Int=1024,
  device::Symbol=:cpu,
  use_amp::Bool=false,
  num_workers::Int=1,
  seed::Union{Int,Nothing}=nothing,
  top_k_predictions::Int=20,
  similarity_threshold::Float64=0.8,
  enable_provenance_tracking::Bool=true,
)
  return GraphMERT.ProcessingOptions(
    batch_size,
    max_length,
    device,
    use_amp,
    num_workers,
    seed,
    top_k_predictions,
    similarity_threshold,
    enable_provenance_tracking,
  )
end
