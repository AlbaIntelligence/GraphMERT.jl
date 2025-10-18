"""
Configuration system for GraphMERT.jl

This module provides comprehensive configuration options for all aspects of
the GraphMERT implementation, including processing, training, and evaluation.
"""

using Dates
using JSON

# ============================================================================
# Processing Configuration
# ============================================================================

"""
    ProcessingOptions

Configuration options for text processing and knowledge graph extraction.
"""
struct ProcessingOptions
  confidence_threshold::Float64
  max_entities::Int
  max_relations::Int
  umls_enabled::Bool
  helper_llm_enabled::Bool
  performance_mode::Symbol
  batch_size::Int
  memory_limit::Int
  timeout_seconds::Int

  function ProcessingOptions(;
    confidence_threshold::Float64=0.8,
    max_entities::Int=100,
    max_relations::Int=50,
    umls_enabled::Bool=true,
    helper_llm_enabled::Bool=true,
    performance_mode::Symbol=:balanced,
    batch_size::Int=32,
    memory_limit::Int=4096,  # MB
    timeout_seconds::Int=300
  )
    @assert 0.0 <= confidence_threshold <= 1.0 "Confidence threshold must be between 0.0 and 1.0"
    @assert max_entities > 0 "Max entities must be positive"
    @assert max_relations > 0 "Max relations must be positive"
    @assert performance_mode in [:fast, :balanced, :accurate] "Performance mode must be :fast, :balanced, or :accurate"
    @assert batch_size > 0 "Batch size must be positive"
    @assert memory_limit > 0 "Memory limit must be positive"
    @assert timeout_seconds > 0 "Timeout must be positive"

    new(confidence_threshold, max_entities, max_relations, umls_enabled,
      helper_llm_enabled, performance_mode, batch_size, memory_limit, timeout_seconds)
  end
end

"""
    GraphMERTConfig

Comprehensive configuration for GraphMERT processing.
"""
struct GraphMERTConfig
  model_path::String
  processing_options::ProcessingOptions
  umls_config::UMLSIntegration
  training_config::MLM_MNM_Training
  performance_config::PerformanceConfig
  logging_config::LoggingConfig

  function GraphMERTConfig(;
    model_path::String="",
    processing_options::ProcessingOptions=ProcessingOptions(),
    umls_config::UMLSIntegration=UMLSIntegration(true),
    training_config::MLM_MNM_Training=MLM_MNM_Training(),
    performance_config::PerformanceConfig=PerformanceConfig(),
    logging_config::LoggingConfig=LoggingConfig()
  )
    new(model_path, processing_options, umls_config, training_config,
      performance_config, logging_config)
  end
end

# ============================================================================
# Training Configuration
# ============================================================================

"""
    MLM_MNM_Training

Configuration for MLM and MNM training objectives.
"""
struct MLM_MNM_Training
  mlm_probability::Float64
  mnm_probability::Float64
  span_length::Int
  boundary_loss_weight::Float64
  learning_rate::Float64
  batch_size::Int
  num_epochs::Int
  warmup_steps::Int
  weight_decay::Float64

  function MLM_MNM_Training(;
    mlm_probability::Float64=0.15,
    mnm_probability::Float64=0.15,
    span_length::Int=3,
    boundary_loss_weight::Float64=1.0,
    learning_rate::Float64=2e-5,
    batch_size::Int=16,
    num_epochs::Int=3,
    warmup_steps::Int=1000,
    weight_decay::Float64=0.01
  )
    @assert 0.0 <= mlm_probability <= 1.0 "MLM probability must be between 0.0 and 1.0"
    @assert 0.0 <= mnm_probability <= 1.0 "MNM probability must be between 0.0 and 1.0"
    @assert span_length > 0 "Span length must be positive"
    @assert boundary_loss_weight >= 0.0 "Boundary loss weight must be non-negative"
    @assert learning_rate > 0.0 "Learning rate must be positive"
    @assert batch_size > 0 "Batch size must be positive"
    @assert num_epochs > 0 "Number of epochs must be positive"
    @assert warmup_steps >= 0 "Warmup steps must be non-negative"
    @assert weight_decay >= 0.0 "Weight decay must be non-negative"

    new(mlm_probability, mnm_probability, span_length, boundary_loss_weight,
      learning_rate, batch_size, num_epochs, warmup_steps, weight_decay)
  end
end

# ============================================================================
# Performance Configuration
# ============================================================================

"""
    PerformanceConfig

Configuration for performance optimization and monitoring.
"""
struct PerformanceConfig
  target_tokens_per_second::Int
  max_memory_gb::Float64
  enable_optimization::Bool
  optimization_level::Symbol
  cache_size::Int
  num_threads::Int

  function PerformanceConfig(;
    target_tokens_per_second::Int=5000,
    max_memory_gb::Float64=4.0,
    enable_optimization::Bool=true,
    optimization_level::Symbol=:balanced,
    cache_size::Int=1000,
    num_threads::Int=4
  )
    @assert target_tokens_per_second > 0 "Target tokens per second must be positive"
    @assert max_memory_gb > 0.0 "Max memory must be positive"
    @assert optimization_level in [:none, :basic, :balanced, :aggressive] "Invalid optimization level"
    @assert cache_size > 0 "Cache size must be positive"
    @assert num_threads > 0 "Number of threads must be positive"

    new(target_tokens_per_second, max_memory_gb, enable_optimization,
      optimization_level, cache_size, num_threads)
  end
end

# ============================================================================
# Logging Configuration
# ============================================================================

"""
    LoggingConfig

Configuration for logging and monitoring.
"""
struct LoggingConfig
  level::Symbol
  enable_file_logging::Bool
  log_file_path::String
  enable_console_logging::Bool
  enable_performance_logging::Bool
  log_format::String

  function LoggingConfig(;
    level::Symbol=:info,
    enable_file_logging::Bool=true,
    log_file_path::String="graphmert.log",
    enable_console_logging::Bool=true,
    enable_performance_logging::Bool=true,
    log_format::String="%Y-%m-%d %H:%M:%S [%level] %message"
  )
    @assert level in [:debug, :info, :warn, :error] "Invalid log level"
    @assert !isempty(log_format) "Log format cannot be empty"

    new(level, enable_file_logging, log_file_path, enable_console_logging,
      enable_performance_logging, log_format)
  end
end

# ============================================================================
# UMLS Configuration
# ============================================================================

"""
    UMLSIntegration

Configuration for UMLS integration.
"""
struct UMLSIntegration
  enabled::Bool
  api_key::String
  base_url::String
  timeout_seconds::Int
  max_retries::Int
  confidence_threshold::Float64
  cache_enabled::Bool
  cache_size::Int

  function UMLSIntegration(;
    enabled::Bool=true,
    api_key::String="",
    base_url::String="https://uts-ws.nlm.nih.gov/rest",
    timeout_seconds::Int=30,
    max_retries::Int=3,
    confidence_threshold::Float64=0.8,
    cache_enabled::Bool=true,
    cache_size::Int=10000
  )
    @assert 0.0 <= confidence_threshold <= 1.0 "Confidence threshold must be between 0.0 and 1.0"
    @assert timeout_seconds > 0 "Timeout must be positive"
    @assert max_retries >= 0 "Max retries must be non-negative"
    @assert cache_size > 0 "Cache size must be positive"

    new(enabled, api_key, base_url, timeout_seconds, max_retries,
      confidence_threshold, cache_enabled, cache_size)
  end
end

# ============================================================================
# Configuration Utilities
# ============================================================================

"""
    load_config(config_path::String)

Load configuration from JSON file.
"""
function load_config(config_path::String)
  if !isfile(config_path)
    throw(ArgumentError("Configuration file not found: $config_path"))
  end

  try
    config_data = JSON.parsefile(config_path)
    return parse_config_dict(config_data)
  catch e
    throw(ArgumentError("Failed to parse configuration file: $e"))
  end
end

"""
    save_config(config::GraphMERTConfig, config_path::String)

Save configuration to JSON file.
"""
function save_config(config::GraphMERTConfig, config_path::String)
  config_dict = config_to_dict(config)

  try
    open(config_path, "w") do io
      JSON.print(io, config_dict, 2)
    end
  catch e
    throw(ArgumentError("Failed to save configuration file: $e"))
  end
end

"""
    parse_config_dict(data::Dict{String, Any})

Parse configuration from dictionary.
"""
function parse_config_dict(data::Dict{String,Any})
  # Parse processing options
  processing_data = get(data, "processing", Dict{String,Any}())
  processing_options = ProcessingOptions(
    confidence_threshold=get(processing_data, "confidence_threshold", 0.8),
    max_entities=get(processing_data, "max_entities", 100),
    max_relations=get(processing_data, "max_relations", 50),
    umls_enabled=get(processing_data, "umls_enabled", true),
    helper_llm_enabled=get(processing_data, "helper_llm_enabled", true),
    performance_mode=Symbol(get(processing_data, "performance_mode", "balanced")),
    batch_size=get(processing_data, "batch_size", 32),
    memory_limit=get(processing_data, "memory_limit", 4096),
    timeout_seconds=get(processing_data, "timeout_seconds", 300)
  )

  # Parse UMLS configuration
  umls_data = get(data, "umls", Dict{String,Any}())
  umls_config = UMLSIntegration(
    enabled=get(umls_data, "enabled", true),
    api_key=get(umls_data, "api_key", ""),
    base_url=get(umls_data, "base_url", "https://uts-ws.nlm.nih.gov/rest"),
    timeout_seconds=get(umls_data, "timeout_seconds", 30),
    max_retries=get(umls_data, "max_retries", 3),
    confidence_threshold=get(umls_data, "confidence_threshold", 0.8),
    cache_enabled=get(umls_data, "cache_enabled", true),
    cache_size=get(umls_data, "cache_size", 10000)
  )

  # Parse training configuration
  training_data = get(data, "training", Dict{String,Any}())
  training_config = MLM_MNM_Training(
    mlm_probability=get(training_data, "mlm_probability", 0.15),
    mnm_probability=get(training_data, "mnm_probability", 0.15),
    span_length=get(training_data, "span_length", 3),
    boundary_loss_weight=get(training_data, "boundary_loss_weight", 1.0),
    learning_rate=get(training_data, "learning_rate", 2e-5),
    batch_size=get(training_data, "batch_size", 16),
    num_epochs=get(training_data, "num_epochs", 3),
    warmup_steps=get(training_data, "warmup_steps", 1000),
    weight_decay=get(training_data, "weight_decay", 0.01)
  )

  # Parse performance configuration
  performance_data = get(data, "performance", Dict{String,Any}())
  performance_config = PerformanceConfig(
    target_tokens_per_second=get(performance_data, "target_tokens_per_second", 5000),
    max_memory_gb=get(performance_data, "max_memory_gb", 4.0),
    enable_optimization=get(performance_data, "enable_optimization", true),
    optimization_level=Symbol(get(performance_data, "optimization_level", "balanced")),
    cache_size=get(performance_data, "cache_size", 1000),
    num_threads=get(performance_data, "num_threads", 4)
  )

  # Parse logging configuration
  logging_data = get(data, "logging", Dict{String,Any}())
  logging_config = LoggingConfig(
    level=Symbol(get(logging_data, "level", "info")),
    enable_file_logging=get(logging_data, "enable_file_logging", true),
    log_file_path=get(logging_data, "log_file_path", "graphmert.log"),
    enable_console_logging=get(logging_data, "enable_console_logging", true),
    enable_performance_logging=get(logging_data, "enable_performance_logging", true),
    log_format=get(logging_data, "log_format", "%Y-%m-%d %H:%M:%S [%level] %message")
  )

  return GraphMERTConfig(
    model_path=get(data, "model_path", ""),
    processing_options=processing_options,
    umls_config=umls_config,
    training_config=training_config,
    performance_config=performance_config,
    logging_config=logging_config
  )
end

"""
    config_to_dict(config::GraphMERTConfig)

Convert configuration to dictionary.
"""
function config_to_dict(config::GraphMERTConfig)
  return Dict{String,Any}(
    "model_path" => config.model_path,
    "processing" => Dict{String,Any}(
      "confidence_threshold" => config.processing_options.confidence_threshold,
      "max_entities" => config.processing_options.max_entities,
      "max_relations" => config.processing_options.max_relations,
      "umls_enabled" => config.processing_options.umls_enabled,
      "helper_llm_enabled" => config.processing_options.helper_llm_enabled,
      "performance_mode" => string(config.processing_options.performance_mode),
      "batch_size" => config.processing_options.batch_size,
      "memory_limit" => config.processing_options.memory_limit,
      "timeout_seconds" => config.processing_options.timeout_seconds
    ),
    "umls" => Dict{String,Any}(
      "enabled" => config.umls_config.enabled,
      "api_key" => config.umls_config.api_key,
      "base_url" => config.umls_config.base_url,
      "timeout_seconds" => config.umls_config.timeout_seconds,
      "max_retries" => config.umls_config.max_retries,
      "confidence_threshold" => config.umls_config.confidence_threshold,
      "cache_enabled" => config.umls_config.cache_enabled,
      "cache_size" => config.umls_config.cache_size
    ),
    "training" => Dict{String,Any}(
      "mlm_probability" => config.training_config.mlm_probability,
      "mnm_probability" => config.training_config.mnm_probability,
      "span_length" => config.training_config.span_length,
      "boundary_loss_weight" => config.training_config.boundary_loss_weight,
      "learning_rate" => config.training_config.learning_rate,
      "batch_size" => config.training_config.batch_size,
      "num_epochs" => config.training_config.num_epochs,
      "warmup_steps" => config.training_config.warmup_steps,
      "weight_decay" => config.training_config.weight_decay
    ),
    "performance" => Dict{String,Any}(
      "target_tokens_per_second" => config.performance_config.target_tokens_per_second,
      "max_memory_gb" => config.performance_config.max_memory_gb,
      "enable_optimization" => config.performance_config.enable_optimization,
      "optimization_level" => string(config.performance_config.optimization_level),
      "cache_size" => config.performance_config.cache_size,
      "num_threads" => config.performance_config.num_threads
    ),
    "logging" => Dict{String,Any}(
      "level" => string(config.logging_config.level),
      "enable_file_logging" => config.logging_config.enable_file_logging,
      "log_file_path" => config.logging_config.log_file_path,
      "enable_console_logging" => config.logging_config.enable_console_logging,
      "enable_performance_logging" => config.logging_config.enable_performance_logging,
      "log_format" => config.logging_config.log_format
    )
  )
end

# ============================================================================
# Default Configurations
# ============================================================================

"""
    default_config()

Get default GraphMERT configuration.
"""
function default_config()
  return GraphMERTConfig()
end

"""
    fast_config()

Get configuration optimized for speed.
"""
function fast_config()
  return GraphMERTConfig(
    processing_options=ProcessingOptions(
      performance_mode=:fast,
      batch_size=64,
      memory_limit=2048
    ),
    performance_config=PerformanceConfig(
      target_tokens_per_second=10000,
      optimization_level=:aggressive
    )
  )
end

"""
    accurate_config()

Get configuration optimized for accuracy.
"""
function accurate_config()
  return GraphMERTConfig(
    processing_options=ProcessingOptions(
      performance_mode=:accurate,
      confidence_threshold=0.9,
      batch_size=8
    ),
    performance_config=PerformanceConfig(
      target_tokens_per_second=2000,
      optimization_level=:basic
    )
  )
end
