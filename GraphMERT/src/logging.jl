"""
Logging system for GraphMERT.jl

This module provides comprehensive logging capabilities for debugging, monitoring,
and performance tracking throughout the GraphMERT implementation.
"""

using Logging
using Dates
using JSON

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
# Logging Levels
# ============================================================================

const LOG_LEVELS = Dict(
  :debug => 0,
  :info => 1,
  :warn => 2,
  :error => 3
)

"""
    should_log(level::Symbol, config::LoggingConfig)

Check if a log level should be logged based on configuration.
"""
function should_log(level::Symbol, config::LoggingConfig)
  return LOG_LEVELS[level] >= LOG_LEVELS[config.level]
end

# ============================================================================
# Logging Functions
# ============================================================================

"""
    log_debug(message::String, context::Dict{String, Any} = Dict{String, Any}())

Log a debug message.
"""
function log_debug(message::String, context::Dict{String,Any}=Dict{String,Any}())
  log_message(:debug, message, context)
end

"""
    log_info(message::String, context::Dict{String, Any} = Dict{String, Any}())

Log an info message.
"""
function log_info(message::String, context::Dict{String,Any}=Dict{String,Any}())
  log_message(:info, message, context)
end

"""
    log_warn(message::String, context::Dict{String, Any} = Dict{String, Any}())

Log a warning message.
"""
function log_warn(message::String, context::Dict{String,Any}=Dict{String,Any}())
  log_message(:warn, message, context)
end

"""
    log_error(message::String, context::Dict{String, Any} = Dict{String, Any}())

Log an error message.
"""
function log_error(message::String, context::Dict{String,Any}=Dict{String,Any}())
  log_message(:error, message, context)
end

"""
    log_message(level::Symbol, message::String, context::Dict{String, Any})

Log a message with specified level and context.
"""
function log_message(level::Symbol, message::String, context::Dict{String,Any})
  config = get_logging_config()

  if !should_log(level, config)
    return
  end

  timestamp = now()
  formatted_message = format_log_message(level, message, context, timestamp, config)

  # Console logging
  if config.enable_console_logging
    println(formatted_message)
  end

  # File logging
  if config.enable_file_logging
    write_to_log_file(formatted_message, config.log_file_path)
  end
end

# ============================================================================
# Performance Logging
# ============================================================================

"""
    log_performance(operation::String, duration::Float64, metrics::Dict{String, Any} = Dict{String, Any}())

Log performance metrics for an operation.
"""
function log_performance(operation::String, duration::Float64, metrics::Dict{String,Any}=Dict{String,Any}())
  config = get_logging_config()

  if !config.enable_performance_logging
    return
  end

  context = Dict{String,Any}(
    "operation" => operation,
    "duration" => duration,
    "metrics" => metrics
  )

  log_info("Performance: $operation completed in $(duration)s", context)
end

"""
    log_memory_usage(operation::String, memory_mb::Float64)

Log memory usage for an operation.
"""
function log_memory_usage(operation::String, memory_mb::Float64)
  context = Dict{String,Any}(
    "operation" => operation,
    "memory_mb" => memory_mb
  )

  log_info("Memory usage: $operation used $(memory_mb)MB", context)
end

"""
    log_processing_speed(operation::String, tokens_per_second::Float64)

Log processing speed for an operation.
"""
function log_processing_speed(operation::String, tokens_per_second::Float64)
  context = Dict{String,Any}(
    "operation" => operation,
    "tokens_per_second" => tokens_per_second
  )

  log_info("Processing speed: $operation processed $(tokens_per_second) tokens/second", context)
end

# ============================================================================
# Model Logging
# ============================================================================

"""
    log_model_loading(model_path::String, duration::Float64)

Log model loading performance.
"""
function log_model_loading(model_path::String, duration::Float64)
  context = Dict{String,Any}(
    "model_path" => model_path,
    "duration" => duration
  )

  log_info("Model loaded: $model_path in $(duration)s", context)
end

"""
    log_model_saving(model_path::String, duration::Float64)

Log model saving performance.
"""
function log_model_saving(model_path::String, duration::Float64)
  context = Dict{String,Any}(
    "model_path" => model_path,
    "duration" => duration
  )

  log_info("Model saved: $model_path in $(duration)s", context)
end

# ============================================================================
# Processing Logging
# ============================================================================

"""
    log_entity_extraction(text_length::Int, num_entities::Int, duration::Float64)

Log entity extraction performance.
"""
function log_entity_extraction(text_length::Int, num_entities::Int, duration::Float64)
  context = Dict{String,Any}(
    "text_length" => text_length,
    "num_entities" => num_entities,
    "duration" => duration
  )

  log_info("Entity extraction: $(num_entities) entities from $(text_length) chars in $(duration)s", context)
end

"""
    log_relation_extraction(num_entities::Int, num_relations::Int, duration::Float64)

Log relation extraction performance.
"""
function log_relation_extraction(num_entities::Int, num_relations::Int, duration::Float64)
  context = Dict{String,Any}(
    "num_entities" => num_entities,
    "num_relations" => num_relations,
    "duration" => duration
  )

  log_info("Relation extraction: $(num_relations) relations from $(num_entities) entities in $(duration)s", context)
end

# ============================================================================
# Evaluation Logging
# ============================================================================

"""
    log_evaluation_metrics(graph::KnowledgeGraph, metrics::Dict{String, Float64})

Log evaluation metrics for a knowledge graph.
"""
function log_evaluation_metrics(graph::KnowledgeGraph, metrics::Dict{String,Float64})
  context = Dict{String,Any}(
    "num_entities" => length(graph.entities),
    "num_relations" => length(graph.relations),
    "metrics" => metrics
  )

  log_info("Evaluation metrics: $(JSON.json(metrics))", context)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    format_log_message(level::Symbol, message::String, context::Dict{String, Any},
                      timestamp::DateTime, config::LoggingConfig)

Format a log message according to the configuration.
"""
function format_log_message(level::Symbol, message::String, context::Dict{String,Any},
  timestamp::DateTime, config::LoggingConfig)
  # Format timestamp
  timestamp_str = Dates.format(timestamp, "yyyy-mm-dd HH:MM:SS")

  # Format context
  context_str = if !isempty(context)
    context_pairs = ["$k: $v" for (k, v) in context]
    " | " * join(context_pairs, " | ")
  else
    ""
  end

  # Format level
  level_str = uppercase(string(level))

  # Create formatted message
  formatted = config.log_format
  formatted = replace(formatted, "%timestamp" => timestamp_str)
  formatted = replace(formatted, "%level" => level_str)
  formatted = replace(formatted, "%message" => message)

  return formatted * context_str
end

"""
    write_to_log_file(message::String, log_file_path::String)

Write a message to the log file.
"""
function write_to_log_file(message::String, log_file_path::String)
  try
    open(log_file_path, "a") do io
      println(io, message)
    end
  catch e
    # If we can't write to the log file, at least print to console
    println("Warning: Could not write to log file $log_file_path: $e")
  end
end

# ============================================================================
# Global Configuration
# ============================================================================

# Global logging configuration
_global_logging_config = LoggingConfig()

"""
    set_logging_config(config::LoggingConfig)

Set the global logging configuration.
"""
function set_logging_config(config::LoggingConfig)
  global _global_logging_config = config
end

"""
    get_logging_config()

Get the current global logging configuration.
"""
function get_logging_config()
  return _global_logging_config
end

# ============================================================================
# Logging Decorators
# ============================================================================

"""
    with_logging(f::Function, operation::String)

Execute a function with logging context.
"""
function with_logging(f::Function, operation::String)
  log_info("Starting: $operation")
  start_time = time()

  try
    result = f()
    duration = time() - start_time
    log_info("Completed: $operation in $(duration)s")
    return result
  catch e
    duration = time() - start_time
    log_error("Failed: $operation after $(duration)s - $e")
    rethrow(e)
  end
end

"""
    with_performance_logging(f::Function, operation::String, metrics::Dict{String, Any} = Dict{String, Any}())

Execute a function with performance logging.
"""
function with_performance_logging(f::Function, operation::String, metrics::Dict{String,Any}=Dict{String,Any}())
  start_time = time()

  try
    result = f()
    duration = time() - start_time
    log_performance(operation, duration, metrics)
    return result
  catch e
    duration = time() - start_time
    log_error("Performance logging failed: $operation after $(duration)s - $e")
    rethrow(e)
  end
end

# ============================================================================
# Log Analysis
# ============================================================================

"""
    analyze_log_file(log_file_path::String)

Analyze a log file for performance metrics.
"""
function analyze_log_file(log_file_path::String)
  if !isfile(log_file_path)
    return Dict{String,Any}()
  end

  performance_metrics = Dict{String,Vector{Float64}}()

  try
    open(log_file_path, "r") do io
      for line in eachline(io)
        if occursin("Performance:", line)
          # Extract performance data
          # This is a simplified parser - in practice, you'd use more sophisticated parsing
          if occursin("duration", line)
            # Extract duration and operation
            # Implementation would depend on log format
          end
        end
      end
    end
  catch e
    log_error("Failed to analyze log file: $e")
  end

  return performance_metrics
end

"""
    get_performance_summary(log_file_path::String)

Get a summary of performance metrics from the log file.
"""
function get_performance_summary(log_file_path::String)
  metrics = analyze_log_file(log_file_path)

  summary = Dict{String,Any}()
  for (operation, durations) in metrics
    summary[operation] = Dict{String,Any}(
      "count" => length(durations),
      "mean" => mean(durations),
      "min" => minimum(durations),
      "max" => maximum(durations),
      "std" => std(durations)
    )
  end

  return summary
end
