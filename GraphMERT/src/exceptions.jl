"""
Exception types for GraphMERT.jl

This module defines custom exception types for comprehensive error handling
throughout the GraphMERT implementation.
"""

# ============================================================================
# Base Exception Types
# ============================================================================

"""
    GraphMERTError <: Exception

Base exception type for all GraphMERT-related errors.
"""
struct GraphMERTError <: Exception
  message::String
  context::Dict{String,Any}

  function GraphMERTError(message::String, context::Dict{String,Any}=Dict{String,Any}())
    new(message, context)
  end
end

function Base.showerror(io::IO, e::GraphMERTError)
  print(io, "GraphMERTError: $(e.message)")
  if !isempty(e.context)
    print(io, "\nContext:")
    for (key, value) in e.context
      print(io, "\n  $key: $value")
    end
  end
end

# ============================================================================
# Model-Related Exceptions
# ============================================================================

"""
    ModelLoadingError <: GraphMERTError

Error occurred while loading a GraphMERT model.
"""
struct ModelLoadingError <: GraphMERTError
  model_path::String
  original_error::Union{Exception,Nothing}

  function ModelLoadingError(model_path::String, message::String="",
    original_error::Union{Exception,Nothing}=nothing)
    context = Dict("model_path" => model_path)
    if original_error !== nothing
      context["original_error"] = string(original_error)
    end
    new(GraphMERTError(message, context), model_path, original_error)
  end
end

function Base.showerror(io::IO, e::ModelLoadingError)
  print(io, "ModelLoadingError: Failed to load model at '$(e.model_path)'")
  if e.original_error !== nothing
    print(io, "\nOriginal error: $(e.original_error)")
  end
end

"""
    ModelSavingError <: GraphMERTError

Error occurred while saving a GraphMERT model.
"""
struct ModelSavingError <: GraphMERTError
  model_path::String
  original_error::Union{Exception,Nothing}

  function ModelSavingError(model_path::String, message::String="",
    original_error::Union{Exception,Nothing}=nothing)
    context = Dict("model_path" => model_path)
    if original_error !== nothing
      context["original_error"] = string(original_error)
    end
    new(GraphMERTError(message, context), model_path, original_error)
  end
end

function Base.showerror(io::IO, e::ModelSavingError)
  print(io, "ModelSavingError: Failed to save model to '$(e.model_path)'")
  if e.original_error !== nothing
    print(io, "\nOriginal error: $(e.original_error)")
  end
end

# ============================================================================
# Processing Exceptions
# ============================================================================

"""
    EntityExtractionError <: GraphMERTError

Error occurred during entity extraction.
"""
struct EntityExtractionError <: GraphMERTError
  text::String
  extraction_stage::String
  original_error::Union{Exception,Nothing}

  function EntityExtractionError(text::String, extraction_stage::String,
    message::String="",
    original_error::Union{Exception,Nothing}=nothing)
    context = Dict(
      "text" => text,
      "extraction_stage" => extraction_stage
    )
    if original_error !== nothing
      context["original_error"] = string(original_error)
    end
    new(GraphMERTError(message, context), text, extraction_stage, original_error)
  end
end

function Base.showerror(io::IO, e::EntityExtractionError)
  print(io, "EntityExtractionError: Failed to extract entities from text")
  print(io, "\nStage: $(e.extraction_stage)")
  if e.original_error !== nothing
    print(io, "\nOriginal error: $(e.original_error)")
  end
end

"""
    RelationExtractionError <: GraphMERTError

Error occurred during relation extraction.
"""
struct RelationExtractionError <: GraphMERTError
  entities::Vector{String}
  extraction_stage::String
  original_error::Union{Exception,Nothing}

  function RelationExtractionError(entities::Vector{String}, extraction_stage::String,
    message::String="",
    original_error::Union{Exception,Nothing}=nothing)
    context = Dict(
      "entities" => entities,
      "extraction_stage" => extraction_stage
    )
    if original_error !== nothing
      context["original_error"] = string(original_error)
    end
    new(GraphMERTError(message, context), entities, extraction_stage, original_error)
  end
end

function Base.showerror(io::IO, e::RelationExtractionError)
  print(io, "RelationExtractionError: Failed to extract relations between entities")
  print(io, "\nStage: $(e.extraction_stage)")
  if e.original_error !== nothing
    print(io, "\nOriginal error: $(e.original_error)")
  end
end

# ============================================================================
# Configuration Exceptions
# ============================================================================

"""
    ConfigurationError <: GraphMERTError

Error in configuration parameters.
"""
struct ConfigurationError <: GraphMERTError
  parameter::String
  value::Any
  expected_type::String

  function ConfigurationError(parameter::String, value::Any, expected_type::String,
    message::String="")
    context = Dict(
      "parameter" => parameter,
      "value" => string(value),
      "expected_type" => expected_type
    )
    new(GraphMERTError(message, context), parameter, value, expected_type)
  end
end

function Base.showerror(io::IO, e::ConfigurationError)
  print(io, "ConfigurationError: Invalid value for parameter '$(e.parameter)'")
  print(io, "\nValue: $(e.value)")
  print(io, "\nExpected type: $(e.expected_type)")
end

# ============================================================================
# Validation Exceptions
# ============================================================================

"""
    ValidationError <: GraphMERTError

Error in data validation.
"""
struct ValidationError <: GraphMERTError
  object_type::String
  validation_rule::String
  details::String

  function ValidationError(object_type::String, validation_rule::String, details::String,
    message::String="")
    context = Dict(
      "object_type" => object_type,
      "validation_rule" => validation_rule,
      "details" => details
    )
    new(GraphMERTError(message, context), object_type, validation_rule, details)
  end
end

function Base.showerror(io::IO, e::ValidationError)
  print(io, "ValidationError: Validation failed for $(e.object_type)")
  print(io, "\nRule: $(e.validation_rule)")
  print(io, "\nDetails: $(e.details)")
end

# ============================================================================
# Performance Exceptions
# ============================================================================

"""
    PerformanceError <: GraphMERTError

Error related to performance constraints.
"""
struct PerformanceError <: GraphMERTError
  constraint::String
  actual_value::Float64
  threshold::Float64

  function PerformanceError(constraint::String, actual_value::Float64, threshold::Float64,
    message::String="")
    context = Dict(
      "constraint" => constraint,
      "actual_value" => actual_value,
      "threshold" => threshold
    )
    new(GraphMERTError(message, context), constraint, actual_value, threshold)
  end
end

function Base.showerror(io::IO, e::PerformanceError)
  print(io, "PerformanceError: Performance constraint violated")
  print(io, "\nConstraint: $(e.constraint)")
  print(io, "\nActual: $(e.actual_value), Threshold: $(e.threshold)")
end

# ============================================================================
# External Service Exceptions
# ============================================================================

"""
    UMLSError <: GraphMERTError

Error in UMLS integration.
"""
struct UMLSError <: GraphMERTError
  operation::String
  entity::String
  original_error::Union{Exception,Nothing}

  function UMLSError(operation::String, entity::String, message::String="",
    original_error::Union{Exception,Nothing}=nothing)
    context = Dict(
      "operation" => operation,
      "entity" => entity
    )
    if original_error !== nothing
      context["original_error"] = string(original_error)
    end
    new(GraphMERTError(message, context), operation, entity, original_error)
  end
end

function Base.showerror(io::IO, e::UMLSError)
  print(io, "UMLSError: UMLS operation failed")
  print(io, "\nOperation: $(e.operation)")
  print(io, "\nEntity: $(e.entity)")
  if e.original_error !== nothing
    print(io, "\nOriginal error: $(e.original_error)")
  end
end

"""
    HelperLLMError <: GraphMERTError

Error in helper LLM integration.
"""
struct HelperLLMError <: GraphMERTError
  operation::String
  prompt::String
  original_error::Union{Exception,Nothing}

  function HelperLLMError(operation::String, prompt::String, message::String="",
    original_error::Union{Exception,Nothing}=nothing)
    context = Dict(
      "operation" => operation,
      "prompt" => prompt
    )
    if original_error !== nothing
      context["original_error"] = string(original_error)
    end
    new(GraphMERTError(message, context), operation, prompt, original_error)
  end
end

function Base.showerror(io::IO, e::HelperLLMError)
  print(io, "HelperLLMError: Helper LLM operation failed")
  print(io, "\nOperation: $(e.operation)")
  if e.original_error !== nothing
    print(io, "\nOriginal error: $(e.original_error)")
  end
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    wrap_exception(original::Exception, context::Dict{String, Any})

Wrap an original exception with GraphMERT context.
"""
function wrap_exception(original::Exception, context::Dict{String,Any})
  return GraphMERTError("Wrapped exception: $(original)", context)
end

"""
    check_performance_constraint(actual::Float64, threshold::Float64, constraint::String)

Check if a performance constraint is violated and throw appropriate error.
"""
function check_performance_constraint(actual::Float64, threshold::Float64, constraint::String)
  if actual > threshold
    throw(PerformanceError(constraint, actual, threshold,
      "Performance constraint violated: $constraint"))
  end
end

"""
    validate_confidence(confidence::Float64, context::String)

Validate confidence score and throw error if invalid.
"""
function validate_confidence(confidence::Float64, context::String)
  if !(0.0 <= confidence <= 1.0)
    throw(ValidationError("confidence", "range",
      "Confidence must be between 0.0 and 1.0, got $confidence"))
  end
end

"""
    safe_execute(f::Function, error_type::Type{<:GraphMERTError}, context::Dict{String, Any})

Safely execute a function and wrap any exceptions.
"""
function safe_execute(f::Function, error_type::Type{<:GraphMERTError}, context::Dict{String,Any})
  try
    return f()
  catch e
    if isa(e, error_type)
      rethrow(e)
    else
      throw(error_type("", "", "", e))
    end
  end
end
