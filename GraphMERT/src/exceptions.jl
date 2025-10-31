"""
Exception definitions for GraphMERT.jl

This module defines custom exceptions used throughout
the GraphMERT implementation.
"""

# using DocStringExtensions  # Temporarily disabled

# ============================================================================
# Base Exceptions
# ============================================================================

"""
    GraphMERTException <: Exception

Base exception for all GraphMERT-related errors.

"""
struct GraphMERTException <: Exception
  message::String
  cause::Union{Exception,Nothing}

  function GraphMERTException(message::String, cause::Union{Exception,Nothing}=nothing)
    new(message, cause)
  end
end

Base.showerror(io::IO, e::GraphMERTException) =
  print(io, "GraphMERTException: $(e.message)")

# ============================================================================
# Model Exceptions
# ============================================================================

"""
    ModelLoadException <: Exception

Exception raised when model loading fails.

"""
struct ModelLoadException <: Exception
  model_path::String
  cause::Union{Exception,Nothing}

  function ModelLoadException(
    model_path::String,
    cause::Union{Exception,Nothing}=nothing,
  )
    new(model_path, cause)
  end
end

Base.showerror(io::IO, e::ModelLoadException) =
  print(io, "ModelLoadException: Failed to load model from: $(e.model_path)")

"""
    ModelSaveException <: Exception

Exception raised when model saving fails.

"""
struct ModelSaveException <: Exception
  model_path::String
  cause::Union{Exception,Nothing}

  function ModelSaveException(
    model_path::String,
    cause::Union{Exception,Nothing}=nothing,
  )
    new(model_path, cause)
  end
end

Base.showerror(io::IO, e::ModelSaveException) =
  print(io, "ModelSaveException: Failed to save model to: $(e.model_path)")

# ============================================================================
# Data Exceptions
# ============================================================================

"""
    DataLoadException <: Exception

Exception raised when data loading fails.

"""
struct DataLoadException <: Exception
  data_path::String
  cause::Union{Exception,Nothing}

  function DataLoadException(data_path::String, cause::Union{Exception,Nothing}=nothing)
    new(data_path, cause)
  end
end

Base.showerror(io::IO, e::DataLoadException) =
  print(io, "DataLoadException: Failed to load data from: $(e.data_path)")

# ============================================================================
# API Exceptions
# ============================================================================

"""
    APIException <: Exception

Exception raised when API calls fail.

"""
struct APIException <: Exception
  api_name::String
  status_code::Int
  cause::Union{Exception,Nothing}

  function APIException(
    api_name::String,
    status_code::Int,
    cause::Union{Exception,Nothing}=nothing,
  )
    new(api_name, status_code, cause)
  end
end

Base.showerror(io::IO, e::APIException) = print(
  io,
  "APIException: API call to '$(e.api_name)' failed with status code: $(e.status_code)",
)

# ============================================================================
# Processing Exceptions
# ============================================================================

"""
    TextProcessingException <: Exception

Exception raised when text processing fails.

"""
struct TextProcessingException <: Exception
  text::String
  cause::Union{Exception,Nothing}

  function TextProcessingException(
    text::String,
    cause::Union{Exception,Nothing}=nothing,
  )
    new(text, cause)
  end
end

Base.showerror(io::IO, e::TextProcessingException) = print(
  io,
  "TextProcessingException: Text processing failed for: $(e.text[1:min(100, length(e.text))])...",
)

# ============================================================================
# Utility Functions
# ============================================================================

"""
    wrap_exception(original::Exception, message::String)

Wrap an original exception with a custom message.

"""
function wrap_exception(original::Exception, message::String)
  return GraphMERTException(message, original)
end

"""
    check_condition(condition::Bool, exception_type::Type{<:Exception}, args...)

Check a condition and throw an exception if it fails.

"""
function check_condition(condition::Bool, exception_type::Type{<:Exception}, args...)
  if !condition
    throw(exception_type(args...))
  end
end
