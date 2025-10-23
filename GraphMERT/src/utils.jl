"""
Utility functions for GraphMERT.jl

This module provides utility functions used throughout
the GraphMERT implementation.
"""

using Logging
using Statistics
using Random

# ============================================================================
# Text Processing Utilities
# ============================================================================

"""
    preprocess_text(text::String; max_length::Int=512)

Preprocess text for GraphMERT processing.
"""
function preprocess_text(text::String; max_length::Int=512)
  # Remove extra whitespace
  text = strip(text)

  # Truncate if too long
  if length(text) > max_length
    text = text[1:max_length]
    @warn "Text truncated to $max_length characters"
  end

  return text
end

"""
    tokenize_text(text::String)

Simple tokenization for text processing.
"""
function tokenize_text(text::String)
  # Simple word tokenization
  words = split(text, r"\s+")
  return filter(word -> !isempty(word), words)
end

"""
    normalize_text(text::String)

Normalize text for consistent processing.
"""
function normalize_text(text::String)
  # Convert to lowercase
  text = lowercase(text)

  # Remove special characters but keep alphanumeric and spaces
  text = replace(text, r"[^\w\s]" => " ")

  # Remove extra whitespace
  text = replace(text, r"\s+" => " ")

  return strip(text)
end

# ============================================================================
# Mathematical Utilities
# ============================================================================

"""
    softmax(x::Vector{Float64})

Compute softmax function.
"""
function softmax(x::Vector{Float64})
  exp_x = exp.(x .- maximum(x))
  return exp_x ./ sum(exp_x)
end

"""
    sigmoid(x::Float64)

Compute sigmoid function.
"""
function sigmoid(x::Float64)
  return 1.0 / (1.0 + exp(-x))
end

"""
    relu(x::Float64)

Compute ReLU function.
"""
function relu(x::Float64)
  return max(0.0, x)
end

"""
    cosine_similarity(a::Vector{Float64}, b::Vector{Float64})

Compute cosine similarity between two vectors.
"""
function cosine_similarity(a::Vector{Float64}, b::Vector{Float64})
  if length(a) != length(b)
    error("Vectors must have the same length")
  end

  dot_product = sum(a .* b)
  norm_a = sqrt(sum(a .^ 2))
  norm_b = sqrt(sum(b .^ 2))

  if norm_a == 0.0 || norm_b == 0.0
    return 0.0
  end

  return dot_product / (norm_a * norm_b)
end

# ============================================================================
# Data Utilities
# ============================================================================

"""
    shuffle_data(data::Vector{T}) where T

Shuffle data randomly.
"""
function shuffle_data(data::Vector{T}) where {T}
  indices = collect(1:length(data))
  shuffle!(indices)
  return data[indices]
end

"""
    split_data(data::Vector{T}, train_ratio::Float64=0.8) where T

Split data into train and test sets.
"""
function split_data(data::Vector{T}, train_ratio::Float64=0.8) where {T}
  @assert 0.0 < train_ratio < 1.0 "train_ratio must be between 0.0 and 1.0"

  shuffled_data = shuffle_data(data)
  split_index = round(Int, length(data) * train_ratio)

  train_data = shuffled_data[1:split_index]
  test_data = shuffled_data[(split_index+1):end]

  return train_data, test_data
end

"""
    batch_data(data::Vector{T}, batch_size::Int) where T

Split data into batches.
"""
function batch_data(data::Vector{T}, batch_size::Int) where {T}
  batches = Vector{Vector{T}}()

  for i = 1:batch_size:length(data)
    end_idx = min(i + batch_size - 1, length(data))
    push!(batches, data[i:end_idx])
  end

  return batches
end

# ============================================================================
# Logging Utilities
# ============================================================================

"""
    log_info(message::String; context::String="")

Log an info message with optional context.
"""
function log_info(message::String; context::String="")
  if !isempty(context)
    @info "[$context] $message"
  else
    @info message
  end
end

"""
    log_warning(message::String; context::String="")

Log a warning message with optional context.
"""
function log_warning(message::String; context::String="")
  if !isempty(context)
    @warn "[$context] $message"
  else
    @warn message
  end
end

"""
    log_error(message::String; context::String="")

Log an error message with optional context.
"""
function log_error(message::String; context::String="")
  if !isempty(context)
    @error "[$context] $message"
  else
    @error message
  end
end

# ============================================================================
# Validation Utilities
# ============================================================================

"""
    validate_not_empty(value::Any, name::String)

Validate that a value is not empty.
"""
function validate_not_empty(value::Any, name::String)
  if isempty(value)
    error("$name cannot be empty")
  end
end

"""
    validate_positive(value::Number, name::String)

Validate that a numeric value is positive.
"""
function validate_positive(value::Number, name::String)
  if value <= 0
    error("$name must be positive, got: $value")
  end
end

"""
    validate_range(value::Number, min_val::Number, max_val::Number, name::String)

Validate that a numeric value is within a range.
"""
function validate_range(value::Number, min_val::Number, max_val::Number, name::String)
  if value < min_val || value > max_val
    error("$name must be between $min_val and $max_val, got: $value")
  end
end

# ============================================================================
# File Utilities
# ============================================================================

"""
    ensure_directory(path::String)

Ensure that a directory exists.
"""
function ensure_directory(path::String)
  if !isdir(path)
    mkpath(path)
    @info "Created directory: $path"
  end
end

"""
    safe_filename(filename::String)

Make a filename safe for filesystem use.
"""
function safe_filename(filename::String)
  # Replace invalid characters with underscores
  safe_name = replace(filename, r"[^\w\-_\.]" => "_")

  # Remove multiple consecutive underscores
  safe_name = replace(safe_name, r"_+" => "_")

  # Remove leading/trailing underscores
  safe_name = strip(safe_name, '_')

  return safe_name
end

# ============================================================================
# Performance Utilities
# ============================================================================

"""
    time_function(f::Function, args...)

Time the execution of a function.
"""
function time_function(f::Function, args...)
  start_time = time()
  result = f(args...)
  end_time = time()

  execution_time = end_time - start_time
  return result, execution_time
end

"""
    memory_usage()

Get current memory usage in MB.
"""
function memory_usage()
  # This is a simplified version - in practice you'd use proper memory profiling
  return 100.0  # Placeholder
end

# ============================================================================
# String Utilities
# ============================================================================

"""
    truncate_string(s::String, max_length::Int; suffix::String="...")

Truncate a string to a maximum length.
"""
function truncate_string(s::String, max_length::Int; suffix::String="...")
  if length(s) <= max_length
    return s
  else
    return s[1:(max_length-length(suffix))] * suffix
  end
end

"""
    join_strings(strings::Vector{String}, separator::String=" ")

Join strings with a separator.
"""
function join_strings(strings::Vector{String}, separator::String=" ")
  return join(strings, separator)
end

# ============================================================================
# Type Utilities
# ============================================================================

"""
    is_type_of(value::Any, expected_type::Type)

Check if a value is of the expected type.
"""
function is_type_of(value::Any, expected_type::Type)
  return isa(value, expected_type)
end

"""
    convert_to_type(value::Any, target_type::Type)

Convert a value to the target type if possible.
"""
function convert_to_type(value::Any, target_type::Type)
  try
    return convert(target_type, value)
  catch e
    error("Cannot convert $value to $target_type: $e")
  end
end

# ============================================================================
# Knowledge Graph Utilities
# ============================================================================

# merge_knowledge_graphs is already implemented in GraphMERT/src/api/batch.jl

"""
    filter_knowledge_graph(kg::KnowledgeGraph;
                          min_confidence::Float64=0.0,
                          entity_types::Vector{String}=String[],
                          relation_types::Vector{String}=String[]) -> KnowledgeGraph

Filter knowledge graph based on confidence thresholds and types.
"""
function filter_knowledge_graph(kg::KnowledgeGraph;
  min_confidence::Float64=0.0,
  entity_types::Vector{String}=String[],
  relation_types::Vector{String}=String[])

  # Filter entities
  filtered_entities = filter(kg.entities) do entity
    # Confidence filter
    if entity.confidence < min_confidence
      return false
    end

    # Entity type filter
    if !isempty(entity_types) && !(entity.label in entity_types)
      return false
    end

    return true
  end

  # Filter relations
  filtered_relations = filter(kg.relations) do relation
    # Confidence filter
    if relation.confidence < min_confidence
      return false
    end

    # Relation type filter
    if !isempty(relation_types) && !(relation.relation_type in relation_types)
      return false
    end

    # Ensure both head and tail entities are still present
    head_present = any(e.text == relation.head for e in filtered_entities)
    tail_present = any(e.text == relation.tail for e in filtered_entities)

    return head_present && tail_present
  end

  # Create filtered metadata
  filtered_metadata = copy(kg.metadata)
  filtered_metadata["filtered"] = true
  filtered_metadata["filter_time"] = now()
  filtered_metadata["min_confidence"] = min_confidence
  filtered_metadata["entity_types"] = entity_types
  filtered_metadata["relation_types"] = relation_types
  filtered_metadata["original_entities"] = length(kg.entities)
  filtered_metadata["original_relations"] = length(kg.relations)
  filtered_metadata["filtered_entities"] = length(filtered_entities)
  filtered_metadata["filtered_relations"] = length(filtered_relations)

  return KnowledgeGraph(filtered_entities, filtered_relations, filtered_metadata, now())
end
