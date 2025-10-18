"""
Utility functions for GraphMERT.jl

This module provides utility functions for validation, serialization, performance
monitoring, and other common operations throughout the GraphMERT implementation.
"""

using Dates
using JSON
using Statistics
using Random
using SparseArrays

# ============================================================================
# Validation Utilities
# ============================================================================

"""
    validate_confidence(confidence::Float64, context::String = "")

Validate that a confidence score is within valid range.
"""
function validate_confidence(confidence::Float64, context::String="")
  if !(0.0 <= confidence <= 1.0)
    error_msg = "Confidence must be between 0.0 and 1.0, got $confidence"
    if !isempty(context)
      error_msg *= " ($context)"
    end
    throw(ArgumentError(error_msg))
  end
  return true
end

"""
    validate_text(text::String, min_length::Int = 1, max_length::Int = 1000000)

Validate that text meets length requirements.
"""
function validate_text(text::String, min_length::Int=1, max_length::Int=1000000)
  if length(text) < min_length
    throw(ArgumentError("Text too short: $(length(text)) < $min_length"))
  end
  if length(text) > max_length
    throw(ArgumentError("Text too long: $(length(text)) > $max_length"))
  end
  return true
end

"""
    validate_entity_id(id::String)

Validate that an entity ID is valid.
"""
function validate_entity_id(id::String)
  if isempty(id)
    throw(ArgumentError("Entity ID cannot be empty"))
  end
  if !isvalid(id)
    throw(ArgumentError("Entity ID contains invalid characters: $id"))
  end
  return true
end

"""
    validate_knowledge_graph_structure(graph::KnowledgeGraph)

Validate the structure of a knowledge graph.
"""
function validate_knowledge_graph_structure(graph::KnowledgeGraph)
  # Check basic structure
  if isempty(graph.entities)
    throw(ArgumentError("Knowledge graph must contain at least one entity"))
  end

  # Check confidence threshold
  validate_confidence(graph.confidence_threshold, "confidence_threshold")

  # Check that all relations reference valid entities
  entity_ids = Set(e.id for e in graph.entities)
  for relation in graph.relations
    if !(relation.head in entity_ids)
      throw(ArgumentError("Relation head '$(relation.head)' not found in entities"))
    end
    if !(relation.tail in entity_ids)
      throw(ArgumentError("Relation tail '$(relation.tail)' not found in entities"))
    end
  end

  return true
end

# ============================================================================
# Serialization Utilities
# ============================================================================

"""
    serialize_knowledge_graph(graph::KnowledgeGraph, format::Symbol = :json)

Serialize a knowledge graph to the specified format.
"""
function serialize_knowledge_graph(graph::KnowledgeGraph, format::Symbol=:json)
  if format == :json
    return serialize_to_json(graph)
  elseif format == :binary
    return serialize_to_binary(graph)
  else
    throw(ArgumentError("Unsupported format: $format"))
  end
end

"""
    serialize_to_json(graph::KnowledgeGraph)

Serialize a knowledge graph to JSON format.
"""
function serialize_to_json(graph::KnowledgeGraph)
  graph_dict = Dict{String,Any}(
    "entities" => [entity_to_dict(e) for e in graph.entities],
    "relations" => [relation_to_dict(r) for r in graph.relations],
    "metadata" => metadata_to_dict(graph.metadata),
    "confidence_threshold" => graph.confidence_threshold,
    "created_at" => Dates.format(graph.created_at, "yyyy-mm-ddTHH:MM:SS.sssZ"),
    "model_info" => model_info_to_dict(graph.model_info),
    "umls_mappings" => graph.umls_mappings,
    "fact_score" => graph.fact_score,
    "validity_score" => graph.validity_score
  )

  return JSON.json(graph_dict, 2)
end

"""
    serialize_to_binary(graph::KnowledgeGraph)

Serialize a knowledge graph to binary format.
"""
function serialize_to_binary(graph::KnowledgeGraph)
  # This would use Julia's serialization capabilities
  # For now, we'll use JSON as a placeholder
  return serialize_to_json(graph)
end

"""
    deserialize_knowledge_graph(data::String, format::Symbol = :json)

Deserialize a knowledge graph from the specified format.
"""
function deserialize_knowledge_graph(data::String, format::Symbol=:json)
  if format == :json
    return deserialize_from_json(data)
  elseif format == :binary
    return deserialize_from_binary(data)
  else
    throw(ArgumentError("Unsupported format: $format"))
  end
end

"""
    deserialize_from_json(data::String)

Deserialize a knowledge graph from JSON format.
"""
function deserialize_from_json(data::String)
  graph_dict = JSON.parse(data)

  entities = [dict_to_entity(e) for e in graph_dict["entities"]]
  relations = [dict_to_relation(r) for r in graph_dict["relations"]]
  metadata = dict_to_metadata(graph_dict["metadata"])
  model_info = dict_to_model_info(graph_dict["model_info"])

  return KnowledgeGraph(
    entities, relations, graph_dict["confidence_threshold"],
    model_info, graph_dict["umls_mappings"],
    graph_dict["fact_score"], graph_dict["validity_score"]
  )
end

# ============================================================================
# Dictionary Conversion Utilities
# ============================================================================

"""
    entity_to_dict(entity::BiomedicalEntity)

Convert an entity to dictionary format.
"""
function entity_to_dict(entity::BiomedicalEntity)
  return Dict{String,Any}(
    "id" => entity.id,
    "text" => entity.text,
    "label" => entity.label,
    "confidence" => entity.confidence,
    "position" => Dict{String,Any}(
      "start" => entity.position.start,
      "stop" => entity.position.stop,
      "line" => entity.position.line,
      "column" => entity.position.column
    ),
    "attributes" => entity.attributes,
    "created_at" => Dates.format(entity.created_at, "yyyy-mm-ddTHH:MM:SS.sssZ")
  )
end

"""
    relation_to_dict(relation::BiomedicalRelation)

Convert a relation to dictionary format.
"""
function relation_to_dict(relation::BiomedicalRelation)
  return Dict{String,Any}(
    "head" => relation.head,
    "tail" => relation.tail,
    "relation_type" => relation.relation_type,
    "confidence" => relation.confidence,
    "attributes" => relation.attributes,
    "created_at" => Dates.format(relation.created_at, "yyyy-mm-ddTHH:MM:SS.sssZ")
  )
end

"""
    metadata_to_dict(metadata::GraphMetadata)

Convert metadata to dictionary format.
"""
function metadata_to_dict(metadata::GraphMetadata)
  return Dict{String,Any}(
    "total_entities" => metadata.total_entities,
    "total_relations" => metadata.total_relations,
    "entity_types" => metadata.entity_types,
    "relation_types" => metadata.relation_types,
    "average_confidence" => metadata.average_confidence,
    "processing_time" => metadata.processing_time,
    "model_version" => metadata.model_version,
    "created_at" => Dates.format(metadata.created_at, "yyyy-mm-ddTHH:MM:SS.sssZ")
  )
end

"""
    model_info_to_dict(model_info::GraphMERTModelInfo)

Convert model info to dictionary format.
"""
function model_info_to_dict(model_info::GraphMERTModelInfo)
  return Dict{String,Any}(
    "model_name" => model_info.model_name,
    "model_version" => model_info.model_version,
    "architecture" => model_info.architecture,
    "parameters" => model_info.parameters,
    "training_data" => model_info.training_data,
    "created_at" => Dates.format(model_info.created_at, "yyyy-mm-ddTHH:MM:SS.sssZ")
  )
end

# ============================================================================
# Performance Monitoring Utilities
# ============================================================================

"""
    measure_execution_time(f::Function, args...)

Measure the execution time of a function.
"""
function measure_execution_time(f::Function, args...)
  start_time = time()
  result = f(args...)
  end_time = time()
  return result, end_time - start_time
end

"""
    measure_memory_usage(f::Function, args...)

Measure the memory usage of a function.
"""
function measure_memory_usage(f::Function, args...)
  # This is a simplified implementation
  # In practice, you'd use more sophisticated memory profiling
  start_memory = get_memory_usage()
  result = f(args...)
  end_memory = get_memory_usage()
  return result, end_memory - start_memory
end

"""
    get_memory_usage()

Get current memory usage in bytes.
"""
function get_memory_usage()
  # This is a simplified implementation
  # In practice, you'd use system-specific memory monitoring
  return 0
end

"""
    get_processing_speed(tokens::Int, duration::Float64)

Calculate processing speed in tokens per second.
"""
function get_processing_speed(tokens::Int, duration::Float64)
  if duration <= 0
    return 0.0
  end
  return tokens / duration
end

"""
    check_performance_constraint(actual::Float64, threshold::Float64, constraint::String)

Check if a performance constraint is violated.
"""
function check_performance_constraint(actual::Float64, threshold::Float64, constraint::String)
  if actual > threshold
    throw(ArgumentError("Performance constraint violated: $constraint (actual: $actual, threshold: $threshold)"))
  end
  return true
end

# ============================================================================
# Text Processing Utilities
# ============================================================================

"""
    preprocess_text(text::String, options::ProcessingOptions)

Preprocess text for GraphMERT processing.
"""
function preprocess_text(text::String, options::ProcessingOptions)
  # Basic text preprocessing
  text = strip(text)
  text = replace(text, r"\s+" => " ")  # Normalize whitespace

  # Validate text length
  validate_text(text, 1, options.memory_limit * 1000)  # Rough character limit

  return text
end

"""
    tokenize_text(text::String, max_length::Int = 512)

Tokenize text for processing.
"""
function tokenize_text(text::String, max_length::Int=512)
  # Simple tokenization - in practice, you'd use a proper tokenizer
  tokens = split(text, " ")

  if length(tokens) > max_length
    tokens = tokens[1:max_length]
  end

  return tokens
end

"""
    calculate_text_statistics(text::String)

Calculate basic text statistics.
"""
function calculate_text_statistics(text::String)
  return Dict{String,Any}(
    "length" => length(text),
    "words" => length(split(text, " ")),
    "sentences" => length(split(text, r"[.!?]+")),
    "characters" => length(text),
    "whitespace" => length(collect(eachmatch(r"\s", text)))
  )
end

# ============================================================================
# Random Utilities
# ============================================================================

"""
    set_random_seed(seed::Int)

Set the random seed for reproducible results.
"""
function set_random_seed(seed::Int)
  Random.seed!(seed)
  return nothing
end

"""
    generate_random_id(prefix::String = "entity", length::Int = 8)

Generate a random ID with the specified prefix and length.
"""
function generate_random_id(prefix::String="entity", length::Int=8)
  random_part = join(rand('a':'z', length))
  return "$(prefix)_$(random_part)"
end

# ============================================================================
# File Utilities
# ============================================================================

"""
    ensure_directory_exists(path::String)

Ensure that a directory exists, creating it if necessary.
"""
function ensure_directory_exists(path::String)
  if !isdir(path)
    mkpath(path)
  end
  return path
end

"""
    safe_file_write(filename::String, data::String)

Safely write data to a file.
"""
function safe_file_write(filename::String, data::String)
  try
    open(filename, "w") do io
      write(io, data)
    end
    return true
  catch e
    throw(ArgumentError("Failed to write file $filename: $e"))
  end
end

"""
    safe_file_read(filename::String)

Safely read data from a file.
"""
function safe_file_read(filename::String)
  if !isfile(filename)
    throw(ArgumentError("File not found: $filename"))
  end

  try
    return read(filename, String)
  catch e
    throw(ArgumentError("Failed to read file $filename: $e"))
  end
end

# ============================================================================
# Validation Utilities
# ============================================================================

"""
    validate_file_exists(filename::String)

Validate that a file exists.
"""
function validate_file_exists(filename::String)
  if !isfile(filename)
    throw(ArgumentError("File not found: $filename"))
  end
  return true
end

"""
    validate_directory_exists(dirname::String)

Validate that a directory exists.
"""
function validate_directory_exists(dirname::String)
  if !isdir(dirname)
    throw(ArgumentError("Directory not found: $dirname"))
  end
  return true
end

# ============================================================================
# Statistics Utilities
# ============================================================================

"""
    calculate_statistics(values::Vector{Float64})

Calculate basic statistics for a vector of values.
"""
function calculate_statistics(values::Vector{Float64})
  if isempty(values)
    return Dict{String,Any}()
  end

  return Dict{String,Any}(
    "count" => length(values),
    "mean" => mean(values),
    "median" => median(values),
    "std" => std(values),
    "min" => minimum(values),
    "max" => maximum(values),
    "sum" => sum(values)
  )
end

"""
    calculate_percentiles(values::Vector{Float64}, percentiles::Vector{Float64} = [25.0, 50.0, 75.0, 90.0, 95.0, 99.0])

Calculate percentiles for a vector of values.
"""
function calculate_percentiles(values::Vector{Float64}, percentiles::Vector{Float64}=[25.0, 50.0, 75.0, 90.0, 95.0, 99.0])
  if isempty(values)
    return Dict{String,Float64}()
  end

  result = Dict{String,Float64}()
  for p in percentiles
    result["p$(p)"] = quantile(values, p / 100.0)
  end

  return result
end
