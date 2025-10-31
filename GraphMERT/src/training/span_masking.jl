"""
Span masking utilities for GraphMERT.jl

This module implements span masking strategies for training
the GraphMERT model with masked language modeling objectives.
"""

using Random
# using DocStringExtensions  # Temporarily disabled

# ============================================================================
# Span Masking Configuration
# ============================================================================

"""
    SpanMaskingConfig

Configuration for span masking operations.

"""
struct SpanMaskingConfig
  min_span_length::Int
  max_span_length::Int
  masking_probability::Float64
  random_seed::Union{Int,Nothing}

  function SpanMaskingConfig(;
    min_span_length::Int=1,
    max_span_length::Int=5,
    masking_probability::Float64=0.15,
    random_seed::Union{Int,Nothing}=nothing
  )
    @assert min_span_length > 0 "Minimum span length must be positive"
    @assert max_span_length >= min_span_length "Maximum span length must be >= minimum"
    @assert 0.0 <= masking_probability <= 1.0 "Masking probability must be between 0 and 1"

    new(min_span_length, max_span_length, masking_probability, random_seed)
  end
end

# ============================================================================
# Span Masking Functions
# ============================================================================

"""
    create_span_masks(sequence_length::Int, config::SpanMaskingConfig)

Create span masks for a sequence of given length.

"""
function create_span_masks(sequence_length::Int, config::SpanMaskingConfig)
  if config.random_seed !== nothing
    Random.seed!(config.random_seed)
  end

  masks = falses(sequence_length)
  num_masked = 0
  target_masked = Int(round(sequence_length * config.masking_probability))

  while num_masked < target_masked
    # Choose random span length
    span_length = rand(config.min_span_length:config.max_span_length)

    # Choose random start position
    max_start = sequence_length - span_length + 1
    if max_start <= 0
      break
    end

    start_pos = rand(1:max_start)

    # Check if span overlaps with existing masks
    if !any(masks[start_pos:start_pos+span_length-1])
      masks[start_pos:start_pos+span_length-1] .= true
      num_masked += span_length
    end
  end

  return masks
end

"""
    apply_span_masking(tokens::Vector{Int}, masks::BitVector, mask_token_id::Int)

Apply span masking to tokens using the provided masks.

"""
function apply_span_masking(tokens::Vector{Int}, masks::BitVector, mask_token_id::Int)
  @assert length(tokens) == length(masks) "Tokens and masks must have same length"

  masked_tokens = copy(tokens)
  masked_tokens[masks] .= mask_token_id

  return masked_tokens
end

"""
    get_span_boundaries(masks::BitVector)

Get the boundaries of masked spans.

"""
function get_span_boundaries(masks::BitVector)
  boundaries = Vector{Tuple{Int,Int}}()

  i = 1
  while i <= length(masks)
    if masks[i]
      start_pos = i
      # Find end of span
      while i <= length(masks) && masks[i]
        i += 1
      end
      end_pos = i - 1
      push!(boundaries, (start_pos, end_pos))
    else
      i += 1
    end
  end

  return boundaries
end

"""
    default_span_masking_config()

Create default span masking configuration.

"""
function default_span_masking_config()
  return SpanMaskingConfig()
end
