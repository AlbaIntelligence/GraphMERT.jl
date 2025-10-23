"""
Attention mechanisms for GraphMERT.jl

This module implements attention decay masks and spatial distance encoding
for the GraphMERT architecture, enabling effective attention over graph structures.
"""

using LinearAlgebra
using SparseArrays

# ============================================================================
# Attention Decay Mask Types
# ============================================================================

"""
    AttentionDecayMask

Represents an attention decay mask for spatial distance encoding.
"""
struct AttentionDecayMask
    mask::Matrix{Float32}
    decay_function::Function
    max_distance::Int
    decay_rate::Float32

    function AttentionDecayMask(
        mask::Matrix{Float32},
        decay_function::Function,
        max_distance::Int,
        decay_rate::Float32,
    )
        @assert max_distance > 0 "Max distance must be positive"
        @assert decay_rate > 0.0 "Decay rate must be positive"
        @assert size(mask, 1) == size(mask, 2) "Mask must be square"

        new(mask, decay_function, max_distance, decay_rate)
    end
end

"""
    SpatialAttentionConfig

Configuration for spatial attention mechanisms.
"""
struct SpatialAttentionConfig
    max_distance::Int
    decay_rate::Float32
    decay_type::Symbol  # :exponential, :linear, :quadratic
    use_distance_bias::Bool
    distance_bias_weight::Float32

    function SpatialAttentionConfig(;
        max_distance::Int = 512,
        decay_rate::Float32 = 0.1f0,
        decay_type::Symbol = :exponential,
        use_distance_bias::Bool = true,
        distance_bias_weight::Float32 = 0.1f0,
    )
        @assert max_distance > 0 "Max distance must be positive"
        @assert decay_rate > 0.0 "Decay rate must be positive"
        @assert decay_type in [:exponential, :linear, :quadratic] "Invalid decay type: $decay_type"
        @assert distance_bias_weight >= 0.0 "Distance bias weight must be non-negative"

        new(max_distance, decay_rate, decay_type, use_distance_bias, distance_bias_weight)
    end
end

# ============================================================================
# Attention Decay Functions
# ============================================================================

"""
    exponential_decay(distance::Int, decay_rate::Float32)

Exponential decay function for attention.
"""
function exponential_decay(distance::Int, decay_rate::Float32)
    return exp(-decay_rate * distance)
end

"""
    linear_decay(distance::Int, decay_rate::Float32)

Linear decay function for attention.
"""
function linear_decay(distance::Int, decay_rate::Float32)
    return max(0.0f0, 1.0f0 - decay_rate * distance)
end

"""
    quadratic_decay(distance::Int, decay_rate::Float32)

Quadratic decay function for attention.
"""
function quadratic_decay(distance::Int, decay_rate::Float32)
    return max(0.0f0, 1.0f0 - decay_rate * distance^2)
end

"""
    get_decay_function(decay_type::Symbol)

Get the appropriate decay function for the given type.
"""
function get_decay_function(decay_type::Symbol)
    if decay_type == :exponential
        return exponential_decay
    elseif decay_type == :linear
        return linear_decay
    elseif decay_type == :quadratic
        return quadratic_decay
    else
        throw(ArgumentError("Unknown decay type: $decay_type"))
    end
end

# ============================================================================
# Attention Mask Creation
# ============================================================================

"""
    create_attention_decay_mask(seq_length::Int, config::SpatialAttentionConfig)

Create an attention decay mask for a sequence of given length.
"""
function create_attention_decay_mask(seq_length::Int, config::SpatialAttentionConfig)
    mask = zeros(Float32, seq_length, seq_length)
    decay_function = get_decay_function(config.decay_type)

    for i = 1:seq_length
        for j = 1:seq_length
            distance = abs(i - j)

            if distance <= config.max_distance
                decay_value = decay_function(distance, config.decay_rate)
                mask[i, j] = decay_value
            else
                mask[i, j] = 0.0f0
            end
        end
    end

    return AttentionDecayMask(mask, decay_function, config.max_distance, config.decay_rate)
end

"""
    create_graph_attention_mask(adjacency_matrix::SparseMatrixCSC{Float32}, config::SpatialAttentionConfig)

Create an attention mask for a graph structure.
"""
function create_graph_attention_mask(
    adjacency_matrix::SparseMatrixCSC{Float32},
    config::SpatialAttentionConfig,
)
    num_nodes = size(adjacency_matrix, 1)
    mask = zeros(Float32, num_nodes, num_nodes)
    decay_function = get_decay_function(config.decay_type)

    # Compute shortest path distances
    distances = compute_shortest_path_distances(adjacency_matrix)

    for i = 1:num_nodes
        for j = 1:num_nodes
            distance = distances[i, j]

            if distance <= config.max_distance
                decay_value = decay_function(distance, config.decay_rate)
                mask[i, j] = decay_value
            else
                mask[i, j] = 0.0f0
            end
        end
    end

    return AttentionDecayMask(mask, decay_function, config.max_distance, config.decay_rate)
end

"""
    create_hierarchical_attention_mask(adjacency_matrices::Vector{SparseMatrixCSC{Float32}}, config::SpatialAttentionConfig)

Create a hierarchical attention mask for multi-level graph structures.
"""
function create_hierarchical_attention_mask(
    adjacency_matrices::Vector{SparseMatrixCSC{Float32}},
    config::SpatialAttentionConfig,
)
    num_levels = length(adjacency_matrices)
    num_nodes = size(adjacency_matrices[1], 1)

    # Initialize mask with zeros
    mask = zeros(Float32, num_nodes, num_nodes)
    decay_function = get_decay_function(config.decay_type)

    # Combine attention from all levels
    for level = 1:num_levels
        level_mask = create_graph_attention_mask(adjacency_matrices[level], config)
        level_weight = 1.0f0 / num_levels  # Equal weight for each level

        mask .+= level_weight .* level_mask.mask
    end

    # Normalize the combined mask
    mask = mask ./ maximum(mask)

    return AttentionDecayMask(mask, decay_function, config.max_distance, config.decay_rate)
end

# ============================================================================
# Distance Computation
# ============================================================================

"""
    compute_shortest_path_distances(adjacency_matrix::SparseMatrixCSC{Float32})

Compute shortest path distances between all pairs of nodes.
"""
function compute_shortest_path_distances(adjacency_matrix::SparseMatrixCSC{Float32})
    num_nodes = size(adjacency_matrix, 1)
    distances = fill(Inf32, num_nodes, num_nodes)

    # Initialize distances
    for i = 1:num_nodes
        distances[i, i] = 0.0f0
        for j = 1:num_nodes
            if adjacency_matrix[i, j] > 0
                distances[i, j] = 1.0f0
            end
        end
    end

    # Floyd-Warshall algorithm
    for k = 1:num_nodes
        for i = 1:num_nodes
            for j = 1:num_nodes
                if distances[i, k] + distances[k, j] < distances[i, j]
                    distances[i, j] = distances[i, k] + distances[k, j]
                end
            end
        end
    end

    # Replace Inf with max_distance + 1 for unreachable nodes
    max_distance = maximum(distances[distances .< Inf])
    distances[distances .== Inf] = max_distance + 1

    return distances
end

"""
    compute_attention_distances(attention_scores::Matrix{Float32}, mask::AttentionDecayMask)

Compute attention distances using the decay mask.
"""
function compute_attention_distances(
    attention_scores::Matrix{Float32},
    mask::AttentionDecayMask,
)
    # Apply the decay mask to attention scores
    masked_scores = attention_scores .* mask.mask

    # Normalize the masked scores
    row_sums = sum(masked_scores, dims = 2)
    row_sums[row_sums .== 0] .= 1.0f0  # Avoid division by zero
    normalized_scores = masked_scores ./ row_sums

    return normalized_scores
end

# ============================================================================
# Attention Mechanisms
# ============================================================================

"""
    apply_attention_decay(attention_scores::Matrix{Float32}, mask::AttentionDecayMask)

Apply attention decay mask to attention scores.
"""
function apply_attention_decay(attention_scores::Matrix{Float32}, mask::AttentionDecayMask)
    return attention_scores .* mask.mask
end

"""
    apply_spatial_bias(attention_scores::Matrix{Float32}, positions::Vector{Int}, config::SpatialAttentionConfig)

Apply spatial bias to attention scores based on positions.
"""
function apply_spatial_bias(
    attention_scores::Matrix{Float32},
    positions::Vector{Int},
    config::SpatialAttentionConfig,
)
    if !config.use_distance_bias
        return attention_scores
    end

    biased_scores = copy(attention_scores)
    seq_length = size(attention_scores, 1)

    for i = 1:seq_length
        for j = 1:seq_length
            distance = abs(positions[i] - positions[j])
            bias = -config.distance_bias_weight * distance
            biased_scores[i, j] += bias
        end
    end

    return biased_scores
end

"""
    create_causal_attention_mask(seq_length::Int)

Create a causal attention mask for autoregressive models.
"""
function create_causal_attention_mask(seq_length::Int)
    mask = zeros(Float32, seq_length, seq_length)

    for i = 1:seq_length
        for j = 1:seq_length
            if j <= i
                mask[i, j] = 1.0f0
            else
                mask[i, j] = 0.0f0
            end
        end
    end

    return mask
end

"""
    create_padding_attention_mask(attention_mask::Vector{Bool})

Create an attention mask for padding tokens.
"""
function create_padding_attention_mask(attention_mask::Vector{Bool})
    seq_length = length(attention_mask)
    mask = zeros(Float32, seq_length, seq_length)

    for i = 1:seq_length
        for j = 1:seq_length
            if attention_mask[i] && attention_mask[j]
                mask[i, j] = 1.0f0
            else
                mask[i, j] = 0.0f0
            end
        end
    end

    return mask
end

# ============================================================================
# Multi-Head Attention with Decay
# ============================================================================

"""
    MultiHeadAttentionWithDecay

Multi-head attention mechanism with decay mask support.
"""
struct MultiHeadAttentionWithDecay
    query_projection::Dense
    key_projection::Dense
    value_projection::Dense
    output_projection::Dense
    num_heads::Int
    head_dim::Int
    dropout::Dropout

    function MultiHeadAttentionWithDecay(
        input_dim::Int,
        num_heads::Int,
        head_dim::Int,
        dropout_rate::Float32,
    )
        @assert input_dim % num_heads == 0 "Input dimension must be divisible by number of heads"

        query_projection = Dense(input_dim, input_dim)
        key_projection = Dense(input_dim, input_dim)
        value_projection = Dense(input_dim, input_dim)
        output_projection = Dense(input_dim, input_dim)
        dropout = Dropout(dropout_rate)

        new(
            query_projection,
            key_projection,
            value_projection,
            output_projection,
            num_heads,
            head_dim,
            dropout,
        )
    end
end

"""
    (attention::MultiHeadAttentionWithDecay)(query::Matrix{Float32}, key::Matrix{Float32}, value::Matrix{Float32},
                                           decay_mask::AttentionDecayMask)

Forward pass for multi-head attention with decay mask.
"""
function (attention::MultiHeadAttentionWithDecay)(
    query::Matrix{Float32},
    key::Matrix{Float32},
    value::Matrix{Float32},
    decay_mask::AttentionDecayMask,
)
    batch_size, seq_length, _ = size(query)

    # Project to query, key, value
    Q = attention.query_projection(query)
    K = attention.key_projection(key)
    V = attention.value_projection(value)

    # Reshape for multi-head attention
    Q = reshape(Q, batch_size, seq_length, attention.num_heads, attention.head_dim)
    K = reshape(K, batch_size, seq_length, attention.num_heads, attention.head_dim)
    V = reshape(V, batch_size, seq_length, attention.num_heads, attention.head_dim)

    # Transpose for attention computation
    Q = permutedims(Q, [1, 3, 2, 4])  # [batch, heads, seq, head_dim]
    K = permutedims(K, [1, 3, 2, 4])
    V = permutedims(V, [1, 3, 2, 4])

    # Compute attention scores
    attention_scores = Q * transpose(K, 3, 4)  # [batch, heads, seq, seq]
    attention_scores = attention_scores / sqrt(attention.head_dim)

    # Apply decay mask
    attention_scores = apply_attention_decay(attention_scores, decay_mask)

    # Softmax
    attention_probs = softmax(attention_scores, dims = 4)
    attention_probs = attention.dropout(attention_probs)

    # Apply attention to values
    context = attention_probs * V  # [batch, heads, seq, head_dim]

    # Reshape back
    context = permutedims(context, [1, 3, 2, 4])  # [batch, seq, heads, head_dim]
    context =
        reshape(context, batch_size, seq_length, attention.num_heads * attention.head_dim)

    # Output projection
    output = attention.output_projection(context)

    return output
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    visualize_attention_mask(mask::AttentionDecayMask)

Create a visualization of the attention mask.
"""
function visualize_attention_mask(mask::AttentionDecayMask)
    # This would create a heatmap visualization
    # For now, return basic statistics
    return Dict{String,Any}(
        "shape" => size(mask.mask),
        "min_value" => minimum(mask.mask),
        "max_value" => maximum(mask.mask),
        "mean_value" => mean(mask.mask),
        "non_zero_ratio" => count(x -> x > 0, mask.mask) / length(mask.mask),
    )
end

"""
    get_attention_statistics(mask::AttentionDecayMask)

Get statistics about the attention mask.
"""
function get_attention_statistics(mask::AttentionDecayMask)
    return Dict{String,Any}(
        "total_elements" => length(mask.mask),
        "non_zero_elements" => count(x -> x > 0, mask.mask),
        "zero_elements" => count(x -> x == 0, mask.mask),
        "min_value" => minimum(mask.mask),
        "max_value" => maximum(mask.mask),
        "mean_value" => mean(mask.mask),
        "std_value" => std(mask.mask),
    )
end
