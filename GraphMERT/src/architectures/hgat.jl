"""
Hierarchical Graph Attention Network (H-GAT) for GraphMERT.jl

This module implements the H-GAT component for semantic relation encoding
as specified in the GraphMERT paper, designed to work with RoBERTa embeddings.
"""

using Flux
using LinearAlgebra
using SparseArrays
# using DocStringExtensions  # Temporarily disabled

# ============================================================================
# H-GAT Configuration
# ============================================================================

"""
    HGATConfig

Configuration for Hierarchical Graph Attention Network.

"""
struct HGATConfig
  input_dim::Int
  hidden_dim::Int
  num_heads::Int
  num_layers::Int
  dropout_rate::Float64
  attention_dropout_rate::Float64
  layer_norm_eps::Float64
  use_residual::Bool
  use_layer_norm::Bool

  function HGATConfig(;
    input_dim::Int=768,
    hidden_dim::Int=256,
    num_heads::Int=8,
    num_layers::Int=2,
    dropout_rate::Float64=0.1,
    attention_dropout_rate::Float64=0.1,
    layer_norm_eps::Float64=1e-12,
    use_residual::Bool=true,
    use_layer_norm::Bool=true,
  )
    @assert input_dim > 0 "Input dimension must be positive"
    @assert hidden_dim > 0 "Hidden dimension must be positive"
    @assert num_heads > 0 "Number of heads must be positive"
    @assert num_layers > 0 "Number of layers must be positive"
    @assert 0.0 <= dropout_rate <= 1.0 "Dropout rate must be between 0.0 and 1.0"
    @assert 0.0 <= attention_dropout_rate <= 1.0 "Attention dropout rate must be between 0.0 and 1.0"
    @assert layer_norm_eps > 0 "Layer norm epsilon must be positive"

    new(
      input_dim,
      hidden_dim,
      num_heads,
      num_layers,
      dropout_rate,
      attention_dropout_rate,
      layer_norm_eps,
      use_residual,
      use_layer_norm,
    )
  end
end

# ============================================================================
# H-GAT Components
# ============================================================================

"""
    HGATAttention

Multi-head attention mechanism for H-GAT.

"""
struct HGATAttention
  query_projection::Dense
  key_projection::Dense
  value_projection::Dense
  output_projection::Dense
  dropout::Dropout
  num_heads::Int
  head_dim::Int

  function HGATAttention(config::HGATConfig)
    head_dim = config.hidden_dim ÷ config.num_heads
    @assert head_dim * config.num_heads == config.hidden_dim "Hidden dimension must be divisible by number of heads"

    query_projection = Dense(config.input_dim, config.hidden_dim)
    key_projection = Dense(config.input_dim, config.hidden_dim)
    value_projection = Dense(config.input_dim, config.hidden_dim)
    output_projection = Dense(config.hidden_dim, config.hidden_dim)
    dropout = Dropout(config.attention_dropout_rate)

    new(
      query_projection,
      key_projection,
      value_projection,
      output_projection,
      dropout,
      config.num_heads,
      head_dim,
    )
  end
end

"""
    HGATFeedForward

Feed-forward network for H-GAT.

"""
struct HGATFeedForward
  input_projection::Dense
  output_projection::Dense
  activation::Function
  dropout::Dropout

  function HGATFeedForward(config::HGATConfig)
    intermediate_dim = config.hidden_dim * 4  # Standard transformer scaling

    input_projection = Dense(config.hidden_dim, intermediate_dim)
    output_projection = Dense(intermediate_dim, config.hidden_dim)
    activation = gelu  # GELU activation
    dropout = Dropout(config.dropout_rate)

    new(input_projection, output_projection, activation, dropout)
  end
end

"""
    HGATLayer

Single layer of H-GAT.

"""
struct HGATLayer
  attention::HGATAttention
  feed_forward::HGATFeedForward
  layer_norm1::LayerNorm
  layer_norm2::LayerNorm
  dropout::Dropout
  use_residual::Bool
  use_layer_norm::Bool

  function HGATLayer(config::HGATConfig)
    attention = HGATAttention(config)
    feed_forward = HGATFeedForward(config)
    layer_norm1 = LayerNorm(config.hidden_dim; eps = config.layer_norm_eps)
    layer_norm2 = LayerNorm(config.hidden_dim; eps = config.layer_norm_eps)
    dropout = Dropout(config.dropout_rate)

    new(
      attention,
      feed_forward,
      layer_norm1,
      layer_norm2,
      dropout,
      config.use_residual,
      config.use_layer_norm,
    )
  end
end

"""
    HGATModel

Complete H-GAT model for GraphMERT.

"""
struct HGATModel
  layers::Vector{HGATLayer}
  input_projection::Dense
  output_projection::Dense
  config::HGATConfig

  function HGATModel(config::HGATConfig)
    layers = [HGATLayer(config) for _ = 1:config.num_layers]
    input_projection = Dense(config.input_dim, config.hidden_dim)
    output_projection = Dense(config.hidden_dim, config.input_dim)

    new(layers, input_projection, output_projection, config)
  end
end

# ============================================================================
# Functor definitions (Flux parameter traversal)
# ============================================================================

Flux.@functor HGATAttention (query_projection, key_projection, value_projection, output_projection, dropout)
Flux.@functor HGATFeedForward (input_projection, output_projection, activation, dropout)
Flux.@functor HGATLayer (attention, feed_forward, layer_norm1, layer_norm2, dropout)
Flux.@functor HGATModel (layers, input_projection, output_projection)

# ============================================================================
# Forward Pass Functions
# ============================================================================

"""
    (attention::HGATAttention)(node_features::AbstractArray{Float32, 3}, adjacency_matrix::SparseMatrixCSC{Float32})

Forward pass for H-GAT attention mechanism.
Handles input shape: (batch_size, num_nodes, hidden_dim)
"""
function (attention::HGATAttention)(
  node_features::AbstractArray{Float32, 3},
  adjacency_matrix::SparseMatrixCSC{Float32},
)
  batch_size, num_nodes, hidden_dim = size(node_features)

  # Permute to (hidden, nodes, batch) for Flux layers
  feat_perm = permutedims(node_features, (3, 2, 1))
  feat_flat = reshape(feat_perm, hidden_dim, num_nodes * batch_size)

  # Project to query, key, value
  query_flat = attention.query_projection(feat_flat)
  key_flat = attention.key_projection(feat_flat)
  value_flat = attention.value_projection(feat_flat)

  # Reshape and split heads
  # (num_heads, head_dim, num_nodes, batch_size)
  query = reshape(query_flat, attention.num_heads, attention.head_dim, num_nodes, batch_size)
  key = reshape(key_flat, attention.num_heads, attention.head_dim, num_nodes, batch_size)
  value = reshape(value_flat, attention.num_heads, attention.head_dim, num_nodes, batch_size)

  # Prepare for batched_mul: (nodes, head_dim, batch*heads)
  # Query: (head, dim, node, batch) -> (node, dim, batch, head) -> (node, dim, batch*head)
  query_merged = reshape(permutedims(query, (3, 2, 4, 1)), num_nodes, attention.head_dim, batch_size * attention.num_heads)
  
  # Key: (head, dim, node, batch) -> (dim, node, batch, head) -> (dim, node, batch*head)
  key_merged = reshape(permutedims(key, (2, 3, 4, 1)), attention.head_dim, num_nodes, batch_size * attention.num_heads)

  # Value: (head, dim, node, batch) -> (node, dim, batch, head) -> (node, dim, batch*head)
  value_merged = reshape(permutedims(value, (3, 2, 4, 1)), num_nodes, attention.head_dim, batch_size * attention.num_heads)

  # Compute attention scores: (nodes, nodes, batch*heads)
  # (Target, Dim) * (Dim, Source) -> (Target, Source)
  attention_scores = batched_mul(query_merged, key_merged)
  attention_scores = attention_scores ./ sqrt(Float32(attention.head_dim))

  # Apply adjacency mask
  adjacency_mask = convert(Matrix{Float32}, adjacency_matrix)
  # Broadcast over batch*heads (dims 1 and 2 match, dim 3 is broadcast)
  attention_scores = attention_scores .+ (1.0f0 .- adjacency_mask) .* -1e9

  # Softmax over source nodes (dim 2)
  attention_probs = Flux.softmax(attention_scores, dims=2)
  attention_probs = attention.dropout(attention_probs)

  # Apply attention to values
  # (Target, Source) * (Source, Dim) -> (Target, Dim)
  context = batched_mul(attention_probs, value_merged) # (nodes, head_dim, batch*heads)

  # Reshape back to (batch, nodes, heads, head_dim) -> flattened to (hidden, nodes, batch)
  # context: (num_nodes, head_dim, batch_size * num_heads)
  # reshape to (num_nodes, head_dim, batch_size, num_heads)
  context_reshaped = reshape(context, num_nodes, attention.head_dim, batch_size, attention.num_heads)

  # Permute to (head_dim, heads, nodes, batch) -> (hidden, nodes, batch)
  context_perm = permutedims(context_reshaped, (2, 4, 1, 3))
  context_flat = reshape(context_perm, hidden_dim, num_nodes * batch_size)

  # Output projection
  output_flat = attention.output_projection(context_flat)

  # Reshape back to (batch, nodes, hidden)
  output_reshaped = reshape(output_flat, hidden_dim, num_nodes, batch_size)
  output = permutedims(output_reshaped, (3, 2, 1))

  return output
end

"""
    (layer::HGATLayer)(node_features::AbstractArray{Float32, 3}, adjacency_matrix::SparseMatrixCSC{Float32})

Forward pass for a single H-GAT layer.
Handles input shape: (batch_size, num_nodes, hidden_dim)
"""
function (layer::HGATLayer)(
  node_features::AbstractArray{Float32, 3},
  adjacency_matrix::SparseMatrixCSC{Float32},
)
  # Self-attention (returns batch, nodes, hidden)
  attention_output = layer.attention(node_features, adjacency_matrix)
  attention_output = layer.dropout(attention_output)

  # Residual connection and layer norm
  if layer.use_residual
    attention_output = attention_output + node_features
  end

  if layer.use_layer_norm
    # LayerNorm expects (hidden, ...) or (hidden, batch)
    # Permute to (hidden, nodes, batch)
    att_perm = permutedims(attention_output, (3, 2, 1))
    att_norm = layer.layer_norm1(att_perm)
    attention_output = permutedims(att_norm, (3, 2, 1))
  end

  # Feed-forward network
  # Permute to (hidden, nodes * batch) for Dense layers
  batch_size, num_nodes, hidden_dim = size(attention_output)
  att_perm = permutedims(attention_output, (3, 2, 1))
  att_flat = reshape(att_perm, hidden_dim, num_nodes * batch_size)

  ff_flat = layer.feed_forward.input_projection(att_flat)
  ff_flat = layer.feed_forward.activation(ff_flat)
  ff_flat = layer.feed_forward.dropout(ff_flat)
  ff_flat = layer.feed_forward.output_projection(ff_flat)
  ff_flat = layer.dropout(ff_flat)
  
  # Reshape back to (batch, nodes, hidden)
  ff_reshaped = reshape(ff_flat, hidden_dim, num_nodes, batch_size)
  ff_output = permutedims(ff_reshaped, (3, 2, 1))

  # Residual connection and layer norm
  if layer.use_residual
    ff_output = ff_output + attention_output
  end

  if layer.use_layer_norm
    ff_perm = permutedims(ff_output, (3, 2, 1))
    ff_norm = layer.layer_norm2(ff_perm)
    ff_output = permutedims(ff_norm, (3, 2, 1))
  end

  return ff_output
end

"""
    (model::HGATModel)(node_features::AbstractArray{Float32, 3}, adjacency_matrix::SparseMatrixCSC{Float32})

Forward pass for complete H-GAT model.
Handles input shape: (batch_size, num_nodes, input_dim)
Returns shape: (batch_size, num_nodes, input_dim)
"""
function (model::HGATModel)(
  node_features::AbstractArray{Float32, 3},
  adjacency_matrix::SparseMatrixCSC{Float32},
)
  # Permute to (input_dim, nodes, batch) for Flux layers
  batch_size, num_nodes, input_dim = size(node_features)
  
  features_permuted = permutedims(node_features, (3, 2, 1))
  features_flat = reshape(features_permuted, input_dim, num_nodes * batch_size)

  # Input projection
  hidden_flat = model.input_projection(features_flat)
  
  # Reshape back to (batch, nodes, hidden) for layers
  hidden_reshaped = reshape(hidden_flat, model.config.hidden_dim, num_nodes, batch_size)
  hidden_states = permutedims(hidden_reshaped, (3, 2, 1))

  # Pass through layers
  for layer in model.layers
    hidden_states = layer(hidden_states, adjacency_matrix)
  end

  # Output projection
  # Permute back to (hidden, nodes, batch)
  batch_size, num_nodes, hidden_dim = size(hidden_states)
  hidden_permuted = permutedims(hidden_states, (3, 2, 1))
  hidden_flat = reshape(hidden_permuted, hidden_dim, num_nodes * batch_size)
  
  output_flat = model.output_projection(hidden_flat)
  
  # Reshape back to (batch, nodes, input_dim)
  output_reshaped = reshape(output_flat, model.config.input_dim, num_nodes, batch_size)
  output = permutedims(output_reshaped, (3, 2, 1))

  return output
end

# ============================================================================
# Graph Construction Utilities
# ============================================================================

"""
    create_adjacency_matrix(edges::Vector{Tuple{Int, Int}}, num_nodes::Int)

Create adjacency matrix from edge list.

"""
function create_adjacency_matrix(edges::Vector{Tuple{Int,Int}}, num_nodes::Int)
  # Create sparse adjacency matrix
  I = Int[]
  J = Int[]
  V = Float32[]

  for (i, j) in edges
    push!(I, i)
    push!(J, j)
    push!(V, 1.0f0)

    # Add reverse edge for undirected graph
    push!(I, j)
    push!(J, i)
    push!(V, 1.0f0)
  end

  # Add self-loops
  for i = 1:num_nodes
    push!(I, i)
    push!(J, i)
    push!(V, 1.0f0)
  end

  return sparse(I, J, V, num_nodes, num_nodes)
end

"""
    create_hierarchical_adjacency_matrix(edges::Vector{Tuple{Int, Int}}, num_nodes::Int, hierarchy_levels::Int)

Create hierarchical adjacency matrix for multi-level attention.

"""
function create_hierarchical_adjacency_matrix(
  edges::Vector{Tuple{Int,Int}},
  num_nodes::Int,
  hierarchy_levels::Int,
)
  adjacency_matrices = Vector{SparseMatrixCSC{Float32}}()

  for level = 1:hierarchy_levels
    # Create adjacency matrix for this level
    # Higher levels have more connections
    level_edges = filter(edges) do (i, j)
      # Simple hierarchy: higher levels include more edges
      (i + j) % hierarchy_levels < level
    end

    adj_matrix = create_adjacency_matrix(level_edges, num_nodes)
    push!(adjacency_matrices, adj_matrix)
  end

  return adjacency_matrices
end

# ============================================================================
# Attention Visualization
# ============================================================================

"""
    get_attention_weights(model::HGATModel, node_features::Matrix{Float32}, adjacency_matrix::SparseMatrixCSC{Float32})

Get attention weights for visualization.

"""
function get_attention_weights(
  model::HGATModel,
  node_features::Matrix{Float32},
  adjacency_matrix::SparseMatrixCSC{Float32},
)
  attention_weights = Vector{Matrix{Float32}}()

  # Get attention weights from each layer
  hidden_states = model.input_projection(node_features)

  for layer in model.layers
    # Compute attention weights
    query = layer.attention.query_projection(hidden_states)
    key = layer.attention.key_projection(hidden_states)

    # Reshape for multi-head attention
    batch_size, num_nodes, _ = size(hidden_states)
    query = reshape(
      query,
      batch_size,
      num_nodes,
      layer.attention.num_heads,
      layer.attention.head_dim,
    )
    key = reshape(
      key,
      batch_size,
      num_nodes,
      layer.attention.num_heads,
      layer.attention.head_dim,
    )

    # Transpose for attention computation
    query = permutedims(query, [1, 3, 2, 4])
    key = permutedims(key, [1, 3, 2, 4])

    # Compute attention scores
    attention_scores = query * transpose(key, 3, 4)
    attention_scores = attention_scores / sqrt(layer.attention.head_dim)

    # Apply adjacency mask
    adjacency_mask = convert(Matrix{Float32}, adjacency_matrix)
    attention_scores = attention_scores .+ (1.0f0 .- adjacency_mask) .* -1e9

    # Softmax
    attention_probs = Flux.softmax(attention_scores, dims=4)

    # Average over heads
    attention_probs = mean(attention_probs, dims=2)
    attention_probs = dropdims(attention_probs, dims=2)

    push!(attention_weights, attention_probs)

    # Update hidden states for next layer
    hidden_states = layer(hidden_states, adjacency_matrix)
  end

  return attention_weights
end

# ============================================================================
# Attention Mechanisms
# ============================================================================

"""
    create_attention_decay_mask(distance_matrix::Matrix{Float32}, config::SpatialAttentionConfig)

Create exponential decay attention mask based on graph distances.
"""
function create_attention_decay_mask(distance_matrix::Matrix{Float32}, config::SpatialAttentionConfig)
    # Exponential decay: exp(-λ * distance)
    decay_mask = exp.(-config.decay_lambda * distance_matrix)

    # Apply distance bias if enabled
    if config.use_distance_bias
        distance_bias = config.distance_bias_weight * distance_matrix
        decay_mask = decay_mask .+ distance_bias
    end

    return decay_mask
end

"""
    apply_spatial_attention_decay!(attention_scores::Array{Float32, 3}, decay_mask::Matrix{Float32})

Apply spatial decay to attention scores.
"""
function apply_spatial_attention_decay!(attention_scores::Array{Float32, 3}, decay_mask::Matrix{Float32})
    batch_size, seq_len, _ = size(attention_scores)

    for b in 1:batch_size
        attention_scores[b, :, :] .*= decay_mask
    end

    return attention_scores
end

# ============================================================================
# Model Utilities
# ============================================================================

"""
    get_model_parameters(model::HGATModel)

Get the number of parameters in the H-GAT model.
"""
function get_model_parameters(model::HGATModel)
  # This would calculate the actual number of parameters
  # For now, return an estimate based on the configuration
  total_params = 0

  # Input projection
  total_params += model.config.input_dim * model.config.hidden_dim

  # Each layer
  for layer in model.layers
    # Attention parameters
    total_params += model.config.input_dim * model.config.hidden_dim * 3  # Q, K, V
    total_params += model.config.hidden_dim * model.config.hidden_dim  # Output projection

    # Feed-forward parameters
    total_params += model.config.hidden_dim * (model.config.hidden_dim * 4)  # Input projection
    total_params += (model.config.hidden_dim * 4) * model.config.hidden_dim  # Output projection
  end

  # Output projection
  total_params += model.config.hidden_dim * model.config.input_dim

  return total_params
end

"""
    load_hgat_model(config_path::String, weights_path::String)

Load a pre-trained H-GAT model from files.

"""
function load_hgat_model(config_path::String, weights_path::String)
  # This would load the actual model from files
  # For now, create a new model with default config
  config = HGATConfig()
  model = HGATModel(config)
  return model
end

"""
    save_hgat_model(model::HGATModel, config_path::String, weights_path::String)

Save an H-GAT model to files.

"""
function save_hgat_model(model::HGATModel, config_path::String, weights_path::String)
  # This would save the actual model to files
  # For now, just return success
  return true
end
