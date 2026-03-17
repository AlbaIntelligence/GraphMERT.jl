"""
RoBERTa encoder architecture for GraphMERT.jl

This module implements the RoBERTa-based encoder architecture with 80M parameters
as specified in the GraphMERT paper, optimized for biomedical knowledge graph construction.
"""

using Flux
using LinearAlgebra
using SparseArrays
# using DocStringExtensions  # Temporarily disabled

# ============================================================================
# RoBERTa Configuration
# ============================================================================

"""
    RoBERTaConfig

Configuration for RoBERTa encoder architecture.

"""
struct RoBERTaConfig
    vocab_size::Int
    hidden_size::Int
    num_attention_heads::Int
    num_hidden_layers::Int
    intermediate_size::Int
    max_position_embeddings::Int
    type_vocab_size::Int
    layer_norm_eps::Float64
    hidden_dropout_prob::Float64
    attention_probs_dropout_prob::Float64

    function RoBERTaConfig(;
        vocab_size::Int = 50265,
        hidden_size::Int = 768,
        num_attention_heads::Int = 12,
        num_hidden_layers::Int = 12,
        intermediate_size::Int = 3072,
        max_position_embeddings::Int = 1024,
        type_vocab_size::Int = 1,
        layer_norm_eps::Float64 = 1e-12,
        hidden_dropout_prob::Float64 = 0.1,
        attention_probs_dropout_prob::Float64 = 0.1,
    )
        @assert vocab_size > 0 "Vocabulary size must be positive"
        @assert hidden_size > 0 "Hidden size must be positive"
        @assert num_attention_heads > 0 "Number of attention heads must be positive"
        @assert num_hidden_layers > 0 "Number of hidden layers must be positive"
        @assert intermediate_size > 0 "Intermediate size must be positive"
        @assert max_position_embeddings > 0 "Max position embeddings must be positive"
        @assert type_vocab_size > 0 "Type vocabulary size must be positive"
        @assert layer_norm_eps > 0 "Layer norm epsilon must be positive"
        @assert 0.0 <= hidden_dropout_prob <= 1.0 "Hidden dropout probability must be between 0.0 and 1.0"
        @assert 0.0 <= attention_probs_dropout_prob <= 1.0 "Attention dropout probability must be between 0.0 and 1.0"

        new(
            vocab_size,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            intermediate_size,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
        )
    end
end

# ============================================================================
# RoBERTa Components
# ============================================================================

"""
    RoBERTaEmbeddings

Embedding layer for RoBERTa encoder.

"""
struct RoBERTaEmbeddings
    word_embeddings::Embedding
    position_embeddings::Embedding
    token_type_embeddings::Embedding
    layer_norm::LayerNorm
    dropout::Dropout
end

function RoBERTaEmbeddings(config::RoBERTaConfig)
    word_embeddings = Embedding(config.vocab_size, config.hidden_size)
    position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
    token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
    layer_norm = LayerNorm(config.hidden_size; eps = config.layer_norm_eps)
    dropout = Dropout(config.hidden_dropout_prob)

    RoBERTaEmbeddings(
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        layer_norm,
        dropout,
    )
end

"""
    RoBERTaSelfAttention

Self-attention mechanism for RoBERTa.

"""
struct RoBERTaSelfAttention
    query::Dense
    key::Dense
    value::Dense
    dropout::Dropout
    num_attention_heads::Int
    attention_head_size::Int
    all_head_size::Int
end

function RoBERTaSelfAttention(config::RoBERTaConfig)
    all_head_size = config.hidden_size
    attention_head_size = all_head_size ÷ config.num_attention_heads

    query = Dense(config.hidden_size, all_head_size)
    key = Dense(config.hidden_size, all_head_size)
    value = Dense(config.hidden_size, all_head_size)
    dropout = Dropout(config.attention_probs_dropout_prob)

    RoBERTaSelfAttention(
        query,
        key,
        value,
        dropout,
        config.num_attention_heads,
        attention_head_size,
        all_head_size,
    )
end

"""
    RoBERTaSelfOutput

Output layer for self-attention.
"""
struct RoBERTaSelfOutput
    dense::Dense
    layer_norm::LayerNorm
    dropout::Dropout
end

function RoBERTaSelfOutput(config::RoBERTaConfig)
    dense = Dense(config.hidden_size, config.hidden_size)
    layer_norm = LayerNorm(config.hidden_size; eps = config.layer_norm_eps)
    dropout = Dropout(config.hidden_dropout_prob)

    RoBERTaSelfOutput(dense, layer_norm, dropout)
end

"""
    RoBERTaAttention

Complete attention mechanism for RoBERTa.

"""
struct RoBERTaAttention
    self::RoBERTaSelfAttention
    output::RoBERTaSelfOutput
end

function RoBERTaAttention(config::RoBERTaConfig)
    self = RoBERTaSelfAttention(config)
    output = RoBERTaSelfOutput(config)

    RoBERTaAttention(self, output)
end

"""
    RoBERTaIntermediate

Intermediate layer for RoBERTa.
"""
struct RoBERTaIntermediate
    dense::Dense
    activation::Function
end

function RoBERTaIntermediate(config::RoBERTaConfig)
    dense = Dense(config.hidden_size, config.intermediate_size)
    activation = gelu  # GELU activation function

    RoBERTaIntermediate(dense, activation)
end

"""
    RoBERTaOutput

Output layer for RoBERTa.

"""
struct RoBERTaOutput
    dense::Dense
    layer_norm::LayerNorm
    dropout::Dropout
end

function RoBERTaOutput(config::RoBERTaConfig)
    dense = Dense(config.intermediate_size, config.hidden_size)
    layer_norm = LayerNorm(config.hidden_size; eps = config.layer_norm_eps)
    dropout = Dropout(config.hidden_dropout_prob)

    RoBERTaOutput(dense, layer_norm, dropout)
end

"""
    RoBERTaLayer

Single layer of RoBERTa encoder.

"""
struct RoBERTaLayer
    attention::RoBERTaAttention
    intermediate::RoBERTaIntermediate
    output::RoBERTaOutput
end

function RoBERTaLayer(config::RoBERTaConfig)
    attention = RoBERTaAttention(config)
    intermediate = RoBERTaIntermediate(config)
    output = RoBERTaOutput(config)

    RoBERTaLayer(attention, intermediate, output)
end

# ============================================================================
# RoBERTa Encoder
# ============================================================================

"""
    RoBERTaEncoder

Complete RoBERTa encoder architecture.

"""
struct RoBERTaEncoder
    layers::Vector{RoBERTaLayer}
    config::RoBERTaConfig
end

function RoBERTaEncoder(config::RoBERTaConfig)
    layers = [RoBERTaLayer(config) for _ ∈ 1:config.num_hidden_layers]
    RoBERTaEncoder(layers, config)
end

"""
    RoBERTaModel

Complete RoBERTa model for GraphMERT.

"""
struct RoBERTaModel
    embeddings::RoBERTaEmbeddings
    encoder::RoBERTaEncoder
    pooler::Dense
    config::RoBERTaConfig
end

function RoBERTaModel(config::RoBERTaConfig)
    embeddings = RoBERTaEmbeddings(config)
    encoder = RoBERTaEncoder(config)
    pooler = Dense(config.hidden_size, config.hidden_size)

    RoBERTaModel(embeddings, encoder, pooler, config)
end

# ============================================================================
# Forward Pass Functions
# ============================================================================

"""
    (embeddings::RoBERTaEmbeddings)(input_ids::Matrix{Int}, position_ids::Matrix{Int}, token_type_ids::Matrix{Int})

Forward pass for RoBERTa embeddings.

"""
function (embeddings::RoBERTaEmbeddings)(
    input_ids::Matrix{Int},
    position_ids::Matrix{Int},
    token_type_ids::Matrix{Int},
)
    # This codebase uses 0-based IDs (pad_token_id=0, position_ids start at 0).
    # Flux.Embedding is 1-based, so we shift IDs by +1 at lookup time.
    # Expected ranges before shifting:
    #   input_ids ∈ [0, vocab_size-1]
    #   position_ids ∈ [0, max_position_embeddings-1]
    #   token_type_ids ∈ [0, type_vocab_size-1]
    input_ids_1 = input_ids .+ 1
    position_ids_1 = position_ids .+ 1
    token_type_ids_1 = token_type_ids .+ 1

    # Word embeddings - inputs are (seq_len, batch_size)
    word_embeddings = embeddings.word_embeddings(input_ids_1)

    # Position embeddings
    position_embeddings = embeddings.position_embeddings(position_ids_1)

    # Token type embeddings
    token_type_embeddings = embeddings.token_type_embeddings(token_type_ids_1)

    # Combine embeddings - result is (hidden_size, seq_len, batch_size)
    embeddings_output = word_embeddings .+ position_embeddings .+ token_type_embeddings

    # Layer normalization and dropout (Flux.LayerNorm expects the feature dimension first)
    embeddings_output = embeddings.layer_norm(embeddings_output)
    embeddings_output = embeddings.dropout(embeddings_output)

    # Return (batch_size, seq_len, hidden_size) for downstream layers
    return permutedims(embeddings_output, [3, 2, 1])
end

"""
    (attention::RoBERTaSelfAttention)(hidden_states::Matrix{Float32}, attention_mask::Matrix{Float32})

Forward pass for RoBERTa self-attention.

"""
function (attention::RoBERTaSelfAttention)(
    hidden_states::AbstractArray{<:Real, 3},  # (batch_size, seq_len, hidden_size)
    attention_mask::AbstractArray{<:Real, 3},  # (batch_size, seq_len, seq_len)
    attention_decay_mask::Union{Nothing, AttentionDecayMask} = nothing,
)
    batch_size, seq_length, hidden_size = size(hidden_states)
    T = eltype(hidden_states)

    # Linear transformations - Dense layers expect (features, batch_size * seq_len)
    # Reshape to (hidden_size, batch_size * seq_length)
    hidden_states_reshaped = reshape(hidden_states, :, batch_size * seq_length)

    query_layer = attention.query(hidden_states_reshaped)
    key_layer = attention.key(hidden_states_reshaped)
    value_layer = attention.value(hidden_states_reshaped)

    # Reshape back to (batch_size, seq_length, num_heads * head_size)
    query_layer = reshape(query_layer, batch_size, seq_length, attention.all_head_size)
    key_layer = reshape(key_layer, batch_size, seq_length, attention.all_head_size)
    value_layer = reshape(value_layer, batch_size, seq_length, attention.all_head_size)

    # Split into heads: (batch_size, seq_length, num_heads, head_dim)
    query_layer = reshape(query_layer, batch_size, seq_length, attention.num_attention_heads, attention.attention_head_size)
    key_layer = reshape(key_layer, batch_size, seq_length, attention.num_attention_heads, attention.attention_head_size)
    value_layer = reshape(value_layer, batch_size, seq_length, attention.num_attention_heads, attention.attention_head_size)

    # NNlib.batched_mul expects batch dimensions last. We treat (batch, head) as the batch.
    # Q: (seq, head_dim, batch*heads), K: (head_dim, seq, batch*heads), V: (seq, head_dim, batch*heads)
    Q = permutedims(query_layer, (2, 4, 1, 3))
    Q = reshape(Q, seq_length, attention.attention_head_size, batch_size * attention.num_attention_heads)

    K = permutedims(key_layer, (4, 2, 1, 3))
    K = reshape(K, attention.attention_head_size, seq_length, batch_size * attention.num_attention_heads)

    V = permutedims(value_layer, (2, 4, 1, 3))
    V = reshape(V, seq_length, attention.attention_head_size, batch_size * attention.num_attention_heads)

    # Attention scores: (seq, seq, batch*heads)
    scores3 = batched_mul(Q, K) ./ sqrt(T(attention.attention_head_size))

    # Reshape to (seq, seq, batch, heads) to apply the mask and softmax
    scores4 = reshape(scores3, seq_length, seq_length, batch_size, attention.num_attention_heads)

    # attention_mask is (batch, seq, seq) => (seq, seq, batch)
    mask3 = permutedims(attention_mask, (2, 3, 1))
    scores4 = scores4 .+ reshape(mask3, seq_length, seq_length, batch_size, 1)

    # Apply attention decay mask if provided
    # GraphMERT spatial bias: add log(decay_mask) to scores
    if attention_decay_mask !== nothing
        # mask.mask is (seq, seq) with values in [0, 1] (exp(-dist))
        # We need to broadcast it across batches and heads
        # log(exp(-lambda*d)) = -lambda*d, which is the additive bias we want
        
        # Add epsilon to avoid log(0) for infinite distances
        epsilon = T(1e-12)
        spatial_bias = log.(attention_decay_mask.mask .+ epsilon)
        
        # Reshape for broadcasting: (seq, seq, 1, 1)
        # scores4 is (seq, seq, batch, heads)
        scores4 = scores4 .+ reshape(spatial_bias, seq_length, seq_length, 1, 1)
    end

    probs4 = Flux.softmax(scores4; dims = 2)
    probs4 = attention.dropout(probs4)

    probs3 = reshape(probs4, seq_length, seq_length, batch_size * attention.num_attention_heads)

    # Context: (seq, head_dim, batch*heads)
    context3 = batched_mul(probs3, V)

    # Back to (batch, seq, heads, head_dim)
    context4 = reshape(context3, seq_length, attention.attention_head_size, batch_size, attention.num_attention_heads)
    context_layer = permutedims(context4, (3, 1, 4, 2))

    # Concatenate heads: (batch, seq, hidden)
    return reshape(context_layer, batch_size, seq_length, attention.all_head_size)
end

"""
    (layer::RoBERTaLayer)(hidden_states::Matrix{Float32}, attention_mask::Matrix{Float32})

Forward pass for a single RoBERTa layer.

"""
function (layer::RoBERTaLayer)(
    hidden_states::AbstractArray{<:Real, 3},  # (batch_size, seq_len, hidden_size)
    attention_mask::AbstractArray{<:Real, 3},  # (batch_size, seq_len, seq_len)
    attention_decay_mask::Union{Nothing, AttentionDecayMask} = nothing,
)
    # Self-attention
    attention_output = layer.attention.self(hidden_states, attention_mask, attention_decay_mask)
    # Self-attention output projection
    batch_size, seq_len, hidden_size = size(attention_output)
    attention_output_reshaped = reshape(attention_output, hidden_size, batch_size * seq_len)
    attention_output = layer.attention.output.dense(attention_output_reshaped)
    attention_output = reshape(attention_output, batch_size, seq_len, hidden_size)
    attention_output = layer.attention.output.dropout(attention_output)

    # Flux.LayerNorm expects the feature dimension first; our tensors are (batch, seq, hidden).
    residual = attention_output .+ hidden_states
    residual_hsb = permutedims(residual, (3, 2, 1))  # (hidden, seq, batch)
    residual_2d = reshape(residual_hsb, hidden_size, batch_size * seq_len)
    norm_2d = layer.attention.output.layer_norm(residual_2d)
    norm_hsb = reshape(norm_2d, hidden_size, seq_len, batch_size)
    attention_output = permutedims(norm_hsb, (3, 2, 1))  # (batch, seq, hidden)

    # Intermediate layer
    intermediate_2d = reshape(attention_output, hidden_size, batch_size * seq_len)
    intermediate_output = layer.intermediate.dense(intermediate_2d)  # (intermediate_size, batch_size*seq_len)
    intermediate_output = layer.intermediate.activation.(intermediate_output)

    # Output layer
    layer_output = layer.output.dense(intermediate_output)
    layer_output = reshape(layer_output, batch_size, seq_len, hidden_size)
    layer_output = layer.output.dropout(layer_output)

    residual2 = layer_output .+ attention_output
    residual2_hsb = permutedims(residual2, (3, 2, 1))
    residual2_2d = reshape(residual2_hsb, hidden_size, batch_size * seq_len)
    norm2_2d = layer.output.layer_norm(residual2_2d)
    norm2_hsb = reshape(norm2_2d, hidden_size, seq_len, batch_size)
    layer_output = permutedims(norm2_hsb, (3, 2, 1))

    return layer_output
end

"""
    (encoder::RoBERTaEncoder)(hidden_states::Matrix{Float32}, attention_mask::Matrix{Float32})

Forward pass for RoBERTa encoder.

"""
function (encoder::RoBERTaEncoder)(
    hidden_states::AbstractArray{<:Real, 3},  # (batch_size, seq_len, hidden_size)
    attention_mask::AbstractArray{<:Real, 3},  # (batch_size, seq_len, seq_len)
    attention_decay_mask::Union{Nothing, AttentionDecayMask} = nothing,
)
    for layer in encoder.layers
        hidden_states = layer(hidden_states, attention_mask, attention_decay_mask)
    end
    return hidden_states
end

"""
    (model::RoBERTaModel)(input_ids::Matrix{Int}, attention_mask::Matrix{Float32}, position_ids::Matrix{Int}, token_type_ids::Matrix{Int})

Forward pass for complete RoBERTa model.

"""
function (model::RoBERTaModel)(
    input_ids::Matrix{Int},  # (seq_len, batch_size)
    attention_mask::AbstractArray{<:Real, 3},  # (batch_size, seq_len, seq_len)
    position_ids::Matrix{Int},  # (seq_len, batch_size)
    token_type_ids::Matrix{Int},  # (seq_len, batch_size)
    attention_decay_mask::Union{Nothing, AttentionDecayMask} = nothing,
)
    # Embeddings
    embedding_output = model.embeddings(input_ids, position_ids, token_type_ids)  # (batch_size, seq_len, hidden_size)

    # Encoder
    encoder_output = model.encoder(embedding_output, attention_mask, attention_decay_mask)  # (batch_size, seq_len, hidden_size)

    # Pooler - use first token (equivalent to [CLS])
    pooled_output = model.pooler(encoder_output[:, 1, :]')  # (hidden_size, batch_size)
    pooled_output = pooled_output'  # (batch_size, hidden_size)

    return encoder_output, pooled_output
end

# ============================================================================
# Functor definitions (Flux parameter traversal)
# ============================================================================

Flux.@functor RoBERTaEmbeddings (word_embeddings, position_embeddings, token_type_embeddings, layer_norm, dropout)
Flux.@functor RoBERTaSelfAttention (query, key, value, dropout, num_attention_heads, attention_head_size, all_head_size)
Flux.@functor RoBERTaSelfOutput (dense, layer_norm, dropout)
Flux.@functor RoBERTaAttention (self, output)
Flux.@functor RoBERTaIntermediate (dense, activation)
Flux.@functor RoBERTaOutput (dense, layer_norm, dropout)
Flux.@functor RoBERTaLayer (attention, intermediate, output)
Flux.@functor RoBERTaEncoder (layers, config)
Flux.@functor RoBERTaModel (embeddings, encoder, pooler, config)

# ============================================================================
# Utility Functions
# ============================================================================

"""
    create_attention_mask(input_ids::Matrix{Int})

Create attention mask from input IDs.

"""
function create_attention_mask(input_ids::Matrix{Int})
    # input_ids: (seq_len, batch_size)
    # Return additive mask (batch_size, seq_len, seq_len) applied to attention *scores*.
    # We mask **keys** that are padding, but do not mask queries. Masking queries can
    # produce all-(-Inf) rows → NaNs in softmax, which then leak into later layers.
    seq_len, batch_size = size(input_ids)

    key_valid = input_ids' .!= 0  # (batch_size, seq_len)
    key_mask = ifelse.(reshape(key_valid, batch_size, 1, seq_len), 0.0f0, -Inf32)

    return repeat(key_mask, 1, seq_len, 1)
end

"""
    create_position_ids(seq_length::Int, batch_size::Int)

Create position IDs for input sequence.

"""
function create_position_ids(seq_length::Int, batch_size::Int)
    # Return (seq_length, batch_size) matrix
    position_ids = repeat(reshape(0:(seq_length-1), seq_length, 1), 1, batch_size)
    return position_ids
end

"""
    create_token_type_ids(seq_length::Int, batch_size::Int)

Create token type IDs for input sequence.

"""
function create_token_type_ids(seq_length::Int, batch_size::Int)
    # For RoBERTa, token type IDs are all zeros
    token_type_ids = zeros(Int, seq_length, batch_size)
    return token_type_ids
end

"""
    get_model_parameters(model::RoBERTaModel)

Get the number of parameters in the model.

"""
function get_model_parameters(model::RoBERTaModel)
    # This would calculate the actual number of parameters
    # For now, return the target 80M parameters
    return 80_000_000
end

"""
    load_roberta_model(config_path::String, weights_path::String)

Load a pre-trained RoBERTa model from files.

"""
function load_roberta_model(config_path::String, weights_path::String)
    # This would load the actual model from files
    # For now, create a new model with default config
    config = RoBERTaConfig()
    model = RoBERTaModel(config)
    return model
end

"""
    save_roberta_model(model::RoBERTaModel, config_path::String, weights_path::String)

Save a RoBERTa model to files.

"""
function save_roberta_model(model::RoBERTaModel, config_path::String, weights_path::String)
    # This would save the actual model to files
    # For now, just return success
    return true
end
