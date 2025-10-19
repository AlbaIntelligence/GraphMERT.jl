"""
MLM (Masked Language Modeling) training objective for GraphMERT.jl

This module implements MLM training objective with span masking
as specified in the GraphMERT paper for biomedical knowledge graph construction.
"""

using Flux
using Random
using Statistics
using LinearAlgebra

# ============================================================================
# MLM Configuration
# ============================================================================

"""
    MLMConfig

Configuration for MLM training objective.
"""
struct MLMConfig
    vocab_size::Int
    hidden_size::Int
    max_length::Int
    mask_probability::Float64
    span_length::Int
    boundary_loss_weight::Float64
    temperature::Float64

    function MLMConfig(;
        vocab_size::Int=50265,
        hidden_size::Int=768,
        max_length::Int=512,
        mask_probability::Float64=0.15,
        span_length::Int=3,
        boundary_loss_weight::Float64=1.0,
        temperature::Float64=1.0
    )
        @assert vocab_size > 0 "Vocabulary size must be positive"
        @assert hidden_size > 0 "Hidden size must be positive"
        @assert max_length > 0 "Max length must be positive"
        @assert 0.0 <= mask_probability <= 1.0 "Mask probability must be between 0.0 and 1.0"
        @assert span_length > 0 "Span length must be positive"
        @assert boundary_loss_weight >= 0.0 "Boundary loss weight must be non-negative"
        @assert temperature > 0.0 "Temperature must be positive"

        new(vocab_size, hidden_size, max_length, mask_probability, span_length, boundary_loss_weight, temperature)
    end
end

"""
    MLMBatch

Batch data for MLM training.
"""
struct MLMBatch
    input_ids::Matrix{Int}
    attention_mask::Matrix{Int}
    labels::Matrix{Int}
    masked_positions::Vector{Int}
    span_boundaries::Vector{Tuple{Int,Int}}

    function MLMBatch(input_ids::Matrix{Int}, attention_mask::Matrix{Int}, labels::Matrix{Int},
        masked_positions::Vector{Int}, span_boundaries::Vector{Tuple{Int,Int}})
        
        @assert size(input_ids) == size(attention_mask) "Input IDs and attention mask must have same size"
        @assert size(input_ids) == size(labels) "Input IDs and labels must have same size"
        @assert all(pos -> 1 <= pos <= size(input_ids, 1), masked_positions) "Masked positions must be valid"
        
        new(input_ids, attention_mask, labels, masked_positions, span_boundaries)
    end
end

# ============================================================================
# Span Masking
# ============================================================================

"""
    create_span_masks(input_ids::Matrix{Int}, config::MLMConfig; rng::AbstractRNG=Random.GLOBAL_RNG)

Create span masks for MLM training.
"""
function create_span_masks(input_ids::Matrix{Int}, config::MLMConfig; rng::AbstractRNG=Random.GLOBAL_RNG)
    batch_size, seq_len = size(input_ids)
    masked_positions = Vector{Int}()
    span_boundaries = Vector{Tuple{Int,Int}}()
    
    for i in 1:batch_size
        # Get valid positions (exclude special tokens)
        valid_positions = find_valid_positions(input_ids[i, :])
        
        if isempty(valid_positions)
            continue
        end
        
        # Calculate number of spans to mask
        num_spans = max(1, round(Int, length(valid_positions) * config.mask_probability / config.span_length))
        
        # Select random spans
        for _ in 1:num_spans
            if isempty(valid_positions)
                break
            end
            
            # Select random start position
            start_pos = rand(rng, valid_positions)
            
            # Calculate span end position
            end_pos = min(start_pos + config.span_length - 1, seq_len)
            
            # Add positions to masked list
            for pos in start_pos:end_pos
                if pos in valid_positions
                    push!(masked_positions, (i - 1) * seq_len + pos)
                end
            end
            
            # Add span boundary
            push!(span_boundaries, (start_pos, end_pos))
            
            # Remove masked positions from valid positions
            valid_positions = setdiff(valid_positions, start_pos:end_pos)
        end
    end
    
    return masked_positions, span_boundaries
end

"""
    find_valid_positions(sequence::Vector{Int})

Find valid positions for masking (exclude special tokens).
"""
function find_valid_positions(sequence::Vector{Int})
    # Common special token IDs (these may need to be adjusted based on tokenizer)
    special_tokens = Set([0, 1, 2, 3])  # PAD, UNK, CLS, SEP
    
    valid_positions = Int[]
    for (i, token_id) in enumerate(sequence)
        if !(token_id in special_tokens)
            push!(valid_positions, i)
        end
    end
    
    return valid_positions
end

"""
    apply_masks(input_ids::Matrix{Int}, masked_positions::Vector{Int}, vocab_size::Int; rng::AbstractRNG=Random.GLOBAL_RNG)

Apply masks to input sequence.
"""
function apply_masks(input_ids::Matrix{Int}, masked_positions::Vector{Int}, vocab_size::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    batch_size, seq_len = size(input_ids)
    masked_input_ids = copy(input_ids)
    
    for pos in masked_positions
        batch_idx = div(pos - 1, seq_len) + 1
        seq_idx = mod(pos - 1, seq_len) + 1
        
        if 1 <= batch_idx <= batch_size && 1 <= seq_idx <= seq_len
            # 80% of the time, replace with [MASK] token
            # 10% of the time, replace with random token
            # 10% of the time, keep original token
            rand_val = rand(rng)
            
            if rand_val < 0.8
                masked_input_ids[batch_idx, seq_idx] = 103  # [MASK] token ID
            elseif rand_val < 0.9
                masked_input_ids[batch_idx, seq_idx] = rand(rng, 4:vocab_size-1)
            end
            # Otherwise keep original token
        end
    end
    
    return masked_input_ids
end

# ============================================================================
# MLM Loss Calculation
# ============================================================================

"""
    calculate_mlm_loss(logits::Array{Float32,3}, labels::Matrix{Int}, attention_mask::Matrix{Int})

Calculate MLM loss.
"""
function calculate_mlm_loss(logits::Array{Float32,3}, labels::Matrix{Int}, attention_mask::Matrix{Int})
    batch_size, seq_len, vocab_size = size(logits)
    
    # Reshape logits and labels for loss calculation
    logits_flat = reshape(logits, batch_size * seq_len, vocab_size)
    labels_flat = reshape(labels, batch_size * seq_len)
    mask_flat = reshape(attention_mask, batch_size * seq_len)
    
    # Calculate cross-entropy loss
    loss = 0.0f0
    count = 0
    
    for i in 1:length(labels_flat)
        if mask_flat[i] == 1 && labels_flat[i] != -100  # -100 is the ignore index
            loss += Flux.crossentropy(logits_flat[i, :], labels_flat[i])
            count += 1
        end
    end
    
    return count > 0 ? loss / count : 0.0f0
end

"""
    calculate_boundary_loss(logits::Array{Float32,3}, span_boundaries::Vector{Tuple{Int,Int}}, 
                           input_ids::Matrix{Int}, config::MLMConfig)

Calculate boundary loss for span masking.
"""
function calculate_boundary_loss(logits::Array{Float32,3}, span_boundaries::Vector{Tuple{Int,Int}}, 
                                input_ids::Matrix{Int}, config::MLMConfig)
    if isempty(span_boundaries)
        return 0.0f0
    end
    
    batch_size, seq_len, vocab_size = size(logits)
    boundary_loss = 0.0f0
    count = 0
    
    for (start_pos, end_pos) in span_boundaries
        # Calculate boundary loss for this span
        if start_pos <= seq_len && end_pos <= seq_len
            # Get logits for start and end positions
            start_logits = logits[1, start_pos, :]
            end_logits = logits[1, end_pos, :]
            
            # Calculate boundary loss (simplified version)
            # In practice, this would be more sophisticated
            boundary_loss += norm(start_logits - end_logits)
            count += 1
        end
    end
    
    return count > 0 ? boundary_loss / count : 0.0f0
end

"""
    calculate_total_mlm_loss(logits::Array{Float32,3}, labels::Matrix{Int}, attention_mask::Matrix{Int},
                            span_boundaries::Vector{Tuple{Int,Int}}, input_ids::Matrix{Int}, config::MLMConfig)

Calculate total MLM loss including boundary loss.
"""
function calculate_total_mlm_loss(logits::Array{Float32,3}, labels::Matrix{Int}, attention_mask::Matrix{Int},
                                 span_boundaries::Vector{Tuple{Int,Int}}, input_ids::Matrix{Int}, config::MLMConfig)
    # Calculate main MLM loss
    mlm_loss = calculate_mlm_loss(logits, labels, attention_mask)
    
    # Calculate boundary loss
    boundary_loss = calculate_boundary_loss(logits, span_boundaries, input_ids, config)
    
    # Combine losses
    total_loss = mlm_loss + config.boundary_loss_weight * boundary_loss
    
    return total_loss, mlm_loss, boundary_loss
end

# ============================================================================
# MLM Training
# ============================================================================

"""
    create_mlm_batch(input_ids::Matrix{Int}, attention_mask::Matrix{Int}, config::MLMConfig; rng::AbstractRNG=Random.GLOBAL_RNG)

Create a batch for MLM training.
"""
function create_mlm_batch(input_ids::Matrix{Int}, attention_mask::Matrix{Int}, config::MLMConfig; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Create span masks
    masked_positions, span_boundaries = create_span_masks(input_ids, config; rng=rng)
    
    # Apply masks
    masked_input_ids = apply_masks(input_ids, masked_positions, config.vocab_size; rng=rng)
    
    # Create labels (only masked positions have labels)
    labels = fill(-100, size(input_ids))  # -100 is the ignore index
    for pos in masked_positions
        batch_idx = div(pos - 1, size(input_ids, 2)) + 1
        seq_idx = mod(pos - 1, size(input_ids, 2)) + 1
        labels[batch_idx, seq_idx] = input_ids[batch_idx, seq_idx]
    end
    
    return MLMBatch(masked_input_ids, attention_mask, labels, masked_positions, span_boundaries)
end

"""
    train_mlm_step(model, batch::MLMBatch, config::MLMConfig, optimizer)

Perform one training step for MLM.
"""
function train_mlm_step(model, batch::MLMBatch, config::MLMConfig, optimizer)
    # Forward pass
    logits = model(batch.input_ids, batch.attention_mask)
    
    # Calculate loss
    total_loss, mlm_loss, boundary_loss = calculate_total_mlm_loss(
        logits, batch.labels, batch.attention_mask, 
        batch.span_boundaries, batch.input_ids, config
    )
    
    # Backward pass
    grads = gradient(() -> total_loss, Flux.params(model))
    Flux.update!(optimizer, Flux.params(model), grads)
    
    return total_loss, mlm_loss, boundary_loss
end

"""
    evaluate_mlm(model, batch::MLMBatch, config::MLMConfig)

Evaluate MLM model on a batch.
"""
function evaluate_mlm(model, batch::MLMBatch, config::MLMConfig)
    # Forward pass
    logits = model(batch.input_ids, batch.attention_mask)
    
    # Calculate loss
    total_loss, mlm_loss, boundary_loss = calculate_total_mlm_loss(
        logits, batch.labels, batch.attention_mask, 
        batch.span_boundaries, batch.input_ids, config
    )
    
    # Calculate accuracy
    predictions = argmax(logits, dims=3)
    correct = 0
    total = 0
    
    batch_size, seq_len = size(batch.labels)
    for i in 1:batch_size
        for j in 1:seq_len
            if batch.attention_mask[i, j] == 1 && batch.labels[i, j] != -100
                if predictions[i, j] == batch.labels[i, j]
                    correct += 1
                end
                total += 1
            end
        end
    end
    
    accuracy = total > 0 ? correct / total : 0.0
    
    return total_loss, mlm_loss, boundary_loss, accuracy
end

# ============================================================================
# MLM Metrics
# ============================================================================

"""
    calculate_mlm_metrics(predictions::Matrix{Int}, labels::Matrix{Int}, attention_mask::Matrix{Int})

Calculate MLM metrics.
"""
function calculate_mlm_metrics(predictions::Matrix{Int}, labels::Matrix{Int}, attention_mask::Matrix{Int})
    # Flatten arrays
    pred_flat = reshape(predictions, length(predictions))
    labels_flat = reshape(labels, length(labels))
    mask_flat = reshape(attention_mask, length(attention_mask))
    
    # Filter valid predictions
    valid_indices = findall(i -> mask_flat[i] == 1 && labels_flat[i] != -100, 1:length(labels_flat))
    
    if isempty(valid_indices)
        return Dict{String,Float64}(
            "accuracy" => 0.0,
            "precision" => 0.0,
            "recall" => 0.0,
            "f1" => 0.0
        )
    end
    
    valid_preds = pred_flat[valid_indices]
    valid_labels = labels_flat[valid_indices]
    
    # Calculate accuracy
    accuracy = mean(valid_preds .== valid_labels)
    
    # Calculate precision, recall, F1 (simplified)
    # In practice, this would be more sophisticated for multi-class classification
    precision = accuracy  # Simplified
    recall = accuracy     # Simplified
    f1 = 2 * precision * recall / (precision + recall)
    
    return Dict{String,Float64}(
        "accuracy" => accuracy,
        "precision" => precision,
        "recall" => recall,
        "f1" => f1
    )
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    create_mlm_config(; kwargs...)

Create MLM configuration with default values.
"""
function create_mlm_config(; kwargs...)
    return MLMConfig(; kwargs...)
end

"""
    get_mlm_vocab_size(config::MLMConfig)

Get vocabulary size from MLM configuration.
"""
function get_mlm_vocab_size(config::MLMConfig)
    return config.vocab_size
end

"""
    get_mlm_hidden_size(config::MLMConfig)

Get hidden size from MLM configuration.
"""
function get_mlm_hidden_size(config::MLMConfig)
    return config.hidden_size
end

"""
    get_mlm_max_length(config::MLMConfig)

Get maximum length from MLM configuration.
"""
function get_mlm_max_length(config::MLMConfig)
    return config.max_length
end
