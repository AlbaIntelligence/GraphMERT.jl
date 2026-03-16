"""
MNM (Masked Node Modeling) training objective for GraphMERT.jl

This module implements MNM training objective as specified in the GraphMERT paper.

MNM trains the model to predict masked semantic tokens (leaf nodes) in the
leafy chain graph structure, enabling learning of semantic space representations
and vocabulary transfer between syntactic and semantic spaces.
"""

using Random
using Flux

# Types will be available from main module after all includes

"""
Select root positions to mask for MLM training.
"""
function select_roots_to_mask(
    graph::LeafyChainGraph,
    mlm_probability::Float64,
    rng::AbstractRNG,
)::Tuple{Vector{Int}, Vector{Tuple{Int,Int}}}  # masked_positions, span_boundaries

    masked_positions = Int[]
    span_boundaries = Tuple{Int,Int}[]

    num_roots = graph.config.num_roots
    pad_id = graph.config.pad_token_id

    valid_roots = findall(t -> t != pad_id, graph.root_tokens)
    if isempty(valid_roots)
        return masked_positions, span_boundaries
    end

    num_to_mask = min(length(valid_roots), max(1, round(Int, length(valid_roots) * mlm_probability)))
    root_positions = shuffle(rng, valid_roots)[1:num_to_mask]

    for root_pos in root_positions
        push!(masked_positions, root_pos)
        span_end = min(root_pos + 2, num_roots)  # Simple span of 3
        push!(span_boundaries, (root_pos, span_end))
    end

    return masked_positions, span_boundaries
end

"""
    select_leaves_to_mask(graph::LeafyChainGraph, mnm_probability::Float64, rng::AbstractRNG)

Select leaf spans to mask for MNM training.

Key difference from MLM: Masks ENTIRE leaf group (all 7 leaves) when selected.
"""
function select_leaves_to_mask(
    graph::LeafyChainGraph,
    mnm_probability::Float64,  # 0.15 as per paper
    rng::AbstractRNG
)::Vector{Tuple{Int,Int}}  # Returns (root_idx, leaf_idx) pairs

    masked_leaves = Tuple{Int,Int}[]

    # Find all roots with injections
    injected_roots = findall(any(graph.injected_mask, dims=2)[:])

    if isempty(injected_roots)
        return masked_leaves
    end

    # Calculate number of roots to mask
    num_to_mask = max(1, round(Int, length(injected_roots) * mnm_probability))

    # Randomly select roots to mask
    roots_to_mask = shuffle(rng, injected_roots)[1:min(num_to_mask, length(injected_roots))]

    # For each selected root, mask ALL its leaves
    for root_idx in roots_to_mask
        for leaf_idx in 1:graph.config.num_leaves_per_root
            # Only add to mask if this leaf was actually injected
            if graph.injected_mask[root_idx, leaf_idx]
                push!(masked_leaves, (root_idx, leaf_idx))
            end
        end
    end

    return masked_leaves
end

"""
    apply_mnm_masks(
        graph::LeafyChainGraph,
        masked_leaves::Vector{Tuple{Int,Int}},
        mask_token_id::Int,
        vocab_size::Int,
        rng::AbstractRNG
    )

Apply MNM masking strategy to leaf nodes.

Masking Strategy (same as MLM):
- 80%: Replace with [MASK] token
- 10%: Replace with random token
- 10%: Keep original token
"""
function apply_mnm_masks(
    graph::LeafyChainGraph,
    masked_leaves::Vector{Tuple{Int,Int}},
    mask_token_id::Int,  # Usually 103 for [MASK]
    vocab_size::Int,
    rng::AbstractRNG
)::Tuple{LeafyChainGraph, Array{Int,3}}

    # Create copy of graph
    masked_graph = deepcopy(graph)

    # Create labels array (-100 = ignore, actual token_id = predict)
    # Shape: [1, num_roots, num_leaves] for batch compatibility
    num_roots, num_leaves = size(graph.leaf_tokens)
    labels = fill(-100, (1, num_roots, num_leaves))

    for (root_idx, leaf_idx) in masked_leaves
        original_token = graph.leaf_tokens[root_idx, leaf_idx]

        # Store label
        labels[1, root_idx, leaf_idx] = original_token

        # Apply masking strategy
        rand_val = rand(rng)

        if rand_val < 0.8
            # 80%: Replace with [MASK]
            masked_graph.leaf_tokens[root_idx, leaf_idx] = mask_token_id
        elseif rand_val < 0.9
            # 10%: Replace with random token
            # Avoid special tokens (0-3: PAD, UNK, CLS, SEP)
            masked_graph.leaf_tokens[root_idx, leaf_idx] = rand(rng, 4:(vocab_size-1))
        end
        # 10%: Keep original (no change)
    end

    return masked_graph, labels
end

"""
Apply MLM masking to root positions.
"""
function apply_mlm_masks(
    graph::LeafyChainGraph,
    masked_positions::Vector{Int},
    mask_token_id::Int,
    vocab_size::Int,
    rng::AbstractRNG,
)::Tuple{LeafyChainGraph, Vector{Int}}

    masked_graph = deepcopy(graph)

    # Root labels are 1D (num_roots); -100 indicates "ignore"
    labels = fill(-100, length(graph.root_tokens))
    pad_id = graph.config.pad_token_id

    for pos in masked_positions
        if !(1 <= pos <= length(graph.root_tokens))
            continue
        end

        original_token = graph.root_tokens[pos]
        if original_token == pad_id
            continue
        end

        labels[pos] = original_token

        rand_val = rand(rng)
        if rand_val < 0.8
            masked_graph.root_tokens[pos] = mask_token_id
        elseif rand_val < 0.9
            if vocab_size > 4
                masked_graph.root_tokens[pos] = rand(rng, 4:(vocab_size-1))
            end
        end
    end

    return masked_graph, labels
end

"""
    calculate_mnm_loss(
        logits::Array{Float32, 3},
        labels::Array{Int, 3},
        graph::LeafyChainGraph
    )

Calculate MNM loss on masked leaf predictions.
"""
function calculate_mnm_loss(
    logits::Array{Float32, 3},  # [batch, seq_len, vocab_size]
    labels::Array{Int, 3},       # [batch, num_roots, num_leaves]
    graph::LeafyChainGraph
)::Float32

    batch_size = size(logits, 1)
    seq_len = size(logits, 2)
    vocab_size = size(logits, 3)
    config = graph.config

    total_loss = 0.0f0
    count = 0

    # Iterate over leaf positions
    for batch_idx in 1:batch_size
        for root_idx in 1:config.num_roots
            for leaf_idx in 1:config.num_leaves_per_root

                label = labels[batch_idx, root_idx, leaf_idx]

                # Skip if not a prediction target
                if label == -100
                    continue
                end

                # Get position in sequence
                seq_pos = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + leaf_idx

                # Ensure seq_pos is within bounds
                if seq_pos > seq_len
                    continue
                end

                # Use one-vs-all BCE on the vocabulary logits so the loss stays
                # well-defined for the full logit vector produced by the LM head.
                leaf_logits = logits[batch_idx, seq_pos, :]
                target = zeros(Float32, vocab_size)
                label_idx = clamp(label + 1, 1, vocab_size)  # convert 0-based token ids to Julia indices
                target[label_idx] = 1.0f0
                loss = Flux.Losses.logitbinarycrossentropy(leaf_logits, target)

                total_loss += loss
                count += 1
            end
        end
    end

    return count > 0 ? total_loss / count : 0.0f0
end

"""
Calculate MLM loss with boundary loss.
"""
function calculate_mlm_loss_with_boundary(
    logits::Array{Float32,3},
    labels::Vector{Int},
    span_boundaries::Vector{Tuple{Int,Int}},
    graph::LeafyChainGraph,
    config,
)::Tuple{Float32, Float32}

    batch_size, seq_len, vocab_size = size(logits)
    @assert batch_size == 1 "calculate_mlm_loss_with_boundary currently assumes batch_size == 1"

    mlm_loss = 0.0f0
    count = 0

    # Root MLM loss: root positions map 1:graph.config.num_roots in the transformer sequence.
    for pos in 1:length(labels)
        label_token = labels[pos]
        if label_token == -100
            continue
        end

        label_idx = label_token + 1  # 0-based token ids → Julia indices
        if !(1 <= label_idx <= vocab_size)
            continue
        end

        lsm = Flux.logsoftmax(@view logits[1, pos, :])
        mlm_loss += -lsm[label_idx]
        count += 1
    end

    mlm_loss = count > 0 ? mlm_loss / count : 0.0f0

    boundary_loss = 0.0f0
    if !isempty(span_boundaries)
        pad_id = graph.config.pad_token_id
        valid_count = 0

        for (start_pos, end_pos) in span_boundaries
            if !(1 <= start_pos <= graph.config.num_roots && 1 <= end_pos <= graph.config.num_roots)
                continue
            end
            if graph.root_tokens[start_pos] == pad_id || graph.root_tokens[end_pos] == pad_id
                continue
            end
            if start_pos <= seq_len && end_pos <= seq_len
                start_logits = @view logits[1, start_pos, :]
                end_logits = @view logits[1, end_pos, :]
                boundary_loss += norm(start_logits - end_logits)
                valid_count += 1
            end
        end

        boundary_loss = valid_count > 0 ? boundary_loss / valid_count : 0.0f0
    end

    return mlm_loss + boundary_loss, boundary_loss
end

"""
    forward_pass_mnm(
        model::GraphMERTModel,
        masked_graph::LeafyChainGraph
    )

Forward pass through GraphMERT for MNM.

Key: Even though leaves are masked, H-GAT still fuses them with:
- Relation embeddings (being trained)
- Head token embeddings (from roots)
"""
function _prepare_roberta_inputs_from_graph(
    masked_graph::LeafyChainGraph,
)::Tuple{Matrix{Int}, Array{Float32, 3}, Matrix{Int}, Matrix{Int}, Int}
    input_ids_vec = graph_to_sequence(masked_graph)
    seq_len = length(input_ids_vec)

    # RoBERTa expects (seq_len, batch_size) with 0-based IDs (we shift to 1-based inside embeddings)
    input_ids = reshape(input_ids_vec, seq_len, 1)

    # (batch, seq, seq) float mask used by attention
    attention_mask = create_attention_mask(input_ids)

    # Position and token type IDs (seq_len, batch_size), 0-based (shifted inside embeddings)
    position_ids = create_position_ids(seq_len, 1)
    token_type_ids = create_token_type_ids(seq_len, 1)

    return input_ids, attention_mask, position_ids, token_type_ids, seq_len
end

function _forward_pass_mnm_inputs(
    model::GraphMERTModel,
    input_ids::Matrix{Int},
    attention_mask::Array{Float32, 3},
    position_ids::Matrix{Int},
    token_type_ids::Matrix{Int},
    seq_len::Int,
)::Array{Float32, 3}
    encoder_output, _ = model.roberta(input_ids, attention_mask, position_ids, token_type_ids)
    batch_size, _, _ = size(encoder_output)

    hidden_dim = size(encoder_output, 3)
    encoder_reshaped = reshape(encoder_output, batch_size * seq_len, hidden_dim)
    encoder_reshaped = permutedims(encoder_reshaped, (2, 1))               # (hidden_dim, batch*seq)
    lm_2d = model.lm_head(encoder_reshaped)                                 # (vocab_size, batch*seq)
    vocab_size = size(lm_2d, 1)
    lm_3d = reshape(lm_2d, vocab_size, batch_size, seq_len)                 # (vocab, batch, seq)
    lm_3d = permutedims(lm_3d, (2, 3, 1))                                   # (batch, seq, vocab)

    return Array{Float32,3}(lm_3d)
end

function forward_pass_mnm(
    model::GraphMERTModel,
    masked_graph::LeafyChainGraph,
)::Array{Float32, 3}  # Returns logits [batch, seq_len, vocab_size]
    input_ids, attention_mask, position_ids, token_type_ids, seq_len =
        _prepare_roberta_inputs_from_graph(masked_graph)

    return _forward_pass_mnm_inputs(
        model,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        seq_len,
    )
end

"""
    train_joint_mlm_mnm_step(
        model::GraphMERTModel,
        graph::LeafyChainGraph,
        mlm_config::MLMConfig,
        mnm_config::MNMConfig,
        optimizer,
        μ::Float64 = 1.0
    )

Perform one joint training step with both MLM and MNM objectives.
"""
function train_joint_mlm_mnm_step(
    model::GraphMERTModel,
    graph::LeafyChainGraph,
    mlm_config::MLMConfig,
    mnm_config::MNMConfig,
    optimizer,
    μ::Float64 = 1.0,
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Tuple{Float32, Float32, Float32}

    # === MLM Masking (Syntactic Space) ===

    # Select root spans to mask
    masked_root_positions, root_span_boundaries = select_roots_to_mask(
        graph, mlm_config.mask_probability, rng
    )

    # Apply MLM masks
    mlm_masked_graph, mlm_labels = apply_mlm_masks(
        graph, masked_root_positions, mlm_config.mask_token_id, mlm_config.vocab_size, rng
    )

    # === MNM Masking (Semantic Space) ===

    # Select leaf spans to mask
    masked_leaf_positions = select_leaves_to_mask(
        mlm_masked_graph,  # Already has MLM masks applied
        mnm_config.mask_probability,
        rng
    )

    # Apply MNM masks (on top of MLM masks)
    joint_masked_graph, mnm_labels = apply_mnm_masks(
        mlm_masked_graph,
        masked_leaf_positions,
        mnm_config.mask_token_id,
        mnm_config.vocab_size,
        rng
    )

    # === Forward Pass ===

    input_ids, attention_mask, position_ids, token_type_ids, seq_len =
        _prepare_roberta_inputs_from_graph(joint_masked_graph)

    logits = _forward_pass_mnm_inputs(
        model,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        seq_len,
    )

    # === Loss Calculation ===

    # MLM Loss (with boundary loss)
    mlm_loss, boundary_loss = calculate_mlm_loss_with_boundary(
        logits,
        mlm_labels,
        root_span_boundaries,
        joint_masked_graph,
        mlm_config
    )

    # MNM Loss
    mnm_loss = calculate_mnm_loss(
        logits,
        mnm_labels,
        joint_masked_graph
    )

    # Joint Loss
    total_loss = mlm_loss + μ * mnm_loss

    # === Backward Pass ===

    ps = Flux.params(model)
    grads = gradient(ps) do
        logits_g = _forward_pass_mnm_inputs(
            model,
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids,
            seq_len,
        )
        mlm_loss_g, _ = calculate_mlm_loss_with_boundary(
            logits_g,
            mlm_labels,
            root_span_boundaries,
            joint_masked_graph,
            mlm_config,
        )
        mnm_loss_g = calculate_mnm_loss(logits_g, mnm_labels, joint_masked_graph)
        return mlm_loss_g + μ * mnm_loss_g
    end
    Flux.update!(optimizer, ps, grads)

    # Recompute scalar losses for reporting (post-update)
    logits2 = _forward_pass_mnm_inputs(
        model,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        seq_len,
    )
    mlm2, _ = calculate_mlm_loss_with_boundary(
        logits2,
        mlm_labels,
        root_span_boundaries,
        joint_masked_graph,
        mlm_config,
    )
    mnm2 = calculate_mnm_loss(logits2, mnm_labels, joint_masked_graph)

    return mlm2 + μ * mnm2, mlm2, mnm2
end

# Export functions for external use
export select_leaves_to_mask,
apply_mnm_masks,
calculate_mnm_loss,
forward_pass_mnm,
train_joint_mlm_mnm_step
