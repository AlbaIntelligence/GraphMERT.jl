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
    rng::AbstractRNG
)::Tuple{Vector{Int}, Vector{Tuple{Int,Int}}}  # masked_positions, span_boundaries

    masked_positions = Int[]
    span_boundaries = Tuple{Int,Int}[]

    # Get all root positions (first part of sequence)
    num_roots = graph.config.num_roots

    # Calculate number of roots to mask
    num_to_mask = max(1, round(Int, num_roots * mlm_probability))

    # Randomly select root positions to mask
    root_positions = shuffle(rng, 1:num_roots)[1:min(num_to_mask, num_roots)]

    for root_pos in root_positions
        push!(masked_positions, root_pos)
        # For span masking, mask a span starting at this root
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
    rng::AbstractRNG
)::Tuple{LeafyChainGraph, Matrix{Int}}

    # Create copy of graph
    masked_graph = deepcopy(graph)

    # Create labels matrix (-100 = ignore, actual token_id = predict)
    labels = fill(-100, size(graph.root_tokens))

    for pos in masked_positions
        if pos <= size(graph.root_tokens, 1)
            original_token = graph.root_tokens[pos]

            # Store label
            labels[pos] = original_token

            # Apply masking strategy (same as MNM)
            rand_val = rand(rng)

            if rand_val < 0.8
                # 80%: Replace with [MASK]
                masked_graph.root_tokens[pos] = mask_token_id
            elseif rand_val < 0.9
                # 10%: Replace with random token
                masked_graph.root_tokens[pos] = rand(rng, 4:(vocab_size-1))
            end
            # 10%: Keep original
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

                # Get logits and compute cross-entropy
                leaf_logits = logits[batch_idx, seq_pos, :]
                loss = Flux.crossentropy(leaf_logits, label + 1)  # +1 because Flux uses 1-based indexing

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
    labels::Matrix{Int},
    span_boundaries::Vector{Tuple{Int,Int}},
    graph::LeafyChainGraph,
    config
)::Tuple{Float32, Float32}

    # Simple MLM loss calculation
    batch_size, seq_len, vocab_size = size(logits)
    mlm_loss = 0.0f0
    count = 0

    for i in 1:batch_size
        for j in 1:seq_len
            if labels[i,j] != -100
                mlm_loss += Flux.crossentropy(logits[i,j,:], labels[i,j] + 1)
                count += 1
            end
        end
    end

    mlm_loss = count > 0 ? mlm_loss / count : 0.0f0

    # Simple boundary loss
    boundary_loss = 0.0f0
    if !isempty(span_boundaries)
        for (start_pos, end_pos) in span_boundaries
            if start_pos <= seq_len && end_pos <= seq_len
                start_logits = logits[1, start_pos, :]
                end_logits = logits[1, end_pos, :]
                boundary_loss += norm(start_logits - end_logits)
            end
        end
        boundary_loss /= length(span_boundaries)
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
function forward_pass_mnm(
    model::GraphMERTModel,
    masked_graph::LeafyChainGraph
)::Array{Float32, 3}  # Returns logits [batch, seq_len, vocab_size]

    # TODO: EPIC 3 STORY 3.1 TASK 3.1.2 - Implement full MNM forward pass
    #
    # CURRENT STATUS: Stub implementation returning dummy logits
    # REASON: Depends on unimplemented GraphMERTModel components (embeddings, hgat, transformer, lm_head)
    #         and graph processing functions (graph_to_sequence, apply_hgat_fusion)
    #
    # REQUIRED COMPONENTS (to be implemented in future EPICs):
    # 1. graph_to_sequence(masked_graph) -> Vector{Int}  # Convert LeafyChainGraph to token sequence
    # 2. model.embeddings(input_ids) -> Matrix{Float32}  # Token embeddings [seq_len, hidden_dim]
    # 3. apply_hgat_fusion(model.hgat, token_embeddings, masked_graph) -> Matrix{Float32}  # H-GAT fusion
    # 4. model.transformer(fused_embeddings) -> Matrix{Float32}  # Transformer layers
    # 5. model.lm_head(hidden_states) -> Matrix{Float32}  # Language modeling head [seq_len, vocab_size]
    #
    # FINAL IMPLEMENTATION SHOULD BE:
    # input_ids = graph_to_sequence(masked_graph)
    # token_embeddings = model.embeddings(input_ids)
    # fused_embeddings = apply_hgat_fusion(model.hgat, token_embeddings, masked_graph)
    # hidden_states = model.transformer(fused_embeddings)
    # logits = model.lm_head(hidden_states)
    # return reshape(logits, 1, size(logits)...)  # Add batch dimension

    # Temporary stub: return random logits with correct shape
    seq_len = masked_graph.config.num_roots + masked_graph.config.num_roots * masked_graph.config.num_leaves_per_root
    vocab_size = 30522  # Default vocab size
    return rand(Float32, 1, seq_len, vocab_size)
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

    logits = forward_pass_mnm(model, joint_masked_graph)

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

    grads = gradient(() -> total_loss, Flux.params(model))
    Flux.update!(optimizer, Flux.params(model), grads)

    return total_loss, mlm_loss, mnm_loss
end

# Export functions for external use
export select_leaves_to_mask,
apply_mnm_masks,
calculate_mnm_loss,
forward_pass_mnm,
train_joint_mlm_mnm_step
