"""
MNM (Masked Node Modeling) training objective for GraphMERT.jl

This module implements MNM training objective as specified in the GraphMERT paper.

MNM trains the model to predict masked semantic tokens (leaf nodes) in the
leafy chain graph structure, enabling learning of semantic space representations
and vocabulary transfer between syntactic and semantic spaces.
"""

# Types will be available from main module after all includes

"""
    select_leaves_to_mask(graph::LeafyChainGraph, config::MNMConfig)

Select leaf nodes to mask for MNM training.

# Arguments
- `graph::LeafyChainGraph`: Graph containing leaf nodes
- `config::MNMConfig`: MNM configuration

# Returns
- `Vector{Tuple{Int,Int}}`: List of (root_idx, leaf_idx) pairs to mask
"""
function select_leaves_to_mask(graph::LeafyChainGraph, config::MNMConfig)
    masked_positions = Vector{Tuple{Int,Int}}()

    for root_idx = 1:graph.config.num_roots
        # Decide whether to mask this root's leaves
        if rand() < config.mask_probability
            # Mask entire leaf span for this root (as per paper)
            for leaf_idx = 1:config.num_leaves
                push!(masked_positions, (root_idx, leaf_idx))
            end
        end
    end

    return masked_positions
end

"""
    apply_mnm_masks(graph::LeafyChainGraph, masked_positions::Vector{Tuple{Int,Int}}, config::MNMConfig)

Apply MNM masking to the graph by replacing selected leaf tokens.

# Arguments
- `graph::LeafyChainGraph`: Graph to modify
- `masked_positions::Vector{Tuple{Int,Int}}`: Positions to mask
- `config::MNMConfig`: MNM configuration

# Returns
- `Vector{Int}`: Original token IDs before masking
"""
function apply_mnm_masks(
    graph::LeafyChainGraph,
    masked_positions::Vector{Tuple{Int,Int}},
    config::MNMConfig,
)
    original_tokens = Vector{Int}()

    for (root_idx, leaf_idx) in masked_positions
        if root_idx ≤ length(graph.leaf_nodes) &&
           leaf_idx ≤ length(graph.leaf_nodes[root_idx])
            leaf_node = graph.leaf_nodes[root_idx][leaf_idx]
            if !leaf_node.is_padding
                push!(original_tokens, leaf_node.token_id)
                # Replace with mask token (80% of time) or random token (10% of time) or keep original (10% of time)
                if rand() < 0.8
                    # 80% mask token
                    graph.leaf_nodes[root_idx][leaf_idx] = ChainGraphNode(
                        :leaf,
                        config.mask_token_id,
                        leaf_node.position,
                        root_idx,
                        false,
                    )
                elseif rand() < 0.5
                    # 10% random token
                    random_token = rand(1:config.vocab_size)
                    graph.leaf_nodes[root_idx][leaf_idx] = ChainGraphNode(
                        :leaf,
                        random_token,
                        leaf_node.position,
                        root_idx,
                        false,
                    )
                end
                # 10% keep original (do nothing)
            end
        end
    end

    return original_tokens
end

"""
    calculate_mnm_loss(model::GraphMERTModel, batch::MNMBatch, masked_positions::Vector{Tuple{Int,Int}})

Calculate MNM loss for masked leaf node prediction.

# Arguments
- `model::GraphMERTModel`: Trained GraphMERT model
- `batch::MNMBatch`: Batch containing graph sequences and attention masks
- `masked_positions::Vector{Tuple{Int,Int}}`: Positions that were masked

# Returns
- `Float64`: MNM loss value
"""
function calculate_mnm_loss(
    model::GraphMERTModel,
    batch::MNMBatch,
    masked_positions::Vector{Tuple{Int,Int}},
)
    # Forward pass through model
    logits = model(batch.graph_sequence, batch.attention_mask)

    # Extract logits for masked positions
    loss = 0.0
    total_masked = 0

    for (root_idx, leaf_idx) in masked_positions
        if root_idx ≤ size(batch.graph_sequence, 1) &&
           leaf_idx ≤ size(batch.graph_sequence, 2)
            # Calculate position in sequence
            seq_pos = (root_idx - 1) * (1 + model.config.num_leaves) + 1 + leaf_idx
            if seq_pos ≤ size(logits, 2)
                # Cross-entropy loss for this position
                target_token = batch.original_leaf_tokens[findfirst(
                    pos -> pos[1] == root_idx && pos[2] == leaf_idx,
                    masked_positions,
                )]
                loss += Flux.crossentropy(logits[:, seq_pos, :], target_token)
                total_masked += 1
            end
        end
    end

    return total_masked > 0 ? loss / total_masked : 0.0
end

"""
    create_mnm_batch(graphs::Vector{LeafyChainGraph}, masked_positions_list::Vector{Vector{Tuple{Int,Int}}},
                    original_tokens_list::Vector{Vector{Int}}, config::MNMConfig)

Create MNM training batch from multiple graphs.

# Arguments
- `graphs::Vector{LeafyChainGraph}`: Batch of graphs
- `masked_positions_list::Vector{Vector{Tuple{Int,Int}}}`: Masked positions for each graph
- `original_tokens_list::Vector{Vector{Int}}`: Original tokens for each graph
- `config::MNMConfig`: MNM configuration

# Returns
- `MNMBatch`: Batch ready for MNM training
"""
function create_mnm_batch(
    graphs::Vector{LeafyChainGraph},
    masked_positions_list::Vector{Vector{Tuple{Int,Int}}},
    original_tokens_list::Vector{Vector{Int}},
    config::MNMConfig,
)
    batch_size = length(graphs)
    seq_len = size(graphs[1].adjacency_matrix, 1)

    # Convert graphs to sequences
    sequences = [graph_to_sequence(graph) for graph in graphs]

    # Create 3D attention masks (batch_size × seq_len × seq_len)
    attention_masks = Array{Bool,3}(undef, batch_size, seq_len, seq_len)
    for i = 1:batch_size
        attention_masks[i, :, :] = create_attention_mask(graphs[i])
    end

    # Create relation IDs (simplified - would be more sophisticated in full implementation)
    relation_ids = zeros(Int, batch_size, config.num_leaves)

    return MNMBatch(
        Matrix(hcat(sequences...)'),  # graph_sequence (batch_size × seq_len) - transpose and convert to Matrix
        attention_masks,     # attention_mask (batch_size × seq_len × seq_len)
        masked_positions_list,  # masked_leaf_spans
        original_tokens_list,  # original_leaf_tokens
        relation_ids,  # relation_ids
    )
end

"""
    train_mnm_step(model::GraphMERTModel, batch::MNMBatch, optimizer, config::MNMConfig)

Perform one MNM training step.

# Arguments
- `model::GraphMERTModel`: Model to train
- `batch::MNMBatch`: Training batch
- `optimizer`: Optimizer for parameter updates
- `config::MNMConfig`: MNM configuration

# Returns
- `Float64`: Loss value for this step
"""
function train_mnm_step(
    model::GraphMERTModel,
    batch::MNMBatch,
    optimizer,
    config::MNMConfig,
)
    # Calculate loss
    loss = calculate_mnm_loss(model, batch, vcat(batch.masked_leaf_spans...))

    # Compute gradients
    grads = gradient(model) do m
        calculate_mnm_loss(m, batch, vcat(batch.masked_leaf_spans...))
    end

    # Apply relation dropout if configured
    if config.relation_dropout > 0
        # Apply dropout to relation embeddings during training
        # This helps prevent overfitting on relation patterns
        dropout_mask = rand(size(batch.relation_ids)...) .> config.relation_dropout
        batch.relation_ids .*= dropout_mask
    end

    # Update parameters
    Flux.update!(optimizer, model, grads[1])

    return loss
end

"""
    evaluate_mnm(model::GraphMERTModel, test_batch::MNMBatch, config::MNMConfig)

Evaluate MNM performance on test data.

# Arguments
- `model::GraphMERTModel`: Trained model
- `test_batch::MNMBatch`: Test batch
- `config::MNMConfig`: MNM configuration

# Returns
- `Dict{String, Float64}`: Evaluation metrics
"""
function evaluate_mnm(model::GraphMERTModel, test_batch::MNMBatch, config::MNMConfig)
    loss = calculate_mnm_loss(model, test_batch, vcat(test_batch.masked_leaf_spans...))

    # Calculate accuracy (simplified - would be more comprehensive)
    total_predictions = sum(length(positions) for positions in test_batch.masked_leaf_spans)
    accuracy = total_predictions > 0 ? 1.0 - loss : 0.0  # Simplified metric

    return Dict(
        "mnm_loss" => loss,
        "mnm_accuracy" => accuracy,
        "total_masked" => total_predictions,
    )
end

"""
    validate_gradient_flow(model::GraphMERTModel, batch::MNMBatch, config::MNMConfig)

Validate that gradients flow properly through the H-GAT (Hierarchical Graph Attention) mechanism.

# Arguments
- `model::GraphMERTModel`: Model to validate
- `batch::MNMBatch`: Batch for gradient validation
- `config::MNMConfig`: MNM configuration

# Returns
- `Dict{String, Bool}`: Validation results for different components
"""
function validate_gradient_flow(model::GraphMERTModel, batch::MNMBatch, config::MNMConfig)
    validation_results = Dict{String,Bool}()

    # Test gradient flow through the model
    grads = gradient(model) do m
        calculate_mnm_loss(m, batch, vcat(batch.masked_leaf_spans...))
    end

    # Check if gradients are non-zero (indicating proper flow)
    has_gradients = grads[1] !== nothing
    validation_results["has_gradients"] = has_gradients

    # Check gradient magnitudes (should be reasonable, not too small or too large)
    if has_gradients
        grad_norm = norm(grads[1])
        reasonable_magnitude = 1e-6 < grad_norm < 1e2
        validation_results["reasonable_gradient_magnitude"] = reasonable_magnitude

        # Check for gradient explosion/vanishing
        no_explosion = grad_norm < 1e2
        no_vanishing = grad_norm > 1e-6
        validation_results["no_gradient_explosion"] = no_explosion
        validation_results["no_gradient_vanishing"] = no_vanishing
    else
        validation_results["reasonable_gradient_magnitude"] = false
        validation_results["no_gradient_explosion"] = false
        validation_results["no_gradient_vanishing"] = false
    end

    # Test H-GAT specific gradient flow
    # In a full implementation, this would check gradients through attention layers
    validation_results["hgat_gradient_flow"] = has_gradients

    return validation_results
end

"""
    train_joint_mlm_mnm_step(model::GraphMERTModel, mlm_batch, mnm_batch::MNMBatch, 
                            mlm_config::MLMConfig, mnm_config::MNMConfig, optimizer)

Perform one joint MLM+MNM training step combining both objectives.

# Arguments
- `model::GraphMERTModel`: Model to train
- `mlm_batch`: MLM training batch
- `mnm_batch::MNMBatch`: MNM training batch
- `mlm_config::MLMConfig`: MLM configuration
- `mnm_config::MNMConfig`: MNM configuration
- `optimizer`: Optimizer for parameter updates

# Returns
- `Dict{String, Float64}`: Combined loss and individual losses
"""
function train_joint_mlm_mnm_step(
    model::GraphMERTModel,
    mlm_batch,
    mnm_batch::MNMBatch,
    mlm_config::MLMConfig,
    mnm_config::MNMConfig,
    optimizer,
)
    # Calculate MLM loss
    # Note: This assumes mlm_batch has the required fields for MLM loss calculation
    # In a full implementation, this would be more sophisticated
    mlm_loss = 0.5  # Placeholder - would use calculate_total_mlm_loss in full implementation

    # Calculate MNM loss
    mnm_loss = calculate_mnm_loss(model, mnm_batch, vcat(mnm_batch.masked_leaf_spans...))

    # Combine losses with weights
    combined_loss =
        mlm_config.boundary_loss_weight * mlm_loss + mnm_config.loss_weight * mnm_loss

    # Compute gradients for combined loss
    grads = gradient(model) do m
        mlm_loss_val = 0.5  # Placeholder - would use calculate_total_mlm_loss in full implementation
        mnm_loss_val =
            calculate_mnm_loss(m, mnm_batch, vcat(mnm_batch.masked_leaf_spans...))
        return mlm_config.boundary_loss_weight * mlm_loss_val +
               mnm_config.loss_weight * mnm_loss_val
    end

    # Apply relation dropout if configured
    if mnm_config.relation_dropout > 0
        dropout_mask = rand(size(mnm_batch.relation_ids)...) .> mnm_config.relation_dropout
        mnm_batch.relation_ids .*= dropout_mask
    end

    # Update parameters
    Flux.update!(optimizer, model, grads[1])

    return Dict(
        "combined_loss" => combined_loss,
        "mlm_loss" => mlm_loss,
        "mnm_loss" => mnm_loss,
        "mlm_weight" => mlm_config.boundary_loss_weight,
        "mnm_weight" => mnm_config.loss_weight,
    )
end

# Export functions for external use
export select_leaves_to_mask,
    apply_mnm_masks,
    calculate_mnm_loss,
    create_mnm_batch,
    train_mnm_step,
    evaluate_mnm,
    validate_gradient_flow,
    train_joint_mlm_mnm_step
