# Document 07: Masked Node Modeling (MNM) Training

## Semantic Space Training Objective

**Status**: 🔴 **CRITICAL - Only 30 lines stub, needs full implementation**
**Priority**: P0 (BLOCKING training pipeline)
**Paper Reference**: Section 4.2.2, Equations 5-6
**Existing Code**: `GraphMERT/src/training/mnm.jl` (30 lines, config stub only)

---

## Overview

**Masked Node Modeling (MNM)** is GraphMERT's novel training objective for learning semantic representations in the leaf space. It operates on semantic triples from the seed KG, teaching the model to predict KG nodes while jointly training relation embeddings through H-GAT.

### Key Differences from MLM

| Aspect              | MLM (Syntactic)                 | MNM (Semantic)           |
|---------------------|---------------------------------|--------------------------|
| **Space**           | Root nodes (text tokens)        | Leaf nodes (KG triples)  |
| **Masking**         | Partial spans (geometric dist.) | Entire leaf spans        |
| **Purpose**         | Learn syntactic patterns        | Learn semantic relations |
| **Training Target** | H-GAT relation embeddings       | Transformer parameters   |
| **Loss Component**  | L_MLM (with boundary loss)      | L_MNM (node prediction)  |

**Critical Insight**: MNM masks **entire leaf spans** (all 7 leaves of a root) rather than partial spans. This ensures gradients flow through the complete triple, allowing relation embeddings to capture the full semantic meaning.

---

## Mathematical Formulation

### MNM Loss (Equation 5 from paper)

$$\mathcal{L}_{\text{MNM}}(\theta) = -\sum_{\ell \in M_g} \log p_\theta(g_\ell \mid G_{\setminus M_g \cup M_x})$$

Where:

- $\theta$: All model parameters (transformer + H-GAT)
- $M_g$: Set of masked leaf nodes (semantic space)
- $M_x$: Set of masked root nodes (syntactic space)
- $g_\ell$: Target semantic token at leaf position $\ell$
- $G$: Complete leafy chain graph
- $G_{\setminus M_g \cup M_x}$: Graph with masked positions

### Joint Training Loss (Equation 6 from paper)

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{MLM}}(\theta) + \mu \cdot \mathcal{L}_{\text{MNM}}(\theta)$$

Where:

- $\mu$: Loss balancing weight (paper uses $\mu = 1.0$)
- $\mathcal{L}_{\text{MLM}}$: Masked language modeling loss (see Doc 06)

**Paper Finding**: Equal weighting ($\mu = 1.0$) works best, suggesting syntactic and semantic objectives are equally important.

---

## MNM Training Process

### Phase 1: Leaf Span Selection

**Input**: Leafy chain graph with injected triples
**Output**: Set of masked leaf positions $M_g$

**Algorithm**:

```julia
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
```

**Key Principle**: Unlike MLM which uses geometric distribution for span length, MNM **always masks the complete leaf group** for a selected root. This is essential for relation embedding training.

### Phase 2: Apply Masks

**Input**: Graph, masked positions
**Output**: Masked graph with labels

```julia
"""
    apply_mnm_masks(
        graph::LeafyChainGraph,
        masked_leaves::Vector{Tuple{Int,Int}},
        mask_token_id::Int,
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
)::Tuple{LeafyChainGraph, Matrix{Int}}

    # Create copy of graph
    masked_graph = deepcopy(graph)

    # Create labels matrix (-100 = ignore, actual token_id = predict)
    labels = fill(-100, size(graph.leaf_tokens))

    for (root_idx, leaf_idx) in masked_leaves
        original_token = graph.leaf_tokens[root_idx, leaf_idx]

        # Store label
        labels[root_idx, leaf_idx] = original_token

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
```

### Phase 3: Forward Pass with H-GAT

**Critical**: Gradient must flow through H-GAT to update relation embeddings.

```julia
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

    # 1. Convert graph to sequence
    input_ids = graph_to_sequence(masked_graph)  # [1024]

    # 2. Get token embeddings
    token_embeddings = model.embeddings(input_ids)  # [1024, hidden_dim]

    # 3. Apply H-GAT to fuse relation information into masked leaves
    # This is where relation embeddings get updated via backprop!
    fused_embeddings = apply_hgat_fusion(
        model.hgat,
        token_embeddings,
        masked_graph
    )

    # 4. Pass through transformer layers
    hidden_states = model.transformer(fused_embeddings)

    # 5. Project to vocabulary
    logits = model.lm_head(hidden_states)  # [1024, vocab_size]

    return logits
end
```

### Phase 4: Loss Calculation

```julia
"""
    calculate_mnm_loss(
        logits::Array{Float32, 3},
        labels::Matrix{Int},
        attention_mask::Matrix{Int},
        graph::LeafyChainGraph
    )

Calculate MNM loss on masked leaf predictions.
"""
function calculate_mnm_loss(
    logits::Array{Float32, 3},  # [batch, 1024, vocab_size]
    labels::Matrix{Int},         # [batch, 128, 7]
    attention_mask::Matrix{Int}, # [batch, 1024]
    graph::LeafyChainGraph
)::Float32

    batch_size = size(logits, 1)
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

                # Get logits and compute cross-entropy
                leaf_logits = logits[batch_idx, seq_pos, :]
                loss = Flux.crossentropy(softmax(leaf_logits), onehot(label, 1:vocab_size))

                total_loss += loss
                count += 1
            end
        end
    end

    return count > 0 ? total_loss / count : 0.0f0
end
```

---

## Gradient Flow and Relation Embedding Training

### Why Entire Leaf Masking Matters

**Paper Insight**: "Relation embeddings must receive gradients from the entire tail so that they can capture its full meaning, not just fragments from individual tokens."

```
Gradient Flow Path:
┌─────────────────────────────────────────────────┐
│  Loss (MNM)                                     │
│    ↓                                            │
│  Logits at masked leaf positions                │
│    ↓                                            │
│  Transformer Layers                             │
│    ↓                                            │
│  Fused Leaf Embeddings                          │
│    ↓                                            │
│  H-GAT Fusion ← RELATION EMBEDDINGS UPDATED     │
│    ↓                                            │
│  Token Embeddings + Relation Embeddings         │
└─────────────────────────────────────────────────┘
```

### Relation Embedding Update

From H-GAT (see Doc 04), for a masked leaf $t_i$:

$$t'_i = t_i + \text{H-GAT}(t_i, r, \{h_1, \ldots, h_m\})$$

Where H-GAT uses learnable relation parameters $\mathbf{W}_r$ and $\mathbf{a}_r$:

$$e_{ij}^{(r)} = \text{LeakyReLU}(\mathbf{a}_r^\top [\mathbf{W}_r t_i \| \mathbf{W}_r h_j])$$

**Backpropagation**: When computing $\frac{\partial \mathcal{L}_{\text{MNM}}}{\partial \mathbf{W}_r}$, gradients flow through:

1. Prediction loss at masked leaf
2. Transformer attention layers
3. Fused leaf embedding $t'_i$
4. H-GAT attention mechanism
5. **Relation matrix $\mathbf{W}_r$**

**Result**: $\mathbf{W}_r$ learns to encode the semantic meaning of relation $r$.

### Dropout on Relation Embeddings

**Paper uses**: 0.3 dropout on relation embeddings (higher than standard 0.1)

**Purpose**: Prevent overfitting on limited semantic examples (~28k triples vs. 124M tokens)

```julia
"""
Apply dropout specifically to relation embeddings during training.
"""
function apply_relation_dropout(
    relation_embeddings::Dict{Symbol, Matrix{Float32}},
    dropout_rate::Float64,
    training::Bool,
    rng::AbstractRNG
)::Dict{Symbol, Matrix{Float32}}

    if !training
        return relation_embeddings
    end

    dropped = Dict{Symbol, Matrix{Float32}}()

    for (relation, embedding) in relation_embeddings
        # Apply dropout mask
        mask = rand(rng, size(embedding)) .> dropout_rate
        dropped[relation] = embedding .* mask ./ (1.0 - dropout_rate)
    end

    return dropped
end
```

---

## Joint MLM + MNM Training

### Combined Training Step

```julia
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
        create_attention_mask(joint_masked_graph),
        joint_masked_graph
    )

    # Joint Loss
    total_loss = mlm_loss + μ * mnm_loss

    # === Backward Pass ===

    grads = gradient(() -> total_loss, Flux.params(model))
    Flux.update!(optimizer, Flux.params(model), grads)

    return total_loss, mlm_loss, mnm_loss
end
```

### Batch Processing

```julia
"""
    create_mnm_batch(
        graphs::Vector{LeafyChainGraph},
        mlm_config::MLMConfig,
        mnm_config::MNMConfig,
        rng::AbstractRNG
    )

Create a batch for joint MLM+MNM training.
"""
function create_mnm_batch(
    graphs::Vector{LeafyChainGraph},
    mlm_config::MLMConfig,
    mnm_config::MNMConfig,
    rng::AbstractRNG
)
    batch_size = length(graphs)
    config = graphs[1].config

    # Pre-allocate batch tensors
    input_ids = zeros(Int, batch_size, config.max_sequence_length)
    attention_masks = zeros(Int, batch_size, config.max_sequence_length)
    mlm_labels = fill(-100, batch_size, config.max_sequence_length)
    mnm_labels = fill(-100, batch_size, config.num_roots, config.num_leaves_per_root)

    # Process each graph in batch
    for (i, graph) in enumerate(graphs)
        # Apply MLM masking
        mlm_masked, mlm_lab = apply_mlm_masks_to_graph(graph, mlm_config, rng)

        # Apply MNM masking
        mnm_masked, mnm_lab = apply_mnm_masks_to_graph(mlm_masked, mnm_config, rng)

        # Store in batch
        input_ids[i, :] = graph_to_sequence(mnm_masked)
        attention_masks[i, :] = create_attention_mask(mnm_masked)
        mlm_labels[i, :] = mlm_lab
        mnm_labels[i, :, :] = mnm_lab
    end

    return (
        input_ids = input_ids,
        attention_masks = attention_masks,
        mlm_labels = mlm_labels,
        mnm_labels = mnm_labels
    )
end
```

---

## Configuration

### MNMConfig Structure

```julia
"""
    MNMConfig

Configuration for Masked Node Modeling training.
"""
struct MNMConfig
    # Vocabulary
    vocab_size::Int                    # 30522 for BioMedBERT
    hidden_size::Int                   # 512 in paper's 80M model
    max_length::Int                    # 1024 (fixed sequence length)

    # Masking
    mask_probability::Float64          # 0.15 (same as MLM)
    mask_token_id::Int                 # Usually 103 for [MASK]

    # Relation embeddings
    relation_embedding_dropout::Float64  # 0.3 (paper)
    num_relations::Int                 # e.g., 28 for diabetes seed KG

    # Training
    temperature::Float64               # 1.0 for loss scaling
    loss_weight::Float64               # μ = 1.0 (equal to MLM)
end

"""
Create MNM configuration matching paper specifications.
"""
function create_mnm_config(;
    vocab_size::Int = 30522,
    hidden_size::Int = 512,
    max_length::Int = 1024,
    mask_probability::Float64 = 0.15,
    mask_token_id::Int = 103,
    relation_embedding_dropout::Float64 = 0.3,
    num_relations::Int = 28,
    temperature::Float64 = 1.0,
    loss_weight::Float64 = 1.0
)::MNMConfig

    @assert 0.0 <= mask_probability <= 1.0 "Mask probability must be in [0,1]"
    @assert 0.0 <= relation_embedding_dropout <= 1.0 "Dropout must be in [0,1]"
    @assert vocab_size > 0 "Vocabulary size must be positive"
    @assert num_relations > 0 "Number of relations must be positive"

    return MNMConfig(
        vocab_size,
        hidden_size,
        max_length,
        mask_probability,
        mask_token_id,
        relation_embedding_dropout,
        num_relations,
        temperature,
        loss_weight
    )
end
```

---

## Worked Example

### Example: Diabetes Triple

**Input Graph**:

```
Root 0: "diabetes" → Leaves: ["disease", "syndrome", <pad>, ..., <pad>]
                     Relations: [:isa, :isa, nothing, ...]
```

**Step 1: Select for MNM Masking**

- Random draw: root 0 selected (probability 0.15)
- Mask ALL leaves of root 0 (entire span)

**Step 2: Apply Masks**

```
Masked leaves: [<MASK>, <MASK>, <pad>, ..., <pad>]
Labels:        [token_id("disease"), token_id("syndrome"), -100, ...]
```

**Step 3: Forward Pass**

- Token embeddings for [<MASK>, <MASK>, ...]
- H-GAT fuses with relation :isa and head "diabetes"
- Relation matrix W_isa gets updated via gradients
- Transformer predicts: ["disease", "syndrome", ...]

**Step 4: Loss**

```
MNM Loss = -log P(disease|context) - log P(syndrome|context)
```

**Step 5: Backprop**

- Gradient flows to W_isa
- W_isa learns: "isa relation connects entities to their hypernyms/categories"

---

## Validation and Testing

### Unit Tests

```julia
@testset "MNM Training" begin

    @testset "Leaf Span Selection" begin
        graph = create_test_graph_with_injections()
        masked = select_leaves_to_mask(graph, 0.15, MersenneTwister(42))

        # Should mask entire leaf spans
        roots_with_masks = unique([r for (r, _) in masked])
        for root_idx in roots_with_masks
            leaves_for_root = [l for (r, l) in masked if r == root_idx]
            # All injected leaves of this root should be masked
            @test length(leaves_for_root) == sum(graph.injected_mask[root_idx, :])
        end
    end

    @testset "MNM Loss Calculation" begin
        logits = randn(Float32, 2, 1024, 30522)  # batch=2
        labels = fill(-100, 2, 128, 7)
        labels[1, 1, 1] = 100  # One prediction target

        loss = calculate_mnm_loss(logits, labels, ones(Int, 2, 1024), test_graph)
        @test loss > 0.0
        @test !isnan(loss)
    end

    @testset "Gradient Flow Through H-GAT" begin
        model = create_test_model()
        graph = create_test_graph_with_injections()

        # Get initial relation embeddings
        initial_W = copy(model.hgat.relation_embeddings[:isa])

        # Training step
        loss, _, _ = train_joint_mlm_mnm_step(model, graph, mlm_config, mnm_config, opt)

        # Relation embeddings should have changed
        @test model.hgat.relation_embeddings[:isa] != initial_W
    end

    @testset "Joint MLM+MNM Training" begin
        model = create_test_model()
        graphs = [create_test_graph_with_injections() for _ in 1:4]

        total_loss, mlm_loss, mnm_loss = train_joint_mlm_mnm_step(
            model, graphs[1], mlm_config, mnm_config, opt, 1.0
        )

        @test total_loss ≈ mlm_loss + mnm_loss
        @test mlm_loss > 0
        @test mnm_loss > 0
    end
end
```

### Integration Tests

```julia
@testset "MNM Integration" begin

    @testset "With H-GAT" begin
        # Ensure H-GAT fusion works with masked leaves
        model = GraphMERTModel(config)
        graph = create_graph_with_masked_leaves()

        embeddings = forward_pass_mnm(model, graph)
        @test size(embeddings) == (1024, config.hidden_size)
    end

    @testset "With Training Pipeline" begin
        # Full training loop
        dataset = load_test_dataset()

        for epoch in 1:2
            for batch in dataset
                loss, mlm, mnm = train_joint_mlm_mnm_step(...)
                @test loss > 0
            end
        end
    end
end
```

---

## Performance Considerations

### Memory

**Additional Memory vs. MLM-only**:

- MNM labels: 128 × 7 × sizeof(Int) = 3.5KB per sample
- Minimal overhead since graph structure already loaded

**Optimization**:

- Process MLM and MNM masks in same forward pass
- Share intermediate activations
- No need for separate batches

### Computation

**Bottlenecks**:

- H-GAT fusion adds ~10% compute overhead
- Relation embedding updates relatively cheap (small matrices)

**Optimizations**:

- Cache non-injected leaves (skip H-GAT)
- Use sparse attention for leaf positions
- Batch H-GAT operations across leaves

---

## Troubleshooting

### Common Issues

**1. Relation Embeddings Not Updating**

- **Symptom**: W_r stays constant
- **Cause**: No gradient flow through H-GAT
- **Fix**: Ensure masked leaves are included in loss calculation

**2. NaN Loss**

- **Symptom**: Loss becomes NaN during training
- **Cause**: Division by zero when no leaves are masked
- **Fix**: Check `count > 0` before normalizing

**3. Imbalanced Relation Training**

- **Symptom**: Some relations learn, others don't
- **Cause**: Uneven relation distribution in seed KG
- **Fix**: Use injection algorithm's diversity balancing

**4. Overfitting on Seed KG**

- **Symptom**: Low MNM loss but poor extraction
- **Cause**: Memorizing seed triples
- **Fix**: Increase relation embedding dropout (0.3)

---

## Integration Points

### With Leafy Chain Graph (Doc 02)

- Requires injected graphs for semantic training
- Uses `injected_mask` to identify maskable leaves
- Respects graph structure for attention

### With H-GAT (Doc 04)

- **Critical dependency**: H-GAT must be differentiable
- Relation embeddings updated via MNM gradients
- Fusion happens even for masked leaves

### With MLM (Doc 06)

- Joint training in same forward pass
- Separate masking but combined loss
- Share transformer computations

### With Seed Injection (Doc 08)

- Quality of seed KG directly impacts MNM
- More diverse seed → better relation embeddings
- Injection algorithm ensures balanced training

### With Training Pipeline

- MNM integrated into main training loop
- Checkpoint includes relation embeddings
- Monitor MLM/MNM loss ratio for balance

---

## Implementation Checklist

- [ ] Define `MNMConfig` structure
- [ ] Implement `select_leaves_to_mask()`
- [ ] Implement `apply_mnm_masks()`
- [ ] Implement `calculate_mnm_loss()`
- [ ] Implement `train_joint_mlm_mnm_step()`
- [ ] Implement relation embedding dropout
- [ ] Add gradient checks for H-GAT flow
- [ ] Write unit tests for masking
- [ ] Write integration tests with H-GAT
- [ ] Profile and optimize performance
- [ ] Document with examples
- [ ] Validate on small dataset

---

## Next Steps

1. **Implement** MNM training in `training/mnm.jl`
2. **Test** gradient flow through H-GAT
3. **Integrate** with existing MLM training
4. **Validate** on small dataset with seed KG
5. **Proceed** to Seed KG Injection (Document 08)

---

**Related Documents**:

- → [Doc 02: Leafy Chain Graphs](02-leafy-chain-graphs.md) - Graph structure
- → [Doc 04: H-GAT Component](04-hgat-component.md) - Relation embedding fusion
- → [Doc 06: MLM Training](06-training-mlm.md) - Syntactic training objective
- → [Doc 08: Seed KG Injection](08-seed-kg-injection.md) - Training data preparation
