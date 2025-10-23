# Document 05: Attention Mechanisms and Graph Encodings
## Spatial Distance Encoding for Chain Graphs

**Status**: 🟡 **Needs Extraction from existing code**
**Priority**: P1 (High - needed for training)
**Paper Reference**: Section 4.2.1, Figure 3
**Existing Code**: Embedded in `architectures/hgat.jl`

---

## Overview

GraphMERT modifies standard transformer attention to incorporate **spatial distance** in the leafy chain graph. This ensures attention weights decay with graph distance, encouraging local coherence.

**Key Modification**: Multiply attention weights by exponential decay mask based on shortest path distances.

---

## Attention Decay Mask

### Mathematical Formulation

**From Paper** (Section 4.2.1):

$$f(sp(i,j)) = \lambda^{\text{GELU}(\sqrt{sp(i,j)} - p)}$$

Where:
- $sp(i,j)$: Shortest path distance between nodes $i$ and $j$
- $\lambda$: Decay base (hyperparameter, paper uses 0.6)
- $p$: Learnable threshold parameter
- $\text{GELU}$: Gaussian Error Linear Unit activation

**Modified Attention**:

$$\tilde{A}_{ij} = A_{ij} \odot f(sp(i,j))$$

Where:
- $A_{ij}$: Original attention weights
- $\tilde{A}_{ij}$: Modified attention weights with spatial decay
- $\odot$: Element-wise multiplication

---

## Component Functions

### 1. Shortest Path Computation

**Algorithm**: Floyd-Warshall on chain graph adjacency matrix

```julia
"""
    compute_shortest_paths(adjacency::SparseMatrixCSC{Float32})

Compute all-pairs shortest paths using Floyd-Warshall.

Returns: Matrix{Int} of size [N, N] where N = 1024
"""
function compute_shortest_paths(adjacency::SparseMatrixCSC{Float32})::Matrix{Int}
    N = size(adjacency, 1)
    dist = fill(typemax(Int) ÷ 2, N, N)

    # Initialize with direct edges
    rows, cols, _ = findnz(adjacency)
    for (i, j) in zip(rows, cols)
        if i != j
            dist[i, j] = 1
        else
            dist[i, j] = 0
        end
    end

    # Floyd-Warshall
    for k in 1:N
        for i in 1:N
            for j in 1:N
                if dist[i, k] + dist[k, j] < dist[i, j]
                    dist[i, j] = dist[i, k] + dist[k, j]
                end
            end
        end
    end

    return dist
end
```

**Complexity**: O(N³) = O(1024³) ≈ 1 billion operations

**Optimization**: Precompute once for the fixed graph structure, share across all samples

### 2. GELU Activation

```julia
"""
    gelu(x::Float32)

Gaussian Error Linear Unit activation.

GELU(x) = x * Φ(x)
where Φ(x) is the cumulative distribution function of standard normal.
"""
function gelu(x::Float32)::Float32
    return x * 0.5f0 * (1.0f0 + tanh(sqrt(2.0f0 / π) * (x + 0.044715f0 * x^3)))
end

# Vectorized version
function gelu(x::AbstractArray{Float32})
    return x .* 0.5f0 .* (1.0f0 .+ tanh.(sqrt(2.0f0 / π) .* (x .+ 0.044715f0 .* x.^3)))
end
```

### 3. Decay Function

```julia
"""
    compute_decay_mask(
        shortest_paths::Matrix{Int},
        λ::Float64 = 0.6,
        p::Float64 = 0.0
    )

Compute exponential decay mask based on shortest paths.

Args:
    shortest_paths: N×N matrix of shortest path distances
    λ: Decay base (0 < λ < 1)
    p: Threshold parameter (learned during training)

Returns: N×N matrix of decay factors
"""
function compute_decay_mask(
    shortest_paths::Matrix{Int},
    λ::Float64 = 0.6,
    p::Float64 = 0.0
)::Matrix{Float32}

    N = size(shortest_paths, 1)
    mask = zeros(Float32, N, N)

    for i in 1:N
        for j in 1:N
            # Compute exponent with GELU
            sp_ij = Float32(shortest_paths[i, j])
            exponent = gelu(sqrt(sp_ij) - p)

            # Exponential decay
            mask[i, j] = λ^exponent
        end
    end

    return mask
end
```

**Paper Modification**: Uses $\sqrt{sp(i,j)}$ instead of $sp(i,j)$ for smoother decay

**Rationale**: "experimentally need a smoother attention decay with respect to the shortest path"

### 4. Apply to Attention

```julia
"""
    apply_attention_decay(
        attention_weights::Array{Float32, 4},
        decay_mask::Matrix{Float32}
    )

Apply spatial decay to attention weights.

Args:
    attention_weights: [batch, heads, seq_len, seq_len]
    decay_mask: [seq_len, seq_len]

Returns: Modified attention weights
"""
function apply_attention_decay(
    attention_weights::Array{Float32, 4},
    decay_mask::Matrix{Float32}
)::Array{Float32, 4}

    batch_size, num_heads, seq_len, _ = size(attention_weights)

    # Broadcast decay mask across batch and heads
    # attention_weights [B, H, N, N]
    # decay_mask [N, N] → [1, 1, N, N]

    decay_expanded = reshape(decay_mask, 1, 1, seq_len, seq_len)

    # Element-wise multiplication
    return attention_weights .* decay_expanded
end
```

---

## Integration with Transformer

### Modified Attention Layer

```julia
"""
    attention_forward_with_decay(
        query::Matrix{Float32},
        key::Matrix{Float32},
        value::Matrix{Float32},
        decay_mask::Matrix{Float32},
        attention_dropout::Float64 = 0.1
    )

Transformer attention with spatial decay applied.
"""
function attention_forward_with_decay(
    query::Matrix{Float32},     # [seq_len, hidden_dim]
    key::Matrix{Float32},
    value::Matrix{Float32},
    decay_mask::Matrix{Float32}, # [seq_len, seq_len]
    attention_dropout::Float64 = 0.1
)

    hidden_dim = size(query, 2)
    seq_len = size(query, 1)

    # Compute attention scores
    # Q * K^T / sqrt(d_k)
    attention_scores = (query * key') / sqrt(Float32(hidden_dim))

    # Apply spatial decay mask
    attention_scores = attention_scores .* decay_mask

    # Softmax
    attention_probs = softmax(attention_scores, dims=2)

    # Dropout
    if attention_dropout > 0.0
        attention_probs = dropout(attention_probs, attention_dropout)
    end

    # Apply to values
    context = attention_probs * value

    return context, attention_probs
end
```

### Integration Point in GraphMERT

**Location**: Between H-GAT fusion and transformer layers

**Flow**:
```
1. Token Embeddings
2. H-GAT Fusion (for injected leaves)
3. → Transformer Layer 1
   a. Multi-head self-attention
   b. Compute Q, K, V
   c. Attention scores
   d. **Apply decay mask** ← HERE
   e. Softmax
   f. Apply to V
   g. Feed-forward
4. → Transformer Layers 2-12 (repeat)
5. Output logits
```

---

## Learnable Threshold Parameter

### Training the Threshold

**Parameter**: $p$ in the decay function

**Purpose**: Learn what constitutes "close" vs "far" in the graph

**Implementation**:

```julia
"""
    LearnableDecayMask

Learnable attention decay mask.
"""
mutable struct LearnableDecayMask
    λ::Float64              # Fixed decay base
    p::Vector{Float32}      # Learnable threshold [1]
    shortest_paths::Matrix{Int}

    function LearnableDecayMask(
        shortest_paths::Matrix{Int},
        λ::Float64 = 0.6,
        init_p::Float32 = 0.0f0
    )
        new(λ, [init_p], shortest_paths)
    end
end

"""
Compute mask with current p value.
"""
function (mask::LearnableDecayMask)()::Matrix{Float32}
    return compute_decay_mask(mask.shortest_paths, mask.λ, mask.p[1])
end

# Register as trainable parameter
Flux.@functor LearnableDecayMask
Flux.trainable(m::LearnableDecayMask) = (p = m.p,)
```

**Training**: $p$ updated via backpropagation along with other model parameters

---

## Precomputation and Caching

### Optimization Strategy

Since graph structure is **fixed**, many computations can be done once:

```julia
"""
    GraphEncodingCache

Precomputed graph encodings.
"""
struct GraphEncodingCache
    adjacency::SparseMatrixCSC{Float32}
    shortest_paths::Matrix{Int}
    base_decay_mask::Matrix{Float32}  # With p=0

    function GraphEncodingCache(config::ChainGraphConfig, λ::Float64 = 0.6)
        # Build adjacency
        adj = build_adjacency_matrix(config)

        # Compute shortest paths (expensive, O(N³))
        sp = compute_shortest_paths(adj)

        # Compute base decay mask
        base_mask = compute_decay_mask(sp, λ, 0.0)

        new(adj, sp, base_mask)
    end
end

# Global cache (shared across all graphs)
const GRAPH_ENCODING_CACHE = Ref{Union{GraphEncodingCache, Nothing}}(nothing)

function get_or_create_cache(config::ChainGraphConfig, λ::Float64 = 0.6)
    if GRAPH_ENCODING_CACHE[] === nothing
        @info "Building graph encoding cache (one-time cost)..."
        GRAPH_ENCODING_CACHE[] = GraphEncodingCache(config, λ)
    end
    return GRAPH_ENCODING_CACHE[]
end
```

**Memory**: ~4MB for shortest paths, negligible for other components

**Time**: ~1-2 seconds for initial computation, then instant access

---

## Shortest Path Properties for Chain Graphs

### Distance Patterns

For the fixed 128-root × 7-leaf structure:

**Root to Root** (chain structure):
```
sp(root_i, root_j) = |i - j|
Example: sp(0, 5) = 5
```

**Root to its Leaf**:
```
sp(root_i, leaf_i,j) = 1  (direct connection)
```

**Leaf to Leaf (same root)**:
```
sp(leaf_i,j, leaf_i,k) = 1  (clique structure)
```

**Root to other Root's Leaf**:
```
sp(root_i, leaf_j,k) = |i - j| + 1
Example: sp(root_0, leaf_5,0) = 5 + 1 = 6
```

**Leaf to Leaf (different roots)**:
```
sp(leaf_i,j, leaf_k,m) = |i - k| + 2
Example: sp(leaf_0,0, leaf_5,0) = 5 + 2 = 7
```

### Decay Visualization

With $\lambda = 0.6$:

| Distance | $\sqrt{distance}$ | Decay Factor |
| -------- | ----------------- | ------------ |
| 0        | 0                 | 1.0          |
| 1        | 1.0               | 0.6          |
| 4        | 2.0               | 0.36         |
| 9        | 3.0               | 0.216        |
| 16       | 4.0               | 0.130        |
| 25       | 5.0               | 0.078        |

**Effect**: Attention drops rapidly with distance, encouraging local interactions

---

## Configuration

### Hyperparameters

```julia
"""
    AttentionDecayConfig

Configuration for attention decay mechanism.
"""
struct AttentionDecayConfig
    lambda::Float64          # Decay base (0.6 in paper)
    init_p::Float32          # Initial threshold (0.0)
    learn_p::Bool            # Whether p is learnable (true)
    shared_across_layers::Bool  # Share mask across layers (true)
end

function default_attention_decay_config()
    AttentionDecayConfig(
        lambda = 0.6,
        init_p = 0.0f0,
        learn_p = true,
        shared_across_layers = true
    )
end
```

**Paper Choices**:
- $\lambda = 0.6$: Found via hyperparameter search
- Shared mask: Same mask used in all 12 transformer layers
- Learnable $p$: Yes, trained with rest of model

---

## Implementation in GraphMERT

### Complete Integration

```julia
"""
    GraphMERTWithDecay

GraphMERT model with attention decay.
"""
struct GraphMERTWithDecay
    embeddings::RoBERTaEmbeddings
    hgat::HGATModel
    transformer_layers::Vector{TransformerLayer}
    decay_mask::LearnableDecayMask
    lm_head::Dense
    config::GraphMERTConfig
end

"""
Forward pass with decay.
"""
function (model::GraphMERTWithDecay)(graph::LeafyChainGraph)
    # 1. Token embeddings
    embeddings = model.embeddings(graph_to_sequence(graph))

    # 2. H-GAT fusion
    fused = apply_hgat(model.hgat, embeddings, graph)

    # 3. Get current decay mask
    mask = model.decay_mask()

    # 4. Transformer layers with decay
    hidden = fused
    for layer in model.transformer_layers
        hidden = layer(hidden, decay_mask=mask)
    end

    # 5. Output projection
    logits = model.lm_head(hidden)

    return logits
end
```

---

## Testing and Validation

### Unit Tests

```julia
@testset "Attention Decay" begin

    @testset "Shortest Paths" begin
        config = default_chain_graph_config()
        adj = build_adjacency_matrix(config)
        sp = compute_shortest_paths(adj)

        # Test chain structure
        @test sp[1, 1] == 0  # Self
        @test sp[1, 2] == 1  # Adjacent roots
        @test sp[1, 10] == 9 # Distant roots

        # Test root-leaf
        @test sp[1, 129] == 1  # Root 0 to its first leaf
    end

    @testset "Decay Function" begin
        sp = [0 1 2; 1 0 1; 2 1 0]
        mask = compute_decay_mask(sp, 0.6, 0.0)

        @test mask[1, 1] ≈ 1.0     # Distance 0
        @test mask[1, 2] ≈ 0.6     # Distance 1
        @test mask[1, 3] < mask[1, 2]  # Decays with distance
    end

    @testset "Learnable Threshold" begin
        sp = zeros(Int, 10, 10)
        decay = LearnableDecayMask(sp, 0.6, 0.0f0)

        # Check trainable
        @test length(Flux.params(decay)) > 0

        # Update p
        decay.p[1] = 1.0f0
        mask = decay()
        @test size(mask) == (10, 10)
    end
end
```

---

## Performance Impact

### Computational Overhead

**Without decay**: Standard attention O(N²) per layer

**With decay**:
- Precomputation: O(N³) once (shared)
- Per forward pass: O(N²) element-wise multiply (negligible)

**Conclusion**: Minimal overhead (~1-2% slower per forward pass)

### Memory Overhead

- Shortest paths: 1024² × 4 bytes = 4MB (shared)
- Decay mask: 1024² × 4 bytes = 4MB (shared)
- Total: ~8MB (negligible on modern GPUs)

---

## Ablation Study (From Paper)

**Without attention decay**:
- Model works but slightly lower accuracy
- Less coherent local structures
- Attention more diffuse

**With attention decay** (λ=0.6):
- Better local coherence
- Improved FActScore by ~2%
- More interpretable attention patterns

**With different λ**:
- λ=0.8: Too weak decay
- λ=0.6: Optimal (paper choice)
- λ=0.4: Too strong, loses long-range dependencies

---

## Implementation Checklist

- [ ] Implement Floyd-Warshall shortest paths
- [ ] Implement GELU activation
- [ ] Implement decay mask computation
- [ ] Create `LearnableDecayMask` structure
- [ ] Integrate into transformer attention
- [ ] Add caching for precomputed values
- [ ] Write unit tests
- [ ] Profile performance impact
- [ ] Validate against paper results

---

**Related Documents**:
- → [Doc 02: Leafy Chain Graphs](02-leafy-chain-graphs.md) - Graph structure
- → [Doc 03: RoBERTa](03-roberta-encoder.md) - Base transformer
- → [Doc 06-07: Training](06-training-mlm.md) - How decay is used
