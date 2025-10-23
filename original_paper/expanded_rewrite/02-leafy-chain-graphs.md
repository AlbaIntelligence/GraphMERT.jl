# Document 02: Leafy Chain Graph Structure

## Foundational Data Structure for GraphMERT

**Status**: 🔴 **CRITICAL - Currently 30 lines, mostly empty**
**Priority**: P0 (BLOCKING)
**Paper Reference**: Section 4.1, Figures 2-3
**Existing Code**: `GraphMERT/src/graphs/leafy_chain.jl` (30 lines, placeholder)

---

## Overview

The **leafy chain graph** is the foundational data structure that enables GraphMERT to unify syntactic and semantic representations. It is the key innovation that allows training on both text tokens (syntactic) and KG triples (semantic) simultaneously.

### Key Concepts

**Syntactic Space** (Text):

- Represents unstructured text as token sequences
- Captures linguistic patterns and surface forms
- Source of MLM training signal

**Semantic Space** (Knowledge Graph):

- Represents structured knowledge as triples
- Captures ontological relationships
- Source of MNM training signal

**Leafy Chain Graph**:

- Bridges both spaces in a unified representation
- Enables joint training and vocabulary transfer
- Regular structure allows efficient sequential encoding

---

## Graph Structure

### Topology

A leafy chain graph `G = (V, E)` consists of:

**Nodes** `V`:

- **Root nodes** (𝑟): Represent syntactic tokens from text
- **Leaf nodes** (𝑙): Represent semantic tokens from KG triples

**Edges** `E`:

- **Root-to-leaf edges**: Connect root tokens to their associated KG triples
- **Leaf-to-leaf edges**: Connect leaves belonging to the same root
- All edges are **undirected**

### Fixed Structure

GraphMERT uses a **regular** (fixed) structure:

- **Number of root nodes**: `N_root = 128` (fixed)
- **Number of leaves per root**: `N_leaf = 7` (fixed)
- **Total nodes**: `N_total = 128 + (128 × 7) = 1024` nodes
- **Sequence length**: 1024 tokens (fits efficiently in GPU memory)

**Rationale for Fixed Structure**:

1. Enables **efficient batch processing** (fixed tensor sizes)
2. Simplifies **graph encoding** (no need for dynamic structures)
3. Allows **precomputation** of shortest paths and attention masks
4. Matches **GPU memory constraints** efficiently

---

## Node Types

### Root Nodes (Syntactic Space)

**Purpose**: Represent tokens from the input text sequence

**Properties**:

- Position in sequence: `1 ≤ i ≤ 128`
- Token ID from vocabulary: `token_id ∈ [0, vocab_size-1]`
- Embedding: Standard transformer token embedding
- Always present (never empty)

**Example**:

```
Text: "Diabetes mellitus is a chronic metabolic disorder..."
Roots: [diabetes, mellitus, is, a, chronic, metabolic, disorder, ...]
       (tokenized to IDs: [2156, 23421, 16, 10, 6844, 19131, 8761, ...])
```

### Leaf Nodes (Semantic Space)

**Purpose**: Represent tail tokens from KG triples

**Properties**:

- Associated with specific root node `r`
- Position within root's leaves: `1 ≤ j ≤ 7`
- Can be:
  - **Empty** (`<pad>` token): No injection
  - **Semantic token**: From KG triple tail
- Embedding: Token embedding + H-GAT fusion

**Structure per Leaf**:

- Connected to **one root node** (the triple's head)
- Connected to **all other leaves** of the same root
- Carries **relation embedding** (via H-GAT)

**Example**:

```
Triple: <diabetes mellitus, isa, disease>
Root: "diabetes" (token position 0)
Leaves for root 0: [disease, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]
                    └─ from KG triple
```

---

## Graph Connectivity

### Edge Types

#### 1. Root-to-Leaf Edges

**Purpose**: Connect syntactic heads to semantic tails

**Structure**:

- Each root `r_i` connects to its 7 leaves: `{l_{i,1}, l_{i,2}, ..., l_{i,7}}`
- Edge carries semantic relation `rel` (encoded via H-GAT)
- Undirected edge

**Example**:

```
Root: "diabetes" (r_0)
  ├─ Leaf 0: "disease" (relation: isa)
  ├─ Leaf 1: <pad>
  ├─ Leaf 2: <pad>
  └─ ... (all 7 leaves connected)
```

#### 2. Leaf-to-Leaf Edges

**Purpose**: Connect leaves of the same root

**Structure**:

- All leaves of root `r_i` are pairwise connected
- Forms a **clique** of size 7 within each root's leaves
- Shortest path between any two leaves: 1

**Rationale**: Enables information flow between different semantic tokens associated with the same syntactic head

**Example**:

```
Root: "diabetes" has leaves [disease, syndrome, <pad>, ...]
Edges: disease ↔ syndrome, disease ↔ <pad>, syndrome ↔ <pad>, ...
```

### Shortest Path Matrix

For attention decay mask computation, we need shortest paths `sp(i,j)`:

**Within same root's leaves**: `sp(l_i, l_j) = 1` (direct connection)
**Between root and its leaf**: `sp(r_i, l_{i,j}) = 1` (direct connection)
**Between different roots**: `sp(r_i, r_j) = |i - j|` (chain distance)
**Between root and other root's leaf**: `sp(r_i, l_{j,k}) = |i - j| + 1`
**Between leaves of different roots**: `sp(l_{i,k}, l_{j,m}) = |i - j| + 2`

**Algorithm**: Floyd-Warshall on the adjacency matrix (precomputed once)

---

## Sequential Encoding

### From Graph to Sequence

The leafy chain graph must be encoded as a **1D sequence** for transformer input.

**Encoding Scheme** (from Figure 6 in paper):

```
Position:  [0     ...     127][128   ...   1023]
Content:   [Root  nodes   ]  [Leaf   nodes     ]
           [r₀ r₁ r₂ ... r₁₂₇][l₀₀...l₀₆ l₁₀...l₁₆ ... l₁₂₇₀...l₁₂₇₆]
```

**Structure**:

- First 128 positions: Root tokens (syntactic)
- Next 896 positions: Leaf tokens (7 leaves × 128 roots)
- Each root's 7 leaves are consecutive

**Leaf Indexing**:

```
Leaf l_{i,j} is at position: 128 + (i × 7) + j

Where:
  i = root index (0 to 127)
  j = leaf index within root (0 to 6)
```

### Padding

**Leaf Padding**:

- Empty leaves use `<pad>` token (token_id = 0 or specific pad_id)
- Most leaves are `<pad>` (sparse injection)
- Pad tokens ignored in loss calculation

**Sequence Padding**:

- If text has < 128 tokens, pad roots to 128
- Always maintain fixed 1024-length sequence

---

## Data Structures (Julia)

### Complete Type Definitions

```julia
"""
    ChainGraphNode

Represents a single node in the leafy chain graph.
"""
struct ChainGraphNode
    # Node identification
    id::Int                      # Global node ID (0-1023)
    node_type::Symbol            # :root or :leaf

    # Position in structure
    root_index::Int              # Which root this belongs to (0-127)
    leaf_index::Union{Int,Nothing}  # If leaf, which position (0-6); nothing if root

    # Token information
    token_id::Int                # Vocabulary token ID
    token_text::String           # Original text (for debugging)
    is_padding::Bool             # Whether this is a <pad> token

    # Semantic information (only for leaves)
    relation::Union{Symbol,Nothing}  # e.g., :isa, :associated_with
    head_text::Union{String,Nothing} # Text of the head entity

    # Embedding info (filled during forward pass)
    embedding::Union{Vector{Float32},Nothing}
end

"""
    LeafyChainGraph

Complete leafy chain graph structure for GraphMERT.
"""
struct LeafyChainGraph
    # Graph structure
    nodes::Vector{ChainGraphNode}     # All 1024 nodes
    adjacency_matrix::SparseMatrixCSC{Float32}  # 1024×1024 adjacency
    shortest_paths::Matrix{Int}       # 1024×1024 shortest path distances

    # Root information (syntactic space)
    root_tokens::Vector{Int}          # 128 token IDs
    root_texts::Vector{String}        # Original text tokens

    # Leaf information (semantic space)
    leaf_tokens::Matrix{Int}          # 128×7 matrix of token IDs
    leaf_relations::Matrix{Union{Symbol,Nothing}}  # 128×7 relations
    injected_mask::Matrix{Bool}       # 128×7 which leaves have injections

    # Metadata
    source_sequence_id::String        # Identifier for source text
    sequence_length::Int              # Original text length (≤128)
    num_injections::Int               # Count of non-padding leaves

    # Configuration
    config::ChainGraphConfig
end

"""
    ChainGraphConfig

Configuration for leafy chain graph construction.
"""
struct ChainGraphConfig
    num_roots::Int                    # Fixed: 128
    num_leaves_per_root::Int          # Fixed: 7
    max_sequence_length::Int          # Fixed: 1024
    pad_token_id::Int                 # Usually 0 or 1
    vocab_size::Int                   # e.g., 30522 for BioMedBERT
end

# Default configuration matching paper
function default_chain_graph_config()
    return ChainGraphConfig(
        num_roots = 128,
        num_leaves_per_root = 7,
        max_sequence_length = 1024,
        pad_token_id = 0,
        vocab_size = 30522
    )
end
```

---

## Construction Algorithms

### Algorithm 1: Create Empty Chain Graph

**Input**: Text tokens (≤128), configuration
**Output**: Chain graph with empty leaves

```julia
"""
    create_empty_chain_graph(tokens::Vector{Int}, config::ChainGraphConfig)

Create a leafy chain graph with all leaves set to <pad>.
"""
function create_empty_chain_graph(
    tokens::Vector{Int},
    token_texts::Vector{String},
    config::ChainGraphConfig
)::LeafyChainGraph

    @assert length(tokens) <= config.num_roots "Too many tokens"

    # Pad tokens to num_roots
    padded_tokens = vcat(tokens, fill(config.pad_token_id, config.num_roots - length(tokens)))
    padded_texts = vcat(token_texts, fill("<pad>", config.num_roots - length(token_texts)))

    # Initialize all leaves as padding
    leaf_tokens = fill(config.pad_token_id, config.num_roots, config.num_leaves_per_root)
    leaf_relations = fill(nothing, config.num_roots, config.num_leaves_per_root)
    injected_mask = falses(config.num_roots, config.num_leaves_per_root)

    # Create nodes
    nodes = Vector{ChainGraphNode}(undef, config.max_sequence_length)

    # Create root nodes
    for i in 1:config.num_roots
        nodes[i] = ChainGraphNode(
            id = i - 1,  # 0-indexed
            node_type = :root,
            root_index = i - 1,
            leaf_index = nothing,
            token_id = padded_tokens[i],
            token_text = padded_texts[i],
            is_padding = (padded_tokens[i] == config.pad_token_id),
            relation = nothing,
            head_text = nothing,
            embedding = nothing
        )
    end

    # Create leaf nodes (all padding initially)
    for i in 1:config.num_roots
        for j in 1:config.num_leaves_per_root
            node_id = config.num_roots + (i-1) * config.num_leaves_per_root + j
            nodes[node_id] = ChainGraphNode(
                id = node_id - 1,
                node_type = :leaf,
                root_index = i - 1,
                leaf_index = j - 1,
                token_id = config.pad_token_id,
                token_text = "<pad>",
                is_padding = true,
                relation = nothing,
                head_text = nothing,
                embedding = nothing
            )
        end
    end

    # Build adjacency matrix
    adj = build_adjacency_matrix(config)

    # Compute shortest paths
    sp = floyd_warshall(adj)

    return LeafyChainGraph(
        nodes = nodes,
        adjacency_matrix = adj,
        shortest_paths = sp,
        root_tokens = padded_tokens,
        root_texts = padded_texts,
        leaf_tokens = leaf_tokens,
        leaf_relations = leaf_relations,
        injected_mask = injected_mask,
        source_sequence_id = "",
        sequence_length = length(tokens),
        num_injections = 0,
        config = config
    )
end
```

### Algorithm 2: Build Adjacency Matrix

**Input**: Configuration
**Output**: Sparse adjacency matrix (1024×1024)

```julia
"""
    build_adjacency_matrix(config::ChainGraphConfig)

Build the fixed adjacency matrix for leafy chain graph structure.
"""
function build_adjacency_matrix(config::ChainGraphConfig)::SparseMatrixCSC{Float32}
    N = config.max_sequence_length
    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float32[]

    # Helper to add undirected edge
    function add_edge(i, j)
        push!(I_vals, i)
        push!(J_vals, j)
        push!(V_vals, 1.0f0)
        push!(I_vals, j)
        push!(J_vals, i)
        push!(V_vals, 1.0f0)
    end

    # 1. Connect consecutive roots (chain structure)
    for i in 1:(config.num_roots - 1)
        add_edge(i, i + 1)
    end

    # 2. Connect each root to its leaves
    for root_idx in 1:config.num_roots
        for leaf_idx in 1:config.num_leaves_per_root
            leaf_id = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + leaf_idx
            add_edge(root_idx, leaf_id)
        end
    end

    # 3. Connect all leaves of same root (clique)
    for root_idx in 1:config.num_roots
        leaves = [
            config.num_roots + (root_idx - 1) * config.num_leaves_per_root + j
            for j in 1:config.num_leaves_per_root
        ]
        for i in 1:length(leaves)
            for j in (i+1):length(leaves)
                add_edge(leaves[i], leaves[j])
            end
        end
    end

    # 4. Add self-loops
    for i in 1:N
        push!(I_vals, i)
        push!(J_vals, i)
        push!(V_vals, 1.0f0)
    end

    return sparse(I_vals, J_vals, V_vals, N, N)
end
```

### Algorithm 3: Inject Triple into Graph

**Input**: Graph, root_index, leaf_index, triple tokens, relation
**Output**: Modified graph with injection

```julia
"""
    inject_triple!(
        graph::LeafyChainGraph,
        root_index::Int,
        leaf_start_index::Int,
        tail_tokens::Vector{Int},
        tail_text::String,
        relation::Symbol,
        head_text::String
    )

Inject a semantic triple into the graph at specified leaf position.
"""
function inject_triple!(
    graph::LeafyChainGraph,
    root_index::Int,          # Which root (0-127)
    leaf_start_index::Int,    # Which leaf group (0-6)
    tail_tokens::Vector{Int}, # Tokens for the tail (≤7)
    tail_text::String,
    relation::Symbol,
    head_text::String
)
    config = graph.config

    @assert 0 <= root_index < config.num_roots "Invalid root index"
    @assert 0 <= leaf_start_index < config.num_leaves_per_root "Invalid leaf index"
    @assert length(tail_tokens) <= config.num_leaves_per_root "Too many tail tokens"

    # Pad tail tokens to num_leaves_per_root if needed
    padded_tail = vcat(tail_tokens, fill(config.pad_token_id, config.num_leaves_per_root - length(tail_tokens)))

    # Update leaf tokens matrix
    graph.leaf_tokens[root_index + 1, :] .= padded_tail

    # Update relation for all leaves of this root
    graph.leaf_relations[root_index + 1, :] .= relation

    # Update injected mask
    for i in 1:length(tail_tokens)
        graph.injected_mask[root_index + 1, i] = true
    end

    # Update node objects
    for j in 1:config.num_leaves_per_root
        node_id = config.num_roots + root_index * config.num_leaves_per_root + j
        is_injected = j <= length(tail_tokens)

        graph.nodes[node_id] = ChainGraphNode(
            id = node_id - 1,
            node_type = :leaf,
            root_index = root_index,
            leaf_index = j - 1,
            token_id = padded_tail[j],
            token_text = is_injected ? tail_text : "<pad>",
            is_padding = !is_injected,
            relation = is_injected ? relation : nothing,
            head_text = is_injected ? head_text : nothing,
            embedding = nothing
        )
    end

    # Update injection count
    graph.num_injections += 1

    return graph
end
```

### Algorithm 4: Floyd-Warshall Shortest Paths

```julia
"""
    floyd_warshall(adj::SparseMatrixCSC{Float32})

Compute all-pairs shortest paths using Floyd-Warshall algorithm.
"""
function floyd_warshall(adj::SparseMatrixCSC{Float32})::Matrix{Int}
    N = size(adj, 1)
    dist = fill(typemax(Int) ÷ 2, N, N)  # Initialize with "infinity"

    # Initialize with direct edges
    rows, cols, _ = findnz(adj)
    for (i, j) in zip(rows, cols)
        if i != j
            dist[i, j] = 1
        else
            dist[i, j] = 0
        end
    end

    # Floyd-Warshall algorithm
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

---

## Sequential Encoding for Transformer

### Algorithm 5: Convert Graph to Token Sequence

```julia
"""
    graph_to_sequence(graph::LeafyChainGraph)

Convert leafy chain graph to 1D token sequence for transformer input.
"""
function graph_to_sequence(graph::LeafyChainGraph)::Vector{Int}
    config = graph.config
    sequence = zeros(Int, config.max_sequence_length)

    # First 128 positions: root tokens
    sequence[1:config.num_roots] = graph.root_tokens

    # Next 896 positions: leaf tokens (row-major order)
    leaf_start = config.num_roots + 1
    for i in 1:config.num_roots
        for j in 1:config.num_leaves_per_root
            pos = leaf_start + (i - 1) * config.num_leaves_per_root + (j - 1)
            sequence[pos] = graph.leaf_tokens[i, j]
        end
    end

    return sequence
end
```

### Algorithm 6: Create Position IDs

```julia
"""
    create_position_ids(graph::LeafyChainGraph)

Create position IDs for the sequence (0-indexed).
"""
function create_position_ids(graph::LeafyChainGraph)::Vector{Int}
    return collect(0:(graph.config.max_sequence_length - 1))
end
```

### Algorithm 7: Create Attention Mask

```julia
"""
    create_attention_mask(graph::LeafyChainGraph)

Create attention mask (1 for real tokens, 0 for padding).
"""
function create_attention_mask(graph::LeafyChainGraph)::Vector{Int}
    config = graph.config
    mask = ones(Int, config.max_sequence_length)

    # Mask padding in roots
    for i in 1:config.num_roots
        if graph.root_tokens[i] == config.pad_token_id
            mask[i] = 0
        end
    end

    # Mask padding in leaves
    leaf_start = config.num_roots
    for i in 1:config.num_roots
        for j in 1:config.num_leaves_per_root
            if !graph.injected_mask[i, j]
                pos = leaf_start + (i - 1) * config.num_leaves_per_root + j
                mask[pos] = 0
            end
        end
    end

    return mask
end
```

---

## Integration with H-GAT

### Relation Encoding

For each injected triple, H-GAT fuses the relation and head information into the leaf embeddings.

**Per-Leaf Fusion** (from H-GAT document):

```julia
function fuse_leaf_with_relation(
    leaf_embedding::Vector{Float32},
    head_embeddings::Vector{Vector{Float32}},
    relation_embedding_matrix::Matrix{Float32},
    relation_embedding_vector::Vector{Float32}
)
    # This is handled by H-GAT component
    # See Document 04 for details
    return fused_embedding
end
```

**Integration Point**: After getting token embeddings from RoBERTa embedding layer, H-GAT processes all injected leaves.

---

## Worked Example

### Example: Diabetes Abstract

**Input Text** (simplified):

```
"Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia."
```

**Tokenized** (BioMedBERT):

```
[2156, 23421, 16, 10, 6844, 19131, 8761, 12890, 34, 25678]
["diabetes", "mellitus", "is", "a", "chronic", "metabolic", "disorder", "characterized", "by", "hyperglycemia"]
```

**Seed KG Triples** (from UMLS):

```
1. <diabetes mellitus, isa, disease>
2. <diabetes mellitus, associated_with, hyperglycemia>
3. <hyperglycemia, finding_site_of, blood>
```

### Step-by-Step Construction

**Step 1**: Create empty graph

- Roots: [diabetes, mellitus, is, a, chronic, metabolic, disorder, characterized, by, hyperglycemia, <pad>, ..., <pad>]
- All leaves: <pad>

**Step 2**: Inject triple 1

- Find "diabetes" at root index 0
- Inject "disease" into leaf 0 of root 0
- Relation: :isa

**Step 3**: Inject triple 2

- Find "diabetes" at root index 0
- Inject "hyperglycemia" into leaf 1 of root 0
- Relation: :associated_with

**Step 4**: Inject triple 3

- Find "hyperglycemia" at root index 9
- Inject "blood" into leaf 0 of root 9
- Relation: :finding_site_of

**Final Graph Structure**:

```
Roots (first 10 shown):
[0]: diabetes      → leaves: [disease, hyperglycemia, <pad>, <pad>, <pad>, <pad>, <pad>]
[1]: mellitus      → leaves: [<pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]
[2]: is            → leaves: [<pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]
...
[9]: hyperglycemia → leaves: [blood, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]
[10-127]: <pad>    → leaves: [<pad>, ...]
```

**Encoded Sequence** (positions 0-1023):

```
[0-127]:    [diabetes, mellitus, is, ..., hyperglycemia, <pad>, ..., <pad>]
[128-134]:  [disease, hyperglycemia, <pad>, ..., <pad>]  ← leaves of root 0
[135-141]:  [<pad>, <pad>, ..., <pad>]                   ← leaves of root 1
...
[191-197]:  [blood, <pad>, ..., <pad>]                   ← leaves of root 9
```

---

## Validation and Testing

### Test Cases

```julia
@testset "Leafy Chain Graph Construction" begin
    config = default_chain_graph_config()

    @testset "Empty Graph Creation" begin
        tokens = [1, 2, 3, 4, 5]
        texts = ["a", "b", "c", "d", "e"]
        graph = create_empty_chain_graph(tokens, texts, config)

        @test length(graph.nodes) == 1024
        @test length(graph.root_tokens) == 128
        @test size(graph.leaf_tokens) == (128, 7)
        @test graph.num_injections == 0
        @test all(graph.injected_mask .== false)
    end

    @testset "Triple Injection" begin
        graph = create_empty_chain_graph([1, 2, 3], ["diabetes", "is", "disease"], config)
        inject_triple!(graph, 0, 0, [100, 101], "chronic disease", :isa, "diabetes")

        @test graph.num_injections == 1
        @test graph.leaf_tokens[1, 1] == 100
        @test graph.leaf_tokens[1, 2] == 101
        @test graph.injected_mask[1, 1] == true
        @test graph.leaf_relations[1, 1] == :isa
    end

    @testset "Sequential Encoding" begin
        graph = create_empty_chain_graph([1, 2, 3], ["a", "b", "c"], config)
        inject_triple!(graph, 0, 0, [100], "tail", :rel, "head")

        seq = graph_to_sequence(graph)
        @test length(seq) == 1024
        @test seq[1:3] == [1, 2, 3]
        @test seq[129] == 100  # First leaf of root 0
    end

    @testset "Shortest Paths" begin
        graph = create_empty_chain_graph([1, 2, 3], ["a", "b", "c"], config)
        sp = graph.shortest_paths

        # Root to itself
        @test sp[1, 1] == 0

        # Adjacent roots
        @test sp[1, 2] == 1

        # Root to its leaf
        @test sp[1, 129] == 1  # root 0 to its first leaf

        # Leaves of same root
        @test sp[129, 130] == 1  # two leaves of root 0
    end
end
```

---

## Performance Considerations

### Memory

**Storage**:

- Nodes: 1024 × sizeof(ChainGraphNode) ≈ 50KB per graph
- Adjacency matrix: sparse, ~10K non-zero entries ≈ 80KB
- Shortest paths: 1024² × sizeof(Int) = 4MB (can be precomputed and shared)

**Optimization**:

- Precompute adjacency and shortest paths for the fixed structure
- Share across all graphs in a batch
- Use sparse matrices for adjacency

### Computation

**Bottlenecks**:

- Floyd-Warshall: O(N³) = O(1024³) ≈ 1B operations (one-time)
- Adjacency construction: O(N²) for dense, O(E) for sparse

**Optimizations**:

- Cache shortest paths globally (structure is fixed)
- Construct adjacency once, reuse for all graphs
- Use sparse matrix operations

---

## Integration Points

### With RoBERTa (Document 03)

- Graph → sequence conversion
- Token IDs input to embedding layer
- Position IDs for positional encoding

### With H-GAT (Document 04)

- Identify injected leaves for relation fusion
- Provide head token indices for attention
- Apply relation-specific transformations

### With MLM Training (Document 06)

- Mask root tokens (syntactic space)
- Compute MLM loss on masked roots
- Standard span masking applies

### With MNM Training (Document 07)

- Mask leaf tokens (semantic space)
- Compute MNM loss on masked leaves
- Entire leaf span masked together

### With Seed Injection (Document 08)

- Identify injection points (which roots match heads)
- Insert triples using `inject_triple!`
- Maintain diversity and relevance constraints

---

## Implementation Checklist

- [ ] Define `ChainGraphNode` type
- [ ] Define `LeafyChainGraph` type
- [ ] Define `ChainGraphConfig` type
- [ ] Implement `create_empty_chain_graph()`
- [ ] Implement `build_adjacency_matrix()`
- [ ] Implement `floyd_warshall()`
- [ ] Implement `inject_triple!()`
- [ ] Implement `graph_to_sequence()`
- [ ] Implement `create_position_ids()`
- [ ] Implement `create_attention_mask()`
- [ ] Write comprehensive unit tests
- [ ] Write integration tests with H-GAT
- [ ] Profile and optimize memory usage
- [ ] Document with examples

---

## Next Steps

1. **Implement** this specification in `graphs/leafy_chain.jl`
2. **Test** thoroughly with small examples
3. **Integrate** with existing H-GAT component
4. **Validate** graph construction on sample texts
5. **Proceed** to MNM training (Document 07)

---

**Related Documents**:

- → [Doc 04: H-GAT Component](04-hgat-component.md) - Relation fusion for leaves
- → [Doc 07: MNM Training](07-training-mnm.md) - Semantic space training
- → [Doc 08: Seed Injection](08-seed-kg-injection.md) - Triple injection algorithm
