"""
Leafy chain graph structure for GraphMERT.jl

This module implements the leafy chain graph structure used for
representing text with semantic nodes in GraphMERT.

The leafy chain graph combines syntactic (text tokens) and semantic (KG triples)
representations in a unified structure that enables joint training.
"""

# Import from main module (will be available after types.jl is included)

# using DocStringExtensions  # Temporarily disabled

"""
    create_empty_chain_graph(tokens::Vector{Int}, token_texts::Vector{String}, config::ChainGraphConfig)

Create a leafy chain graph with all leaves set to <pad>.

"""
function create_empty_chain_graph(
    tokens::Vector{Int},
    token_texts::Vector{String},
    config::ChainGraphConfig,
)::LeafyChainGraph

    @assert length(tokens) <= config.num_roots "Too many tokens"

    # Pad tokens to num_roots
    padded_tokens = vcat(tokens, fill(config.pad_token_id, config.num_roots - length(tokens)))
    padded_texts = vcat(token_texts, fill("<pad>", config.num_roots - length(token_texts)))

    # Initialize all leaves as padding
    leaf_tokens = fill(config.pad_token_id, config.num_roots, config.num_leaves_per_root)
    leaf_relations = Matrix{Union{Symbol, Nothing}}(fill(nothing, config.num_roots, config.num_leaves_per_root))
    injected_mask = Matrix{Bool}(falses(config.num_roots, config.num_leaves_per_root))

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
            embedding = nothing,
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
                embedding = nothing,
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
        config = config,
    )
end

"""
    build_adjacency_matrix(config::ChainGraphConfig)

Build the fixed adjacency matrix for leafy chain graph structure.

"""
function build_adjacency_matrix(config::ChainGraphConfig)::SparseMatrixCSC{Float32}
    N = config.max_sequence_length
    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float32[]

    # Helper to add directed edge (following paper: no undirected, no self-loops)
    function add_edge(i, j)
        push!(I_vals, i)
        push!(J_vals, j)
        push!(V_vals, 1.0f0)
    end

    # 1. Connect consecutive roots (chain structure)
    for i in 1:(config.num_roots-1)
        add_edge(i, i + 1)
    end

    # 2. Connect each root to its leaves (star structure)
    for root_idx in 1:config.num_roots
        for leaf_idx in 1:config.num_leaves_per_root
            leaf_id = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + leaf_idx
            add_edge(root_idx, leaf_id)
        end
    end

    # Note: No leaf-to-leaf connections, no self-loops (following paper specification)

    return sparse(I_vals, J_vals, V_vals, N, N)
end

"""
    floyd_warshall(adj::SparseMatrixCSC{Float32})

Compute all-pairs shortest paths using Floyd-Warshall algorithm.

"""
function floyd_warshall(adj::SparseMatrixCSC{Float32})::Matrix{Int}
    N = size(adj, 1)
    dist = fill(typemax(Int) ÷ 2, N, N)  # Initialize with "infinity"

    # Set self-distances to 0
    for i in 1:N
        dist[i, i] = 0
    end

    # Initialize with direct edges
    rows, cols, _ = findnz(adj)
    for (i, j) in zip(rows, cols)
        dist[i, j] = 1  # Direct edge distance
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

"""
    inject_triple!(graph::LeafyChainGraph, root_index::Int, leaf_start_index::Int, tail_tokens::Vector{Int}, tail_text::String, relation::Symbol, head_text::String)

Inject a semantic triple into the graph at specified leaf position.

"""
function inject_triple!(
    graph::LeafyChainGraph,
    root_index::Int,          # Which root (0-127)
    leaf_start_index::Int,    # Which leaf group (0-6)
    tail_tokens::Vector{Int}, # Tokens for the tail (≤7)
    tail_text::String,
    relation::Symbol,
    head_text::String,
)
    config = graph.config

    @assert 0 <= root_index < config.num_roots "Invalid root index"
    @assert 0 <= leaf_start_index < config.num_leaves_per_root "Invalid leaf index"
    @assert length(tail_tokens) <= config.num_leaves_per_root "Too many tail tokens"

    # Pad tail tokens to num_leaves_per_root if needed
    padded_tail = vcat(tail_tokens, fill(config.pad_token_id, config.num_leaves_per_root - length(tail_tokens)))

    # Update leaf tokens matrix
    graph.leaf_tokens[root_index+1, :] .= padded_tail

    # Update relation for all leaves of this root
    graph.leaf_relations[root_index+1, :] .= relation

    # Update injected mask
    for i in 1:length(tail_tokens)
        graph.injected_mask[root_index+1, i] = true
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
            embedding = nothing,
        )
    end

    # Update injection count
    graph.num_injections += 1

    return graph
end

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

"""
    create_position_ids(graph::LeafyChainGraph)

Create position IDs for the sequence (0-indexed).

"""
function create_position_ids(graph::LeafyChainGraph)::Vector{Int}
    return collect(0:(graph.config.max_sequence_length-1))
end

"""
    validate_chain_graph(graph::LeafyChainGraph) -> Bool

Validate the structural integrity of a leafy chain graph.

Checks:
- Correct number of nodes
- Adjacency matrix dimensions
- Root chain connectivity
- Root-leaf star connectivity
- No self-loops
- No leaf-to-leaf connections

# Arguments
- `graph`: LeafyChainGraph to validate

# Returns
- true if valid, throws AssertionError if invalid
"""
function validate_chain_graph(graph::LeafyChainGraph)::Bool
    config = graph.config
    total_nodes = config.num_roots * (1 + config.num_leaves_per_root)

    # Check node count
    @assert length(graph.nodes) == config.max_sequence_length "Incorrect number of nodes"

    # Check adjacency matrix
    @assert size(graph.adjacency_matrix) == (config.max_sequence_length, config.max_sequence_length) "Incorrect adjacency matrix size"

    # Check root nodes
    for i in 1:config.num_roots
        node = graph.nodes[i]
        @assert node.node_type == :root "Node $i should be root"
        @assert node.root_index == i-1 "Root index mismatch for node $i"
        @assert isnothing(node.leaf_index) "Root node should not have leaf_index"
    end

    # Check leaf nodes
    for i in (config.num_roots+1):config.max_sequence_length
        node = graph.nodes[i]
        @assert node.node_type == :leaf "Node $i should be leaf"
        @assert !isnothing(node.leaf_index) "Leaf node should have leaf_index"
        @assert 0 <= node.leaf_index < config.num_leaves_per_root "Invalid leaf_index"
    end

    # Check no self-loops
    @assert all(diag(graph.adjacency_matrix) .== 0) "Self-loops detected"

    # Check connectivity patterns
    adj = graph.adjacency_matrix

    # Root chain: each root (except last) should connect to next root
    for root_idx in 1:(config.num_roots-1)
        @assert adj[root_idx, root_idx+1] > 0 "Missing root chain connection: $root_idx → $(root_idx+1)"
    end

    # Root stars: each root should connect to exactly num_leaves_per_root leaves
    for root_idx in 1:config.num_roots
        leaf_start = config.num_roots + (root_idx-1)*config.num_leaves_per_root + 1
        leaf_end = config.num_roots + root_idx*config.num_leaves_per_root

        connected_leaves = sum(adj[root_idx, leaf_start:leaf_end])
        @assert connected_leaves == config.num_leaves_per_root "Root $root_idx should connect to $(config.num_leaves_per_root) leaves, found $connected_leaves"
    end

    # No leaf-to-leaf connections
    leaf_start = config.num_roots + 1
    @assert all(adj[leaf_start:end, leaf_start:end] .== 0) "Leaf-to-leaf connections detected"

    return true
end

# ============================================================================
# Graph Manipulation Utilities
# ============================================================================

"""
    get_root_node_indices(config::ChainGraphConfig) -> UnitRange{Int}

Get the 1-based indices of root nodes in the graph.
"""
function get_root_node_indices(config::ChainGraphConfig)::UnitRange{Int}
    return 1:config.num_roots
end

"""
    get_leaf_node_indices(config::ChainGraphConfig, root_idx::Int) -> UnitRange{Int}

Get the 1-based indices of leaf nodes for a specific root.
"""
function get_leaf_node_indices(config::ChainGraphConfig, root_idx::Int)::UnitRange{Int}
    @assert 1 <= root_idx <= config.num_roots "Invalid root index"
    leaf_start = config.num_roots + (root_idx-1)*config.num_leaves_per_root + 1
    leaf_end = config.num_roots + root_idx*config.num_leaves_per_root
    return leaf_start:leaf_end
end

"""
    get_node_index(config::ChainGraphConfig, root_idx::Int, leaf_idx::Union{Int,Nothing}=nothing) -> Int

Get the 1-based node index for a root or leaf position.
"""
function get_node_index(config::ChainGraphConfig, root_idx::Int, leaf_idx::Union{Int,Nothing}=nothing)::Int
    @assert 1 <= root_idx <= config.num_roots "Invalid root index"
    if isnothing(leaf_idx)
        return root_idx  # Root node
    else
        @assert 0 <= leaf_idx < config.num_leaves_per_root "Invalid leaf index"
        return config.num_roots + (root_idx-1)*config.num_leaves_per_root + leaf_idx + 1
    end
end

"""
    create_empty_chain_graph(config::ChainGraphConfig) -> LeafyChainGraph

Create an empty leafy chain graph with the specified configuration.

The graph consists of:
- 128 roots arranged in a chain structure
- Each root has 7 leaves arranged in a star structure
- Total: 1024 nodes (128 roots + 896 leaves)

# Arguments
- `config`: ChainGraphConfig with graph parameters

# Returns
- Empty LeafyChainGraph ready for triple injection
"""
function create_empty_chain_graph(config::ChainGraphConfig)::LeafyChainGraph
    @assert config.num_roots > 0 "num_roots must be positive"
    @assert config.num_leaves_per_root > 0 "num_leaves_per_root must be positive"

    total_nodes = config.num_roots * (1 + config.num_leaves_per_root)
    @assert total_nodes <= config.max_sequence_length "total nodes exceed max_sequence_length"

    # Create nodes
    nodes = Vector{ChainGraphNode}(undef, config.max_sequence_length)
    node_idx = 1

    # Create root nodes (0 to num_roots-1)
    for root_idx in 0:(config.num_roots-1)
        nodes[node_idx] = ChainGraphNode(
            id = node_idx - 1,  # 0-based indexing
            node_type = :root,
            root_index = root_idx,
            token_id = config.pad_token_id,  # Initially padding
            token_text = "<pad>",
            is_padding = true
        )
        node_idx += 1
    end

    # Create leaf nodes (num_roots to total_nodes-1)
    for root_idx in 0:(config.num_roots-1)
        for leaf_idx in 0:(config.num_leaves_per_root-1)
            nodes[node_idx] = ChainGraphNode(
                id = node_idx - 1,  # 0-based indexing
                node_type = :leaf,
                root_index = root_idx,
                leaf_index = leaf_idx,
                token_id = config.pad_token_id,  # Initially padding
                token_text = "<pad>",
                is_padding = true
            )
            node_idx += 1
        end
    end

    # Build adjacency matrix
    adj_matrix = build_adjacency_matrix(config)

    # Initialize shortest paths (will be computed later)
    shortest_paths = floyd_warshall(adj_matrix)

    # Initialize arrays
    root_tokens = fill(config.pad_token_id, config.num_roots)
    root_texts = fill("<pad>", config.num_roots)
    leaf_tokens = fill(config.pad_token_id, (config.num_roots, config.num_leaves_per_root))
    leaf_relations = Matrix{Union{Symbol,Nothing}}(fill(nothing, (config.num_roots, config.num_leaves_per_root)))
    injected_mask = Matrix{Bool}(falses(config.num_roots, config.num_leaves_per_root))

    return LeafyChainGraph(
        nodes = nodes,
        adjacency_matrix = adj_matrix,
        shortest_paths = shortest_paths,
        root_tokens = root_tokens,
        root_texts = root_texts,
        leaf_tokens = leaf_tokens,
        leaf_relations = leaf_relations,
        injected_mask = injected_mask,
        source_sequence_id = "",
        sequence_length = 0,
        num_injections = 0,
        config = config
    )
end

# Export functions for external use
export default_chain_graph_config,
    create_empty_chain_graph,
    build_adjacency_matrix,
    floyd_warshall,
    inject_triple!,
    graph_to_sequence,
    create_attention_mask,
    create_position_ids,
    validate_chain_graph,
    get_root_node_indices,
    get_leaf_node_indices,
    get_node_index
