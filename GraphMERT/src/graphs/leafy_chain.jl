"""
Leafy chain graph structure for GraphMERT.jl

This module implements the leafy chain graph structure used for
representing text with semantic nodes in GraphMERT.

The leafy chain graph combines syntactic (text tokens) and semantic (KG triples)
representations in a unified structure that enables joint training.
"""

# Import from main module (will be available after types.jl is included)

"""
    default_chain_graph_config(vocab_size::Int=30522)

Create default configuration for chain graph construction.

# Arguments
- `vocab_size::Int`: Size of token vocabulary (default: 30522 for BioMedBERT)

# Returns
- `ChainGraphConfig`: Default configuration with 128 roots, 7 leaves per root
"""
function default_chain_graph_config(vocab_size::Int = 30522)
    return ChainGraphConfig(
        128,        # num_roots
        7,          # num_leaves_per_root
        vocab_size,
        1,          # pad_token_id
        103,        # mask_token_id
        true,        # precompute_shortest_paths
    )
end

"""
    create_empty_chain_graph(config::ChainGraphConfig=default_chain_graph_config())

Create an empty leafy chain graph with proper structure but no content.

# Arguments
- `config::ChainGraphConfig`: Configuration for graph structure

# Returns
- `LeafyChainGraph`: Empty graph with correct dimensions
"""
function create_empty_chain_graph(config = default_chain_graph_config())
    # Create root nodes (all padding initially)
    root_nodes = [
        ChainGraphNode(:root, config.pad_token_id, i, nothing, true) for
        i = 1:config.num_roots
    ]

    # Create leaf nodes (all padding initially, organized by root)
    leaf_nodes = [
        [
            ChainGraphNode(
                :leaf,
                config.pad_token_id,
                config.num_roots + (i - 1) * (config.num_leaves_per_root) + j,
                i,
                true,
            ) for j = 1:config.num_leaves_per_root
        ] for i = 1:config.num_roots
    ]

    # Create adjacency matrix (all false initially)
    total_nodes = config.num_roots * (1 + config.num_leaves_per_root)
    adjacency_matrix = Matrix{Bool}(falses(total_nodes, total_nodes))

    # Connect roots to their leaves
    for root_idx = 1:config.num_roots
        root_pos = root_idx
        for leaf_idx = 1:config.num_leaves_per_root
            leaf_pos =
                config.num_roots + (root_idx - 1) * config.num_leaves_per_root + leaf_idx
            adjacency_matrix[root_pos, leaf_pos] = true
            adjacency_matrix[leaf_pos, root_pos] = true
        end
    end

    # Connect leaves within same root group
    for root_idx = 1:config.num_roots
        for i = 1:config.num_leaves_per_root
            for j = (i+1):config.num_leaves_per_root
                leaf_pos_i =
                    config.num_roots + (root_idx - 1) * config.num_leaves_per_root + i
                leaf_pos_j =
                    config.num_roots + (root_idx - 1) * config.num_leaves_per_root + j
                adjacency_matrix[leaf_pos_i, leaf_pos_j] = true
                adjacency_matrix[leaf_pos_j, leaf_pos_i] = true
            end
        end
    end

    return LeafyChainGraph(root_nodes, leaf_nodes, adjacency_matrix, config)
end

"""
    build_adjacency_matrix(graph::LeafyChainGraph)

Build adjacency matrix for graph connectivity (already implemented in constructor).

# Arguments
- `graph::LeafyChainGraph`: Graph to analyze

# Returns
- `Matrix{Bool}`: Adjacency matrix showing node connectivity
"""
function build_adjacency_matrix(graph::LeafyChainGraph)
    return graph.adjacency_matrix
end

"""
    floyd_warshall(graph::LeafyChainGraph)

Compute shortest paths between all pairs of nodes using Floyd-Warshall algorithm.

# Arguments
- `graph::LeafyChainGraph`: Graph to compute paths for

# Returns
- `Matrix{Int}`: Shortest path distances (0 for unreachable, ∞ for self)
"""
function floyd_warshall(graph::LeafyChainGraph)
    if !graph.config.precompute_shortest_paths
        return nothing
    end

    n = size(graph.adjacency_matrix, 1)
    dist = fill(typemax(Int), n, n)

    # Initialize distances
    for i = 1:n
        dist[i, i] = 0
        for j = 1:n
            if graph.adjacency_matrix[i, j]
                dist[i, j] = 1
            end
        end
    end

    # Floyd-Warshall algorithm
    for k = 1:n
        for i = 1:n
            for j = 1:n
                if dist[i, k] != typemax(Int) && dist[k, j] != typemax(Int)
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
                end
            end
        end
    end

    # Handle unreachable nodes (set to 0 for consistency with paper)
    dist[dist .== typemax(Int)] .= 0

    return dist
end

"""
    inject_triple!(graph::LeafyChainGraph, triple::SemanticTriple, root_idx::Int)

Inject a semantic triple into the graph as leaf nodes.

# Arguments
- `graph::LeafyChainGraph`: Graph to modify
- `triple::SemanticTriple`: Triple to inject
- `root_idx::Int`: Which root node to attach the triple to

# Returns
- `Bool`: Success status
"""
function inject_triple!(graph::LeafyChainGraph, triple::SemanticTriple, root_idx::Int)
    if root_idx < 1 || root_idx > graph.config.num_roots
        return false
    end

    if length(triple.tail_tokens) > graph.config.num_leaves_per_root
        return false  # Triple too long for available leaf slots
    end

    # Find available leaf slots for this root
    available_slots = findall(node -> node.is_padding, graph.leaf_nodes[root_idx])
    if length(available_slots) < length(triple.tail_tokens)
        return false  # Not enough space
    end

    # Inject tokens into leaf nodes
    for (i, token_id) in enumerate(triple.tail_tokens)
        slot_idx = available_slots[i]
        leaf_pos =
            graph.config.num_roots +
            (root_idx - 1) * graph.config.num_leaves_per_root +
            slot_idx

        # Update leaf node
        graph.leaf_nodes[root_idx][slot_idx] =
            ChainGraphNode(:leaf, token_id, leaf_pos, root_idx, false)

        # Update adjacency matrix to connect to this leaf
        graph.adjacency_matrix[root_idx, leaf_pos] = true
        graph.adjacency_matrix[leaf_pos, root_idx] = true
    end

    # Update shortest paths if precomputed
    if graph.config.precompute_shortest_paths && graph.shortest_paths !== nothing
        graph.shortest_paths = floyd_warshall(graph)
    end

    return true
end

"""
    graph_to_sequence(graph::LeafyChainGraph)

Convert graph to sequential representation for transformer input.

# Arguments
- `graph::LeafyChainGraph`: Graph to convert

# Returns
- `Vector{Int}`: Token sequence [r1, l1_1, l1_2, ..., l1_7, r2, l2_1, ..., r128, l128_7]
"""
function graph_to_sequence(graph::LeafyChainGraph)
    sequence = Vector{Int}()

    for root_idx = 1:graph.config.num_roots
        # Add root token
        push!(sequence, graph.root_nodes[root_idx].token_id)

        # Add leaf tokens for this root
        for leaf in graph.leaf_nodes[root_idx]
            push!(sequence, leaf.token_id)
        end
    end

    return sequence
end

"""
    create_attention_mask(graph::LeafyChainGraph)

Create attention mask based on graph structure and shortest paths.

# Arguments
- `graph::LeafyChainGraph`: Graph to create mask for

# Returns
- `Matrix{Bool}`: Attention mask (true where attention allowed)
"""
function create_attention_mask(graph::LeafyChainGraph)
    if graph.shortest_paths === nothing
        # Fallback: use adjacency matrix
        return copy(graph.adjacency_matrix)
    end

    n = size(graph.shortest_paths, 1)
    mask = copy(graph.adjacency_matrix)

    # Apply attention decay based on shortest paths
    for i = 1:n
        for j = 1:n
            if i != j && graph.shortest_paths[i, j] > 0
                # Apply exponential decay (simplified version)
                decay_factor = exp(-0.6 * sqrt(graph.shortest_paths[i, j]))
                if decay_factor < 0.1  # Threshold for attention cutoff
                    mask[i, j] = false
                    mask[j, i] = false
                end
            end
        end
    end

    return mask
end

"""
    create_leafy_chain_from_text(text::String, config::ChainGraphConfig=default_chain_graph_config())

Create a leafy chain graph from text tokens.

# Arguments
- `text::String`: Input text to convert
- `config::ChainGraphConfig`: Graph configuration

# Returns
- `LeafyChainGraph`: Graph with text tokens as root nodes
"""
function create_leafy_chain_from_text(
    text::String,
    config::ChainGraphConfig = default_chain_graph_config(),
)
    # Tokenize text (simplified - would use proper tokenizer)
    tokens = [Int(hash(c) & 0x7FFFFFFFFFFFFFFF) % config.vocab_size for c in text]  # Simple tokenization for demo

    # Create empty graph
    graph = create_empty_chain_graph(config)

    # Fill root nodes with tokens (pad if necessary)
    for i = 1:min(length(tokens), config.num_roots)
        graph.root_nodes[i] = ChainGraphNode(:root, tokens[i], i, nothing, false)
    end

    return graph
end

# Export functions for external use
export default_chain_graph_config,
    create_empty_chain_graph,
    build_adjacency_matrix,
    floyd_warshall,
    inject_triple!,
    graph_to_sequence,
    create_attention_mask,
    create_leafy_chain_from_text
