"""
Leafy chain graph structure for GraphMERT.jl

This module implements the leafy chain graph structure for text representation
as specified in the GraphMERT paper, with root and leaf nodes for semantic encoding.
"""

using LightGraphs
using MetaGraphs
using SparseArrays
using LinearAlgebra

# ============================================================================
# Leafy Chain Graph Types
# ============================================================================

"""
    LeafyChainGraph

Represents a leafy chain graph structure for text representation.
"""
struct LeafyChainGraph
  root_nodes::Vector{Int}
  leaf_nodes::Vector{Int}
  edges::Vector{Tuple{Int,Int}}
  node_features::Dict{Int,Vector{Float64}}
  edge_weights::Dict{Tuple{Int,Int},Float64}
  node_types::Dict{Int,Symbol}
  adjacency_matrix::SparseMatrixCSC{Float64}

  function LeafyChainGraph(root_nodes::Vector{Int}, leaf_nodes::Vector{Int},
    edges::Vector{Tuple{Int,Int}},
    node_features::Dict{Int,Vector{Float64}}=Dict{Int,Vector{Float64}}(),
    edge_weights::Dict{Tuple{Int,Int},Float64}=Dict{Tuple{Int,Int},Float64}(),
    node_types::Dict{Int,Symbol}=Dict{Int,Symbol}())

    # Create adjacency matrix
    num_nodes = length(root_nodes) + length(leaf_nodes)
    I = Int[]
    J = Int[]
    V = Float64[]

    for (i, j) in edges
      push!(I, i)
      push!(J, j)
      weight = get(edge_weights, (i, j), 1.0)
      push!(V, weight)
    end

    adjacency_matrix = sparse(I, J, V, num_nodes, num_nodes)

    new(root_nodes, leaf_nodes, edges, node_features, edge_weights, node_types, adjacency_matrix)
  end
end

"""
    LeafyChainNode

Represents a node in the leafy chain graph.
"""
struct LeafyChainNode
  id::Int
  node_type::Symbol  # :root, :leaf, :intermediate
  features::Vector{Float64}
  position::Int  # Position in the original text
  text::String  # Original text content
  confidence::Float64

  function LeafyChainNode(id::Int, node_type::Symbol, features::Vector{Float64},
    position::Int, text::String, confidence::Float64)
    @assert node_type in [:root, :leaf, :intermediate] "Invalid node type: $node_type"
    @assert 0.0 <= confidence <= 1.0 "Confidence must be between 0.0 and 1.0"
    @assert position >= 0 "Position must be non-negative"

    new(id, node_type, features, position, text, confidence)
  end
end

"""
    LeafyChainEdge

Represents an edge in the leafy chain graph.
"""
struct LeafyChainEdge
  source::Int
  target::Int
  edge_type::Symbol  # :semantic, :syntactic, :hierarchical
  weight::Float64
  confidence::Float64

  function LeafyChainEdge(source::Int, target::Int, edge_type::Symbol,
    weight::Float64, confidence::Float64)
    @assert edge_type in [:semantic, :syntactic, :hierarchical] "Invalid edge type: $edge_type"
    @assert weight >= 0.0 "Weight must be non-negative"
    @assert 0.0 <= confidence <= 1.0 "Confidence must be between 0.0 and 1.0"

    new(source, target, edge_type, weight, confidence)
  end
end

# ============================================================================
# Graph Construction
# ============================================================================

"""
    create_leafy_chain_graph(text::String, tokens::Vector{String}, semantic_nodes::Vector{String})

Create a leafy chain graph from text and semantic nodes.
"""
function create_leafy_chain_graph(text::String, tokens::Vector{String}, semantic_nodes::Vector{String})
  # Create root nodes for each token
  root_nodes = Int[]
  leaf_nodes = Int[]
  edges = Tuple{Int,Int}[]
  node_features = Dict{Int,Vector{Float64}}()
  edge_weights = Dict{Tuple{Int,Int},Float64}()
  node_types = Dict{Int,Symbol}()

  node_id = 1

  # Create root nodes for tokens
  for (i, token) in enumerate(tokens)
    push!(root_nodes, node_id)
    node_types[node_id] = :root
    node_features[node_id] = zeros(Float64, 768)  # Placeholder for RoBERTa features
    node_id += 1
  end

  # Create leaf nodes for semantic entities
  for (i, semantic_node) in enumerate(semantic_nodes)
    push!(leaf_nodes, node_id)
    node_types[node_id] = :leaf
    node_features[node_id] = zeros(Float64, 768)  # Placeholder for semantic features
    node_id += 1
  end

  # Create edges between root and leaf nodes
  for (root_idx, root_node) in enumerate(root_nodes)
    for (leaf_idx, leaf_node) in enumerate(leaf_nodes)
      # Simple heuristic: connect if there's semantic similarity
      # In practice, this would use more sophisticated matching
      if root_idx <= length(semantic_nodes) && leaf_idx <= length(semantic_nodes)
        push!(edges, (root_node, leaf_node))
        edge_weights[(root_node, leaf_node)] = 1.0
      end
    end
  end

  # Create hierarchical edges between root nodes
  for i in 1:(length(root_nodes)-1)
    push!(edges, (root_nodes[i], root_nodes[i+1]))
    edge_weights[(root_nodes[i], root_nodes[i+1])] = 0.8
  end

  return LeafyChainGraph(root_nodes, leaf_nodes, edges, node_features, edge_weights, node_types)
end

"""
    create_hierarchical_leafy_chain(text::String, tokens::Vector{String}, semantic_nodes::Vector{String}, hierarchy_levels::Int)

Create a hierarchical leafy chain graph with multiple levels.
"""
function create_hierarchical_leafy_chain(text::String, tokens::Vector{String}, semantic_nodes::Vector{String}, hierarchy_levels::Int)
  # Start with basic leafy chain
  graph = create_leafy_chain_graph(text, tokens, semantic_nodes)

  # Add intermediate nodes for hierarchy
  intermediate_nodes = Int[]
  new_edges = copy(graph.edges)
  new_edge_weights = copy(graph.edge_weights)
  new_node_types = copy(graph.node_types)
  new_node_features = copy(graph.node_features)

  node_id = maximum(vcat(graph.root_nodes, graph.leaf_nodes)) + 1

  for level in 1:hierarchy_levels
    # Create intermediate nodes for this level
    level_nodes = Int[]
    for i in 1:div(length(graph.root_nodes), 2^level)
      push!(level_nodes, node_id)
      new_node_types[node_id] = :intermediate
      new_node_features[node_id] = zeros(Float64, 768)
      node_id += 1
    end

    # Connect to previous level
    if level == 1
      # Connect to root nodes
      for (i, intermediate_node) in enumerate(level_nodes)
        root_start = (i - 1) * 2^(hierarchy_levels - level) + 1
        root_end = min(i * 2^(hierarchy_levels - level), length(graph.root_nodes))

        for root_idx in root_start:root_end
          if root_idx <= length(graph.root_nodes)
            push!(new_edges, (graph.root_nodes[root_idx], intermediate_node))
            new_edge_weights[(graph.root_nodes[root_idx], intermediate_node)] = 0.9
          end
        end
      end
    else
      # Connect to previous level intermediate nodes
      prev_level_nodes = intermediate_nodes
      for (i, intermediate_node) in enumerate(level_nodes)
        prev_start = (i - 1) * 2 + 1
        prev_end = min(i * 2, length(prev_level_nodes))

        for prev_idx in prev_start:prev_end
          if prev_idx <= length(prev_level_nodes)
            push!(new_edges, (prev_level_nodes[prev_idx], intermediate_node))
            new_edge_weights[(prev_level_nodes[prev_idx], intermediate_node)] = 0.9
          end
        end
      end
    end

    append!(intermediate_nodes, level_nodes)
  end

  # Connect final level to leaf nodes
  if !isempty(intermediate_nodes)
    final_level_nodes = intermediate_nodes[end-div(length(intermediate_nodes), 2^hierarchy_levels)+1:end]
    for (i, leaf_node) in enumerate(graph.leaf_nodes)
      if i <= length(final_level_nodes)
        push!(new_edges, (final_level_nodes[i], leaf_node))
        new_edge_weights[(final_level_nodes[i], leaf_node)] = 0.95
      end
    end
  end

  return LeafyChainGraph(graph.root_nodes, graph.leaf_nodes, new_edges,
    new_node_features, new_edge_weights, new_node_types)
end

# ============================================================================
# Graph Operations
# ============================================================================

"""
    get_root_nodes(graph::LeafyChainGraph)

Get all root nodes in the graph.
"""
function get_root_nodes(graph::LeafyChainGraph)
  return graph.root_nodes
end

"""
    get_leaf_nodes(graph::LeafyChainGraph)

Get all leaf nodes in the graph.
"""
function get_leaf_nodes(graph::LeafyChainGraph)
  return graph.leaf_nodes
end

"""
    get_intermediate_nodes(graph::LeafyChainGraph)

Get all intermediate nodes in the graph.
"""
function get_intermediate_nodes(graph::LeafyChainGraph)
  intermediate_nodes = Int[]
  for (node_id, node_type) in graph.node_types
    if node_type == :intermediate
      push!(intermediate_nodes, node_id)
    end
  end
  return intermediate_nodes
end

"""
    get_node_features(graph::LeafyChainGraph, node_id::Int)

Get features for a specific node.
"""
function get_node_features(graph::LeafyChainGraph, node_id::Int)
  return get(graph.node_features, node_id, Float64[])
end

"""
    set_node_features!(graph::LeafyChainGraph, node_id::Int, features::Vector{Float64})

Set features for a specific node.
"""
function set_node_features!(graph::LeafyChainGraph, node_id::Int, features::Vector{Float64})
  graph.node_features[node_id] = features
  return nothing
end

"""
    get_edge_weight(graph::LeafyChainGraph, source::Int, target::Int)

Get the weight of an edge between two nodes.
"""
function get_edge_weight(graph::LeafyChainGraph, source::Int, target::Int)
  return get(graph.edge_weights, (source, target), 0.0)
end

"""
    set_edge_weight!(graph::LeafyChainGraph, source::Int, target::Int, weight::Float64)

Set the weight of an edge between two nodes.
"""
function set_edge_weight!(graph::LeafyChainGraph, source::Int, target::Int, weight::Float64)
  graph.edge_weights[(source, target)] = weight
  return nothing
end

# ============================================================================
# Graph Analysis
# ============================================================================

"""
    get_graph_statistics(graph::LeafyChainGraph)

Get statistics about the leafy chain graph.
"""
function get_graph_statistics(graph::LeafyChainGraph)
  total_nodes = length(graph.root_nodes) + length(graph.leaf_nodes) + length(get_intermediate_nodes(graph))

  return Dict{String,Any}(
    "total_nodes" => total_nodes,
    "root_nodes" => length(graph.root_nodes),
    "leaf_nodes" => length(graph.leaf_nodes),
    "intermediate_nodes" => length(get_intermediate_nodes(graph)),
    "total_edges" => length(graph.edges),
    "density" => length(graph.edges) / (total_nodes * (total_nodes - 1) / 2),
    "max_degree" => maximum([sum(graph.adjacency_matrix[i, :]) for i in 1:size(graph.adjacency_matrix, 1)]),
    "min_degree" => minimum([sum(graph.adjacency_matrix[i, :]) for i in 1:size(graph.adjacency_matrix, 1)])
  )
end

"""
    get_node_degree(graph::LeafyChainGraph, node_id::Int)

Get the degree of a specific node.
"""
function get_node_degree(graph::LeafyChainGraph, node_id::Int)
  return sum(graph.adjacency_matrix[node_id, :])
end

"""
    get_neighbors(graph::LeafyChainGraph, node_id::Int)

Get all neighbors of a specific node.
"""
function get_neighbors(graph::LeafyChainGraph, node_id::Int)
  neighbors = Int[]
  for (source, target) in graph.edges
    if source == node_id
      push!(neighbors, target)
    elseif target == node_id
      push!(neighbors, source)
    end
  end
  return neighbors
end

# ============================================================================
# Graph Visualization
# ============================================================================

"""
    to_lightgraph(graph::LeafyChainGraph)

Convert leafy chain graph to LightGraphs format.
"""
function to_lightgraph(graph::LeafyChainGraph)
  num_nodes = length(graph.root_nodes) + length(graph.leaf_nodes) + length(get_intermediate_nodes(graph))
  g = SimpleGraph(num_nodes)

  for (source, target) in graph.edges
    add_edge!(g, source, target)
  end

  return g
end

"""
    to_metagraph(graph::LeafyChainGraph)

Convert leafy chain graph to MetaGraphs format.
"""
function to_metagraph(graph::LeafyChainGraph)
  num_nodes = length(graph.root_nodes) + length(graph.leaf_nodes) + length(get_intermediate_nodes(graph))
  mg = MetaGraph(num_nodes)

  # Add node properties
  for (node_id, node_type) in graph.node_types
    set_prop!(mg, node_id, :node_type, node_type)
  end

  for (node_id, features) in graph.node_features
    set_prop!(mg, node_id, :features, features)
  end

  # Add edges
  for (source, target) in graph.edges
    add_edge!(mg, source, target)
    weight = get(graph.edge_weights, (source, target), 1.0)
    set_prop!(mg, source, target, :weight, weight)
  end

  return mg
end

# ============================================================================
# Graph Persistence
# ============================================================================

"""
    save_leafy_chain_graph(graph::LeafyChainGraph, filepath::String)

Save a leafy chain graph to file.
"""
function save_leafy_chain_graph(graph::LeafyChainGraph, filepath::String)
  graph_data = Dict{String,Any}(
    "root_nodes" => graph.root_nodes,
    "leaf_nodes" => graph.leaf_nodes,
    "edges" => graph.edges,
    "node_features" => graph.node_features,
    "edge_weights" => graph.edge_weights,
    "node_types" => graph.node_types
  )

  open(filepath, "w") do io
    JSON.print(io, graph_data, 2)
  end

  return true
end

"""
    load_leafy_chain_graph(filepath::String)

Load a leafy chain graph from file.
"""
function load_leafy_chain_graph(filepath::String)
  graph_data = JSON.parsefile(filepath)

  root_nodes = graph_data["root_nodes"]
  leaf_nodes = graph_data["leaf_nodes"]
  edges = [(edge[1], edge[2]) for edge in graph_data["edges"]]
  node_features = Dict{Int,Vector{Float64}}(graph_data["node_features"])
  edge_weights = Dict{Tuple{Int,Int},Float64}([(edge[1], edge[2]) => weight for (edge, weight) in graph_data["edge_weights"]])
  node_types = Dict{Int,Symbol}([(node_id, Symbol(node_type)) for (node_id, node_type) in graph_data["node_types"]])

  return LeafyChainGraph(root_nodes, leaf_nodes, edges, node_features, edge_weights, node_types)
end
