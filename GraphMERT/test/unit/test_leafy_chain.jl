"""
Unit tests for Leafy Chain Graph implementation

Tests the core functionality of the leafy chain graph structure including:
- Graph creation and validation
- Adjacency matrix construction
- Shortest path computation (Floyd-Warshall)
- Triple injection
- Sequence encoding
- Attention mask creation
"""

using Test
using GraphMERT
using LinearAlgebra

# Import types for testing
using GraphMERT: ChainGraphNode, ChainGraphConfig, LeafyChainGraph, MNMConfig, SemanticTriple

@testset "Leafy Chain Graph Tests" begin

@testset "Graph Creation" begin
config = default_chain_graph_config()
@test config.num_roots == 128
@test config.num_leaves_per_root == 7
@test config.max_sequence_length == 1024
@test config.pad_token_id == 0

    graph = create_empty_chain_graph(config)
@test length(graph.nodes) == 1024  # 128 roots + 896 leaves
@test size(graph.adjacency_matrix) == (1024, 1024)
@test size(graph.shortest_paths) == (1024, 1024)
@test size(graph.leaf_tokens) == (128, 7)
    @test size(graph.injected_mask) == (128, 7)

# Check all root tokens are padding initially
@test all(token == config.pad_token_id for token in graph.root_tokens)
    @test all(text == "<pad>" for text in graph.root_texts)

# Check all leaf tokens are padding initially
@test all(token == config.pad_token_id for token in graph.leaf_tokens)
  @test all(isnothing(rel) for rel in graph.leaf_relations)
    @test all(!mask for mask in graph.injected_mask)

    # Validate graph structure
    @test validate_chain_graph(graph)
  end

  @testset "Adjacency Matrix Construction" begin
  config = default_chain_graph_config()
  graph = create_empty_chain_graph(config)

  # Test that adjacency matrix follows the structure:
  # - Chain connections between consecutive roots
  # - Star connections from each root to its leaves
  # - No leaf-to-leaf connections (following paper)

  adj = graph.adjacency_matrix

  # Test root chain: each root (except last) connects to next root
  for root_idx in 1:(config.num_roots-1)
      @test adj[root_idx, root_idx+1] > 0  # Directed edge root_idx -> root_idx+1
  end

  # Test root stars: each root connects to exactly num_leaves_per_root leaves
  for root_idx in 1:config.num_roots
  leaf_start = config.num_roots + (root_idx-1)*config.num_leaves_per_root + 1
  leaf_end = config.num_roots + root_idx*config.num_leaves_per_root
  connected_leaves = sum(adj[root_idx, leaf_start:leaf_end])
  @test connected_leaves == config.num_leaves_per_root
  end

  # Test no self-loops
    @test all(diag(adj) .== 0)

    # Test no leaf-to-leaf connections
    leaf_start = config.num_roots + 1
    @test all(adj[leaf_start:end, leaf_start:end] .== 0)
  end

  @testset "Floyd-Warshall Shortest Paths" begin
  config = default_chain_graph_config()
  adj_matrix = build_adjacency_matrix(config)

  # Compute shortest paths
  dist = floyd_warshall(adj_matrix)

  # Check dimensions
  total_nodes = config.max_sequence_length
  @test size(dist) == (total_nodes, total_nodes)

  # Check diagonal (distance to self should be 0)
  for i in 1:total_nodes
  @test dist[i, i] == 0
  end

  # Check direct connections (distance should be 1)
  for root_idx in 1:config.num_roots
  for leaf_idx in 1:config.num_leaves_per_root
    leaf_pos = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + leaf_idx
  @test dist[root_idx, leaf_pos] == 1  # Root to leaf
  end
  end

  # Check root chain distances
    for i in 1:config.num_roots
    for j in (i+1):config.num_roots
      expected_dist = j - i  # Direct chain distance
    @test dist[i, j] == expected_dist
  end
  end

  # Check that disconnected nodes have infinite distance (represented as large number)
  # Leaves should be disconnected from each other
  leaf_start = config.num_roots + 1
  for i in leaf_start:total_nodes
    for j in (i+1):total_nodes
        @test dist[i, j] == typemax(Int) ÷ 2  # Infinity representation
      end
    end
  end

  @testset "Triple Injection" begin
  config = default_chain_graph_config()
  graph = create_empty_chain_graph(config)

  # Test successful injection
  tail_tokens = [100, 200]
  inject_triple!(graph, 0, 0, tail_tokens, "metformin", :treats, "diabetes")

  # Check that leaf tokens were updated
  @test graph.leaf_tokens[1, 1] == 100
    @test graph.leaf_tokens[1, 2] == 200
  @test graph.leaf_tokens[1, 3] == config.pad_token_id  # Rest should be padding

    # Check that injected mask was updated
  @test graph.injected_mask[1, 1] == true
  @test graph.injected_mask[1, 2] == true
  @test graph.injected_mask[1, 3] == false

  # Check that relations were set
    @test graph.leaf_relations[1, 1] == :treats
  @test graph.leaf_relations[1, 2] == :treats

  # Check injection count
  @test graph.num_injections == 1

  # Test edge cases
    @test_throws AssertionError inject_triple!(graph, -1, 0, tail_tokens, "test", :rel, "head")  # Invalid root
    @test_throws AssertionError inject_triple!(graph, 0, 10, tail_tokens, "test", :rel, "head")  # Invalid leaf index
    @test_throws AssertionError inject_triple!(graph, 0, 0, [1,2,3,4,5,6,7,8], "test", :rel, "head")  # Too many tokens
  end

  @testset "Graph to Sequence Conversion" begin
  config = default_chain_graph_config()
  graph = create_empty_chain_graph(config)

  # Inject a test triple to modify some nodes
  inject_triple!(graph, 0, 0, [100, 150], "entity", :relation, "test")

    sequence = graph_to_sequence(graph)

    # Check sequence structure
  @test length(sequence) == config.max_sequence_length
  @test sequence[1] == config.pad_token_id  # First root is still padding
  @test sequence[129] == 100  # First injected leaf token (after 128 roots)
  @test sequence[130] == 150  # Second injected leaf token

    # Check padding tokens for unfilled nodes
  @test sequence[131] == config.pad_token_id  # Remaining leaves are padding
  @test sequence[132] == config.pad_token_id
  @test sequence[133] == config.pad_token_id
  @test sequence[134] == config.pad_token_id
  @test sequence[135] == config.pad_token_id
  end

  @testset "Attention Mask Creation" begin
  config = default_chain_graph_config()
  graph = create_empty_chain_graph(config)

  # Initially all tokens are padding, so attention mask should be all zeros
    # (except roots are padding, leaves are uninjected)
  mask = create_attention_mask(graph)

  @test length(mask) == config.max_sequence_length

  # Since all tokens are padding/uninjected, mask should be all zeros
  @test all(mask .== 0)

  # Inject a triple to make some tokens "real"
    inject_triple!(graph, 0, 0, [100, 200], "entity", :relation, "head")

  mask_after = create_attention_mask(graph)

  # Now some positions should be 1 (attended to)
  @test mask_after[129] == 1  # First injected leaf
  @test mask_after[130] == 1  # Second injected leaf
  @test mask_after[131] == 0  # Uninjected leaf
  end
end
