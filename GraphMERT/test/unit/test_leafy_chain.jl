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

# Import types for testing
using GraphMERT: ChainGraphNode, ChainGraphConfig, LeafyChainGraph, MNMConfig, SemanticTriple

@testset "Leafy Chain Graph Tests" begin

  @testset "Graph Creation" begin
    config = default_chain_graph_config()
    @test config.num_roots == 128
    @test config.num_leaves_per_root == 7
    @test config.vocab_size == 30522
    @test config.pad_token_id == 1
    @test config.mask_token_id == 103

    graph = create_empty_chain_graph(config)
    @test length(graph.root_nodes) == 128
    @test length(graph.leaf_nodes) == 128
    @test all(length(leaves) == 7 for leaves in graph.leaf_nodes)

    # Check all nodes are padding initially
    @test all(node.is_padding for node in graph.root_nodes)
    @test all(all(node.is_padding for node in leaves) for leaves in graph.leaf_nodes)

    # Check adjacency matrix size
    total_nodes = 128 * (1 + 7)  # 128 roots + 128*7 leaves
    @test size(graph.adjacency_matrix) == (total_nodes, total_nodes)
  end

  @testset "Adjacency Matrix Construction" begin
    config = default_chain_graph_config()
    graph = create_empty_chain_graph(config)

    # Test root-to-leaf connections
    for root_idx in 1:config.num_roots
      root_pos = root_idx
      for leaf_idx in 1:config.num_leaves_per_root
        leaf_pos = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + leaf_idx
        @test graph.adjacency_matrix[root_pos, leaf_pos] == true
        @test graph.adjacency_matrix[leaf_pos, root_pos] == true
      end
    end

    # Test leaf-to-leaf connections within same root
    for root_idx in 1:config.num_roots
      for i in 1:config.num_leaves_per_root
        for j in (i+1):config.num_leaves_per_root
          leaf_pos_i = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + i
          leaf_pos_j = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + j
          @test graph.adjacency_matrix[leaf_pos_i, leaf_pos_j] == true
          @test graph.adjacency_matrix[leaf_pos_j, leaf_pos_i] == true
        end
      end
    end
  end

  @testset "Floyd-Warshall Shortest Paths" begin
    config = default_chain_graph_config()
    graph = create_empty_chain_graph(config)

    # Compute shortest paths
    dist = floyd_warshall(graph)

    # Check dimensions
    total_nodes = config.num_roots * (1 + config.num_leaves_per_root)
    @test size(dist) == (total_nodes, total_nodes)

    # Check diagonal (distance to self should be 0)
    for i in 1:total_nodes
      @test dist[i, i] == 0
    end

    # Check direct connections (distance should be 1)
    for root_idx in 1:config.num_roots
      root_pos = root_idx
      for leaf_idx in 1:config.num_leaves_per_root
        leaf_pos = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + leaf_idx
        @test dist[root_pos, leaf_pos] == 1
        @test dist[leaf_pos, root_pos] == 1
      end
    end

    # Check leaf-to-leaf within same root (distance should be 1)
    for root_idx in 1:config.num_roots
      for i in 1:config.num_leaves_per_root
        for j in (i+1):config.num_leaves_per_root
          leaf_pos_i = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + i
          leaf_pos_j = config.num_roots + (root_idx - 1) * config.num_leaves_per_root + j
          @test dist[leaf_pos_i, leaf_pos_j] == 1
          @test dist[leaf_pos_j, leaf_pos_i] == 1
        end
      end
    end
  end

  @testset "Triple Injection" begin
    config = default_chain_graph_config()
    graph = create_empty_chain_graph(config)

    # Create a test triple
    triple = SemanticTriple("diabetes", nothing, "treats", "metformin",
      [2156, 23421], 0.95, "test")  # Example token IDs

    # Test successful injection
    @test inject_triple!(graph, triple, 1) == true

    # Check that root node is no longer padding
    @test graph.root_nodes[1].is_padding == false

    # Check that leaf nodes have the injected tokens
    @test graph.leaf_nodes[1][1].token_id == 2156
    @test graph.leaf_nodes[1][2].token_id == 23421
    @test graph.leaf_nodes[1][1].is_padding == false
    @test graph.leaf_nodes[1][2].is_padding == false

    # Test injection failure cases
    long_triple = SemanticTriple("very_long_entity_name", nothing, "treats", "very_long_tail_entity",
      [1, 2, 3, 4, 5, 6, 7, 8], 0.95, "test")  # Too many tokens
    @test inject_triple!(graph, long_triple, 1) == false  # Should fail

    @test inject_triple!(graph, triple, 999) == false  # Invalid root index
  end

  @testset "Graph to Sequence Conversion" begin
    config = default_chain_graph_config()
    graph = create_empty_chain_graph(config)

    # Inject a test triple to modify some nodes
    triple = SemanticTriple("test", nothing, "relation", "entity", [100, 150], 0.9, "test")
    @test inject_triple!(graph, triple, 1) == true

    sequence = graph_to_sequence(graph)

    # Check sequence structure
    @test length(sequence) == config.num_roots * (1 + config.num_leaves_per_root)
    @test sequence[1] == config.pad_token_id  # First root is still padding
    @test sequence[2] == 100  # First injected token
    @test sequence[3] == 150  # Second injected token

    # Check padding tokens for unfilled nodes
    @test sequence[4] == config.pad_token_id  # Remaining leaves are padding
    @test sequence[5] == config.pad_token_id
    @test sequence[6] == config.pad_token_id
    @test sequence[7] == config.pad_token_id
    @test sequence[8] == config.pad_token_id
  end

  @testset "Attention Mask Creation" begin
    config = default_chain_graph_config()
    graph = create_empty_chain_graph(config)

    mask = create_attention_mask(graph)

    # Check dimensions
    total_nodes = config.num_roots * (1 + config.num_leaves_per_root)
    @test size(mask) == (total_nodes, total_nodes)

    # All nodes should be able to attend to themselves
    for i in 1:total_nodes
      @test mask[i, i] == true
    end

    # Test that connected nodes can attend to each other
    # Root 1 should be able to attend to its leaves
    for leaf_idx in 1:config.num_leaves_per_root
      leaf_pos = config.num_roots + leaf_idx
      @test mask[1, leaf_pos] == true
      @test mask[leaf_pos, 1] == true
    end

    # Test with precomputed shortest paths
    if graph.config.precompute_shortest_paths
      graph.shortest_paths = floyd_warshall(graph)
      mask_with_paths = create_attention_mask(graph)

      # Should be similar but may have some differences due to path-based decay
      @test size(mask_with_paths) == size(mask)

      # Self-attention should still be preserved
      for i in 1:total_nodes
        @test mask_with_paths[i, i] == true
      end
    end
  end

  @testset "Text to Graph Conversion" begin
    config = default_chain_graph_config()
    text = "diabetes mellitus is a chronic condition"

    graph = create_leafy_chain_from_text(text, config)

    @test length(graph.root_nodes) == config.num_roots

    # First few root nodes should have non-padding tokens
    @test graph.root_nodes[1].is_padding == false
    @test graph.root_nodes[2].is_padding == false
    @test graph.root_nodes[3].is_padding == false
    @test graph.root_nodes[4].is_padding == false

    # Later root nodes should be padding (text is short)
    @test graph.root_nodes[5].is_padding == true
  end

  @testset "Type Validation" begin
    # Test ChainGraphNode validation
    @test ChainGraphNode(:root, 100, 1, nothing, false)  # Valid root
    @test ChainGraphNode(:leaf, 200, 129, 1, false)     # Valid leaf

    @test_throws AssertionError ChainGraphNode(:invalid, 100, 1, nothing, false)  # Invalid type
    @test_throws AssertionError ChainGraphNode(:root, -1, 1, nothing, false)     # Invalid token_id
    @test_throws AssertionError ChainGraphNode(:leaf, 100, 1, nothing, false)     # Leaf without parent_root

    # Test ChainGraphConfig validation
    @test ChainGraphConfig(64, 5, 30000)  # Valid config

    @test_throws AssertionError ChainGraphConfig(0, 5, 30000)     # Invalid num_roots
    @test_throws AssertionError ChainGraphConfig(64, -1, 30000)   # Invalid num_leaves_per_root
    @test_throws AssertionError ChainGraphConfig(64, 5, -1)       # Invalid vocab_size

    # Test MNMConfig validation (use the actual constructor)
    @test MNMConfig(30000, 512, 7, 0.15)  # Use 4-argument constructor

    @test_throws AssertionError MNMConfig(0, 512, 7, 0.15)     # Invalid vocab_size
    @test_throws AssertionError MNMConfig(30000, 0, 7, 0.15)   # Invalid hidden_size
    @test_throws AssertionError MNMConfig(30000, 512, -1, 0.15) # Invalid num_leaves
    @test_throws AssertionError MNMConfig(30000, 512, 7, 1.5)   # Invalid mask_probability

    # Test SemanticTriple validation
    tokens = Int[100, 200, 300]
    @test SemanticTriple("head", nothing, "relation", "tail", tokens, 0.8, "test")

    @test_throws AssertionError SemanticTriple("", nothing, "relation", "tail", tokens, 0.8, "test")        # Empty head
    @test_throws AssertionError SemanticTriple("head", nothing, "", "tail", tokens, 0.8, "test")           # Empty relation
    @test_throws AssertionError SemanticTriple("head", nothing, "relation", "", tokens, 0.8, "test")        # Empty tail
    @test_throws AssertionError SemanticTriple("head", nothing, "relation", "tail", Int[], 0.8, "test")     # Empty tokens
    @test_throws AssertionError SemanticTriple("head", nothing, "relation", "tail", tokens, 1.5, "test")     # Invalid score
    @test_throws AssertionError SemanticTriple("head", nothing, "relation", "tail", tokens, 0.8, "")         # Empty source
  end
end
