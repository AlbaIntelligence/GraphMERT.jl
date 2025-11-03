"""
Tests for leafy chain graph functionality.

This test suite validates the core graph construction, manipulation,
and conversion functions that form the foundation of GraphMERT.
"""

using Test
using GraphMERT
using LinearAlgebra

@testset "Leafy Chain Graph Construction" begin

    @testset "Default Configuration" begin
        config = default_chain_graph_config()
        @test config.num_roots == 128
        @test config.num_leaves_per_root == 7
        @test config.max_sequence_length == 1024
        @test config.pad_token_id == 0
    end

    @testset "Graph Creation" begin
        config = default_chain_graph_config()
        graph = create_empty_chain_graph(config)

        @test length(graph.nodes) == 1024
        @test size(graph.adjacency_matrix) == (1024, 1024)
        @test size(graph.shortest_paths) == (1024, 1024)
        @test length(graph.root_tokens) == 128
        @test size(graph.leaf_tokens) == (128, 7)
        @test graph.num_injections == 0
    end

    @testset "Node Structure Validation" begin
        config = default_chain_graph_config()
        graph = create_empty_chain_graph(config)

        # Test root nodes
        for i in 1:config.num_roots
            node = graph.nodes[i]
            @test node.node_type == :root
            @test node.root_index == i-1
            @test isnothing(node.leaf_index)
            @test node.is_padding == true
            @test node.token_id == config.pad_token_id
        end

        # Test leaf nodes
        for i in (config.num_roots+1):config.max_sequence_length
            node = graph.nodes[i]
            @test node.node_type == :leaf
            @test !isnothing(node.leaf_index)
            @test 0 <= node.leaf_index < config.num_leaves_per_root
            @test node.is_padding == true
        end
    end

    @testset "Adjacency Matrix Structure" begin
        config = ChainGraphConfig(num_roots=4, num_leaves_per_root=2, max_sequence_length=12)
        adj = build_adjacency_matrix(config)

        # Check dimensions
        @test size(adj) == (12, 12)

        # Check no self-loops
        @test all(diag(adj) .== 0)

        # Check root chain: roots 1→2→3→4
        @test adj[1,2] > 0  # 1→2
        @test adj[2,3] > 0  # 2→3
        @test adj[3,4] > 0  # 3→4
        @test adj[4,1] == 0  # No 4→1

        # Check root stars: each root connects to 2 leaves
        @test sum(adj[1, 5:6]) == 2  # Root 1 → leaves 5,6
        @test sum(adj[2, 7:8]) == 2  # Root 2 → leaves 7,8
        @test sum(adj[3, 9:10]) == 2 # Root 3 → leaves 9,10
        @test sum(adj[4, 11:12]) == 2 # Root 4 → leaves 11,12

        # Check no leaf-to-leaf connections
        @test all(adj[5:end, 5:end] .== 0)
    end

    @testset "Graph Validation" begin
        config = default_chain_graph_config()
        graph = create_empty_chain_graph(config)

        # Should pass validation
        @test validate_chain_graph(graph) == true

        # Test validation failures
        graph.nodes[1] = ChainGraphNode(id=0, node_type=:leaf, root_index=0, token_id=0, token_text="")
        @test_throws AssertionError validate_chain_graph(graph)
    end

    @testset "Utility Functions" begin
        config = ChainGraphConfig(num_roots=4, num_leaves_per_root=2, max_sequence_length=12)

        @test get_root_node_indices(config) == 1:4
        @test get_leaf_node_indices(config, 1) == 5:6
        @test get_leaf_node_indices(config, 2) == 7:8

        @test get_node_index(config, 1) == 1  # Root 1
        @test get_node_index(config, 2) == 2  # Root 2
        @test get_node_index(config, 1, 0) == 5  # Root 1, leaf 0
        @test get_node_index(config, 1, 1) == 6  # Root 1, leaf 1
    end

    @testset "Triple Injection" begin
        config = default_chain_graph_config()
        graph = create_empty_chain_graph(config)

        # Test triple injection: (head=diabetes, relation=treats, tail=metformin)
        inject_triple!(graph, 0, 0, [123, 456], "metformin", :treats, "diabetes")

        @test graph.num_injections == 1
        @test graph.leaf_tokens[1, 1] == 123
        @test graph.leaf_tokens[1, 2] == 456
        @test graph.leaf_relations[1, 1] == :treats
        @test graph.leaf_relations[1, 2] == :treats
        @test graph.injected_mask[1, 1] == true
        @test graph.injected_mask[1, 2] == true

        # Check node updates
        node_129 = graph.nodes[129]  # First leaf of root 0 (1-based: 128 roots + 1)
        @test node_129.token_id == 123
        @test node_129.relation == :treats
        @test node_129.head_text == "diabetes"
        @test node_129.token_text == "metformin"
        @test !node_129.is_padding
    end

    @testset "Sequence Conversion" begin
        config = default_chain_graph_config()
        graph = create_empty_chain_graph(config)

        # Inject a triple
        inject_triple!(graph, 0, 0, [123, 456], "test", :rel, "target")

        seq = graph_to_sequence(graph)
        mask = create_attention_mask(graph)
        pos_ids = create_position_ids(graph)

        @test length(seq) == 1024
        @test length(mask) == 1024
        @test length(pos_ids) == 1024

        # Check position IDs
        @test pos_ids[1] == 0
        @test pos_ids[end] == 1023

        # Check root tokens (should be padding)
        @test all(seq[1:128] .== 0)

        # Check injected leaf tokens
        @test seq[129] == 123  # First leaf of root 0
        @test seq[130] == 456  # Second leaf

        # Check attention mask (padding positions should be 0)
        @test mask[129] == 1  # Injected token
        @test mask[131] == 0  # Non-injected leaf
    end

    @testset "Floyd-Warshall Shortest Paths" begin
        # Create small test graph
        config = ChainGraphConfig(num_roots=3, num_leaves_per_root=1, max_sequence_length=6)
        adj = build_adjacency_matrix(config)
        paths = floyd_warshall(adj)

        @test size(paths) == (6, 6)

        # Check direct connections (distance 1)
        @test paths[1,2] == 1  # Root 1 → Root 2
        @test paths[1,4] == 1  # Root 1 → Leaf 1

        # Check indirect connections
        @test paths[1,3] == 2  # Root 1 → Root 2 → Root 3

        # Check self distances
        @test all(diag(paths) .== 0)
    end

end
