"""
Test suite for Leafy Chain Graph implementation
"""

using Test

# Import GraphMERT module
push!(LOAD_PATH, "../GraphMERT/src")
using GraphMERT

@testset "Leafy Chain Graph Tests" begin

    @testset "Configuration Tests" begin
        config = default_chain_graph_config()
        @test config.num_roots == 128
        @test config.num_leaves_per_root == 7
        @test config.max_sequence_length == 1024
        @test config.pad_token_id == 0
        @test config.vocab_size == 30522
    end

    @testset "Empty Graph Creation" begin
        config = default_chain_graph_config()
        tokens = [1, 2, 3, 4, 5]
        texts = ["diabetes", "mellitus", "is", "a", "disease"]
        graph = create_empty_chain_graph(tokens, texts, config)

        @test length(graph.nodes) == 1024
        @test length(graph.root_tokens) == 128
        @test size(graph.leaf_tokens) == (128, 7)
        @test size(graph.leaf_relations) == (128, 7)
        @test size(graph.injected_mask) == (128, 7)
        @test graph.num_injections == 0

        # Check first few root nodes
        @test graph.root_tokens[1:5] == [1, 2, 3, 4, 5]
        @test graph.root_texts[1:5] == ["diabetes", "mellitus", "is", "a", "disease"]
        @test all(graph.root_tokens[6:128] .== config.pad_token_id)
        @test all(graph.root_texts[6:128] .== "<pad>")
    end

    @testset "Adjacency Matrix" begin
        config = default_chain_graph_config()
        adj = build_adjacency_matrix(config)

        @test size(adj) == (1024, 1024)
        @test eltype(adj) == SparseMatrixCSC{Float32}

        # Check self-loops
        @test all(adj[i,i] == 1.0f0 for i in 1:1024)

        # Check root-to-leaf connections (first root to its leaves)
        root_idx = 1
        for leaf_offset in 0:6
            leaf_idx = 129 + leaf_offset
            @test adj[root_idx, leaf_idx] == 1.0f0
            @test adj[leaf_idx, root_idx] == 1.0f0
        end

        # Check leaf-to-leaf connections (leaves of first root)
        for i in 0:5
            for j in (i+1):6
                leaf1 = 129 + i
                leaf2 = 129 + j
                @test adj[leaf1, leaf2] == 1.0f0
                @test adj[leaf2, leaf1] == 1.0f0
            end
        end
    end

    @testset "Floyd-Warshall" begin
        config = default_chain_graph_config()
        adj = build_adjacency_matrix(config)
        sp = floyd_warshall(adj)

        @test size(sp) == (1024, 1024)
        @test eltype(sp) == Matrix{Int}

        # Self distances should be 0
        @test all(sp[i,i] == 0 for i in 1:1024)

        # Adjacent nodes should have distance 1
        @test sp[1,2] == 1  # Consecutive roots
        @test sp[1,129] == 1  # Root to its first leaf
        @test sp[129,130] == 1  # Leaves of same root
    end

    @testset "Triple Injection" begin
        config = default_chain_graph_config()
        tokens = [1, 2, 3]
        texts = ["diabetes", "mellitus", "is"]
        graph = create_empty_chain_graph(tokens, texts, config)

        # Inject a triple
        root_idx = 0  # First root (0-indexed)
        tail_tokens = [100, 101]  # "chronic disease"
        tail_text = "chronic disease"
        relation = :isa
        head_text = "diabetes"

        inject_triple!(graph, root_idx, 0, tail_tokens, tail_text, relation, head_text)

        @test graph.num_injections == 1
        @test graph.leaf_tokens[1, 1] == 100
        @test graph.leaf_tokens[1, 2] == 101
        @test graph.leaf_relations[1, 1] == :isa
        @test graph.injected_mask[1, 1] == true
        @test graph.injected_mask[1, 2] == true

        # Check node objects
        leaf1_node = graph.nodes[129]  # First leaf of first root
        @test leaf1_node.node_type == :leaf
        @test leaf1_node.token_id == 100
        @test leaf1_node.relation == :isa
        @test leaf1_node.head_text == "diabetes"
    end

    @testset "Graph to Sequence" begin
        config = default_chain_graph_config()
        tokens = [1, 2, 3]
        texts = ["diabetes", "mellitus", "is"]
        graph = create_empty_chain_graph(tokens, texts, config)

        sequence = graph_to_sequence(graph)

        @test length(sequence) == 1024
        @test sequence[1:3] == [1, 2, 3]
        @test all(sequence[4:128] .== config.pad_token_id)  # Padded roots
        @test all(sequence[129:1024] .== config.pad_token_id)  # Empty leaves
    end

    @testset "Attention Mask" begin
        config = default_chain_graph_config()
        tokens = [1, 2, 3]
        texts = ["diabetes", "mellitus", "is"]
        graph = create_empty_chain_graph(tokens, texts, config)

        attention_mask = create_attention_mask(graph)

        @test length(attention_mask) == 1024
        @test all(attention_mask[1:3] .== 1)  # Valid root tokens
        @test all(attention_mask[4:128] .== 0)  # Padded root tokens
        @test all(attention_mask[129:1024] .== 0)  # Empty leaf tokens
    end

    @testset "Position IDs" begin
        config = default_chain_graph_config()
        tokens = [1, 2, 3]
        texts = ["diabetes", "mellitus", "is"]
        graph = create_empty_chain_graph(tokens, texts, config)

        position_ids = create_position_ids(graph)

        @test length(position_ids) == 1024
        @test position_ids == collect(0:1023)
    end

end
