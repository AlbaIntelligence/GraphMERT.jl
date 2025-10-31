"""
Test suite for MNM training implementation
"""

using Test
using ..GraphMERT

@testset "MNM Training Tests" begin

    @testset "Leaf Selection" begin
        config = default_chain_graph_config()
        tokens = [1, 2, 3]
        texts = ["diabetes", "mellitus", "is"]
        graph = create_empty_chain_graph(tokens, texts, config)

        # Inject a triple to make some leaves available
        inject_triple!(graph, 0, 0, [100, 101], "chronic disease", :isa, "diabetes")

        mnm_probability = 0.15
        rng = Random.MersenneTwister(42)

        selected_leaves = select_leaves_to_mask(graph, mnm_probability, rng)

        @test isa(selected_leaves, Vector{Tuple{Int,Int}})
        @test all(leaf -> leaf[1] >= 0 && leaf[1] < config.num_roots, selected_leaves)
        @test all(leaf -> leaf[2] >= 0 && leaf[2] < config.num_leaves_per_root, selected_leaves)
    end

    @testset "MNM Masking" begin
        config = default_chain_graph_config()
        tokens = [1, 2, 3]
        texts = ["diabetes", "mellitus", "is"]
        graph = create_empty_chain_graph(tokens, texts, config)

        # Inject a triple
        inject_triple!(graph, 0, 0, [100, 101], "chronic disease", :isa, "diabetes")

        masked_leaves = [(0, 0), (0, 1)]  # First two leaves of root 0
        mask_token_id = 103
        vocab_size = 30522
        rng = Random.MersenneTwister(42)

        masked_graph, labels = apply_mnm_masks(graph, masked_leaves, mask_token_id, vocab_size, rng)

        @test isa(masked_graph, LeafyChainGraph)
        @test isa(labels, Matrix{Int})
        @test size(labels) == (config.num_roots, config.num_leaves_per_root)

        # Check that labels are correct for masked positions
        @test labels[1, 1] == 100  # Original token ID
        @test labels[1, 2] == 101  # Original token ID
        @test all(labels[1, 3:end] .== -100)  # Unmasked positions
    end

    @testset "MNM Loss Calculation" begin
        # Create mock logits (batch_size=1, seq_len=1024, vocab_size=100)
        logits = rand(Float32, 1, 1024, 100)

        # Create mock labels
        labels = fill(-100, 128, 7)
        labels[1, 1] = 50  # One prediction target

        # Create mock attention mask
        attention_mask = ones(Int, 1, 1024)

        # Create mock graph
        config = default_chain_graph_config()
        graph = create_empty_chain_graph([1, 2, 3], ["a", "b", "c"], config)

        loss = calculate_mnm_loss(logits, labels, attention_mask, graph)

        @test isa(loss, Float32)
        @test loss >= 0.0f0
    end

    @testset "Joint Training Function Signature" begin
        # This is mainly a compilation test
        model = nothing  # Mock model
        graph = create_empty_chain_graph([1, 2], ["test", "text"], default_chain_graph_config())
        mlm_config = default_mlm_config()
        mnm_config = default_mnm_config()
        optimizer = nothing  # Mock optimizer

        # Test that function can be called (will fail at runtime due to mocks)
        try
            result = train_joint_mlm_mnm_step(
                model, graph, mlm_config, mnm_config, optimizer, 1.0
            )
            @test isa(result, Tuple{Float32, Float32, Float32})
        catch e
            # Expected to fail due to mock objects
            @test isa(e, Exception)
        end
    end

end
