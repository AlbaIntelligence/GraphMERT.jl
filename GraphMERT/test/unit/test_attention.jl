"""
Unit tests for Attention Mechanisms implementation

Tests the spatial attention functionality including:
- Attention decay mask creation
- Graph-based attention masks
- Spatial bias application
- Integration with distance matrices
"""

using Test
using GraphMERT
using LinearAlgebra

@testset "Attention Mechanisms Tests" begin

    @testset "SpatialAttentionConfig" begin
        config = SpatialAttentionConfig()
        @test config.max_distance == 512
        @test config.decay_lambda == 0.6
        @test config.decay_p_init == 1.0
        @test config.use_distance_bias == true
        @test config.distance_bias_weight == 0.1

        # Test custom config
        custom_config = SpatialAttentionConfig(
            decay_lambda = 0.8,
            use_distance_bias = false
        )
        @test custom_config.decay_lambda == 0.8
        @test custom_config.use_distance_bias == false
    end

    @testset "create_attention_decay_mask (sequence)" begin
        config = SpatialAttentionConfig(max_distance = 10, decay_lambda = 0.5)
        mask = create_attention_decay_mask(5, config)

        @test size(mask.mask) == (5, 5)
        @test mask.mask[1, 1] == 1.0f0  # Self-attention
        @test mask.mask[1, 2] < 1.0f0   # Distance 1, should decay
        @test mask.mask[1, 5] < mask.mask[1, 2]  # Further distance, more decay
    end

    @testset "create_attention_decay_mask (graph)" begin
        # Create a simple distance matrix (from Floyd-Warshall)
        dist_matrix = Float32[
            0 1 2 3;
            1 0 1 2;
            2 1 0 1;
            3 2 1 0
        ]

        config = SpatialAttentionConfig(decay_lambda = 0.5, use_distance_bias = false)
        decay_mask = create_attention_decay_mask(dist_matrix, config)

        @test size(decay_mask) == (4, 4)
        @test decay_mask[1, 1] == 1.0  # exp(-0.5 * 0) = 1
        @test decay_mask[1, 2] ≈ exp(-0.5 * 1)  # exp(-0.5 * 1)
        @test decay_mask[1, 4] ≈ exp(-0.5 * 3)  # exp(-0.5 * 3)

        # Test with distance bias
        config_with_bias = SpatialAttentionConfig(
            decay_lambda = 0.5,
            use_distance_bias = true,
            distance_bias_weight = 0.2  # Different weight
        )
        decay_mask_bias = create_attention_decay_mask(dist_matrix, config_with_bias)

        # With bias, the values should be different
        @test decay_mask_bias != decay_mask
        @test decay_mask_bias[1, 2] ≈ exp(-0.5 * 1) + 0.2 * 1
    end

    @testset "create_graph_attention_mask" begin
        # Create a simple adjacency matrix (chain structure)
        using SparseArrays
        adj = sparse(Float32[
            0 1 0 0;
            1 0 1 0;
            0 1 0 1;
            0 0 1 0
        ])

        config = SpatialAttentionConfig(decay_lambda = 1.0)
        attention_mask = create_graph_attention_mask(adj, config)

        @test size(attention_mask.mask) == (4, 4)
        @test attention_mask.mask[1, 1] == 1.0f0  # Self-attention

        # Check that directly connected nodes have higher attention
        @test attention_mask.mask[1, 2] > attention_mask.mask[1, 3]  # 1->2 closer than 1->3
    end

    @testset "apply_spatial_bias" begin
        attention_scores = rand(Float32, 4, 4)
        positions = [1, 2, 3, 4]

        config = SpatialAttentionConfig(use_distance_bias = true, distance_bias_weight = 0.2)

        biased_scores = apply_spatial_bias(attention_scores, positions, config)

        # Should be different from original
        @test biased_scores != attention_scores

        # Test with bias disabled
        config_no_bias = SpatialAttentionConfig(use_distance_bias = false)
        unbiased_scores = apply_spatial_bias(attention_scores, positions, config_no_bias)

        @test unbiased_scores == attention_scores
    end

    @testset "Integration with LeafyChainGraph" begin
        # Test integration with actual graph structures
        config = GraphMERT.default_chain_graph_config()
        graph = GraphMERT.create_empty_chain_graph([101, 102], ["hello", "world"], config)

        # The graph should have shortest paths computed
        @test size(graph.shortest_paths) == (1024, 1024)

        # Create attention decay mask from graph distances
        attention_config = SpatialAttentionConfig(decay_lambda = 0.8, use_distance_bias = false)
        decay_mask = create_attention_decay_mask(Float32.(graph.shortest_paths), attention_config)

        @test size(decay_mask) == (1024, 1024)
        @test decay_mask[1, 1] == 1.0f0  # Self-attention preserved

        # Check that exponential decay works (closer nodes have higher attention)
        @test decay_mask[1, 2] > decay_mask[1, 10]  # Closer nodes have higher attention
    end

end
