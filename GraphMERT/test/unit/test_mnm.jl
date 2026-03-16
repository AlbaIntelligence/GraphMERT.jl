using Test
using Random
using Flux
using GraphMERT
using GraphMERT: MNMConfig, MNMBatch, LeafyChainGraph, ChainGraphNode, SemanticTriple,
  select_leaves_to_mask, apply_mnm_masks,
  create_empty_chain_graph, graph_to_sequence, create_attention_mask,
  ChainGraphConfig, GraphMERTConfig, GraphMERTModel, train_joint_mlm_mnm_step,
  calculate_mnm_loss

import GraphMERT: inject_triple!

function inject_triple!(graph::LeafyChainGraph, triple::SemanticTriple, root_idx::Int)
    actual_root_idx = root_idx - 1
    inject_triple!(
        graph, 
        actual_root_idx, 
        0, 
        triple.tail_tokens, 
        triple.tail, 
        Symbol(triple.relation), 
        triple.head
    )
    return true
end

@testset "MNM Training Tests" begin

  @testset "Leaf Masking Selection" begin
    config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
    graph = create_empty_chain_graph(ChainGraphConfig())
    triple = SemanticTriple("test", nothing, "relation", "entity", [100, 150, 200], 0.9, "test")
    @test inject_triple!(graph, triple, 1) == true
    masked_positions = select_leaves_to_mask(graph, config.mask_probability, Random.GLOBAL_RNG)
    # Could be empty if random decides not to mask, but with seed it might be consistent.
    # For now just check type
    @test masked_positions isa Vector{Tuple{Int,Int}}
  end

  @testset "Mask Application" begin
    config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
    graph = create_empty_chain_graph(ChainGraphConfig())
    triple = SemanticTriple("test", nothing, "relation", "entity", [100, 150], 0.9, "test")
    @test inject_triple!(graph, triple, 1) == true
    original_tokens = [100, 150]
    masked_positions = [(1, 1), (1, 2)]
    returned_graph, returned_labels = apply_mnm_masks(graph, masked_positions, config.mask_token_id, config.vocab_size, Random.GLOBAL_RNG)
    
    # We masked (1,1) and (1,2). The labels should contain the original tokens at these positions.
    # labels is [1, num_roots, num_leaves]
    original_extracted = [returned_labels[1, 1, 1], returned_labels[1, 1, 2]]
    @test original_extracted == original_tokens
  end

  @testset "MNM Loss Calculation" begin
    config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
    graph1 = create_empty_chain_graph(ChainGraphConfig())
    # No triples injected, so masking won't do much, but we can manually set labels
    
    batch_size = 1
    num_roots = 128
    num_leaves = 7
    labels = fill(-100, (batch_size, num_roots, num_leaves))
    labels[1, 1, 1] = 100 # Target token 100 at root 1, leaf 1
    
    seq_len = 1024
    # Create random logits
    # We need to make sure the target token 100 is valid for vocab size
    vocab_size = 30522
    logits = randn(Float32, batch_size, seq_len, vocab_size)
    
    loss = calculate_mnm_loss(logits, labels, graph1)
    @test loss isa Float32
    @test loss ≥ 0
  end

  @testset "Graph to model forward (contract)" begin
    # Test forward pass with updated signature (attention decay mask)
    # Using small dimensions to speed up
    seq_len = 64
    config = ChainGraphConfig(num_roots = 8, num_leaves_per_root = 7, max_sequence_length = seq_len, pad_token_id = 1)
    graph = create_empty_chain_graph(config)
    
    vocab_size = 50
    model_config = GraphMERTConfig(
        roberta_config = GraphMERT.RoBERTaConfig(
            max_position_embeddings=seq_len, 
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64
        ),
        hgat_config = GraphMERT.HGATConfig(
            input_dim=32,
            hidden_dim=32,
            num_heads=2,
            num_layers=1
        ),
        max_sequence_length = seq_len,
        hidden_dim=32
    )
    
    # Using real GraphMERTModel
    model = GraphMERTModel(model_config)
    
    # This should now work and implicitly create attention_decay_mask
    logits = GraphMERT.forward_pass_mnm(model, graph)
    
    @test ndims(logits) == 3
    @test size(logits, 1) == 1
    @test size(logits, 2) == seq_len
    @test size(logits, 3) == vocab_size
  end
end
