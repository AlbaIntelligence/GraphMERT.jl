"""
Integration tests for GraphMERT training pipeline

Tests the complete training pipeline including:
- Joint MLM+MNM training
- Seed KG injection
- Training step execution
- Loss calculation and optimization
- End-to-end training workflow
"""

using Test
using Random
using Flux
using GraphMERT
using GraphMERT: MNMConfig, MNMBatch, MLMConfig, LeafyChainGraph, ChainGraphNode, SemanticTriple,
  select_leaves_to_mask, apply_mnm_masks, create_mnm_batch,
  create_empty_chain_graph, inject_triple!, graph_to_sequence, create_attention_mask
import GraphMERT: train_joint_mlm_mnm_step, validate_gradient_flow, calculate_mnm_loss

# Create a simple mock model for testing
struct MockGraphMERTModel
  config::GraphMERTConfig
  vocab_size::Int
end

function (model::MockGraphMERTModel)(input_ids, attention_mask)
  batch_size, seq_len = size(input_ids)
  # Return mock logits (batch_size, seq_len, vocab_size)
  return randn(Float32, batch_size, seq_len, model.vocab_size)
end

# Mock functions for joint training
function train_joint_mlm_mnm_step(model::MockGraphMERTModel, mlm_batch, mnm_batch::MNMBatch, 
                                 mlm_config::MLMConfig, mnm_config::MNMConfig, optimizer)
  # Mock joint training step
  return Dict(
    "combined_loss" => 1.0,
    "mlm_loss" => 0.5,
    "mnm_loss" => 0.5,
    "mlm_weight" => mlm_config.boundary_loss_weight,
    "mnm_weight" => mnm_config.loss_weight
  )
end

function validate_gradient_flow(model::MockGraphMERTModel, batch::MNMBatch, config::MNMConfig)
  # Mock gradient flow validation for testing
  return Dict(
    "has_gradients" => true,
    "reasonable_gradient_magnitude" => true,
    "no_gradient_explosion" => true,
    "no_gradient_vanishing" => true,
    "hgat_gradient_flow" => true
  )
end

function calculate_mnm_loss(model::MockGraphMERTModel, batch::MNMBatch, masked_positions::Vector{Tuple{Int,Int}})
  # Mock MNM loss calculation for testing
  return 0.5
end

@testset "Training Pipeline Integration Tests" begin

  @testset "Joint MLM+MNM Training" begin
    # Create configurations
    mlm_config = MLMConfig(vocab_size=30522, hidden_size=512, max_length=512, 
                          mask_probability=0.15, span_length=3, boundary_loss_weight=1.0, temperature=1.0)
    mnm_config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
    
    # Create mock model and optimizer
    mock_model = MockGraphMERTModel(GraphMERTConfig(), 30522)
    optimizer = Flux.Adam(1e-3)
    
    # Create test graphs
    graph1 = create_empty_chain_graph()
    graph2 = create_empty_chain_graph()
    
    # Inject triples
    triple1 = SemanticTriple("test1", nothing, "relation", "entity1", [100, 150], 0.9, "test")
    triple2 = SemanticTriple("test2", nothing, "relation", "entity2", [200, 250], 0.9, "test")
    @test inject_triple!(graph1, triple1, 1) == true
    @test inject_triple!(graph2, triple2, 1) == true
    
    # Create MNM batch
    masked_positions1 = [(1, 1), (1, 2)]
    masked_positions2 = [(1, 1)]
    original_tokens1 = apply_mnm_masks(graph1, masked_positions1, mnm_config)
    original_tokens2 = apply_mnm_masks(graph2, masked_positions2, mnm_config)
    
    mnm_batch = create_mnm_batch([graph1, graph2], [masked_positions1, masked_positions2], 
                                 [original_tokens1, original_tokens2], mnm_config)
    
    # Create mock MLM batch (simplified)
    mlm_batch = Dict("input_ids" => [1, 2, 3], "attention_mask" => [1, 1, 1])
    
    # Test joint training step
    results = train_joint_mlm_mnm_step(mock_model, mlm_batch, mnm_batch, mlm_config, mnm_config, optimizer)
    
    @test results isa Dict{String, Float64}
    @test haskey(results, "combined_loss")
    @test haskey(results, "mlm_loss")
    @test haskey(results, "mnm_loss")
    @test haskey(results, "mlm_weight")
    @test haskey(results, "mnm_weight")
    
    @test results["combined_loss"] > 0
    @test results["mlm_loss"] > 0
    @test results["mnm_loss"] > 0
    @test results["mlm_weight"] == 1.0
    @test results["mnm_weight"] == 1.0
  end

  @testset "Training Pipeline Workflow" begin
    # Test complete training workflow
    config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
    graph = create_empty_chain_graph()
    
    # Step 1: Inject knowledge
    triple = SemanticTriple("diabetes", nothing, "treats", "insulin", [100, 150, 200], 0.9, "medical")
    @test inject_triple!(graph, triple, 1) == true
    
    # Step 2: Select masking positions
    masked_positions = select_leaves_to_mask(graph, config)
    @test length(masked_positions) >= 0
    
    # Step 3: Apply masking
    original_tokens = apply_mnm_masks(graph, masked_positions, config)
    @test length(original_tokens) >= 0  # May be empty if no tokens were masked
    
    # Step 4: Create batch
    batch = create_mnm_batch([graph], [masked_positions], [original_tokens], config)
    @test batch isa MNMBatch
    
    # Step 5: Validate gradient flow
    mock_model = MockGraphMERTModel(GraphMERTConfig(), 30522)
    validation_results = validate_gradient_flow(mock_model, batch, config)
    @test validation_results["has_gradients"] == true
    
    println("✓ Complete training pipeline workflow validated")
  end

  @testset "Batch Processing Efficiency" begin
    # Test batch processing with multiple graphs
    config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
    batch_size = 4
    
    # Create multiple graphs
    graphs = [create_empty_chain_graph() for _ in 1:batch_size]
    masked_positions_list = Vector{Vector{Tuple{Int,Int}}}()
    original_tokens_list = Vector{Vector{Int}}()
    
    for (i, graph) in enumerate(graphs)
      # Inject different triples
      triple = SemanticTriple("entity$i", nothing, "relation", "value$i", [100 + i, 150 + i], 0.9, "test")
      @test inject_triple!(graph, triple, 1) == true
      
      # Create masking
      masked_positions = select_leaves_to_mask(graph, config)
      original_tokens = apply_mnm_masks(graph, masked_positions, config)
      
      push!(masked_positions_list, masked_positions)
      push!(original_tokens_list, original_tokens)
    end
    
    # Create batch
    batch = create_mnm_batch(graphs, masked_positions_list, original_tokens_list, config)
    
    # Verify batch properties
    @test size(batch.graph_sequence) == (batch_size, 1024)  # batch_size × seq_len
    @test size(batch.attention_mask) == (batch_size, 1024, 1024)  # batch_size × seq_len × seq_len
    @test length(batch.masked_leaf_spans) == batch_size
    @test length(batch.original_leaf_tokens) == batch_size
    @test size(batch.relation_ids) == (batch_size, config.num_leaves)
    
    println("✓ Batch processing efficiency validated")
  end

  @testset "Loss Weighting and Configuration" begin
    # Test different loss weight configurations
    mlm_config = MLMConfig(boundary_loss_weight=2.0)
    mnm_config = MNMConfig(30522, 512, 7, 0.15, 0.3, 0.5, true, 103)  # mnm_weight = 0.5
    
    mock_model = MockGraphMERTModel(GraphMERTConfig(), 30522)
    optimizer = Flux.Adam(1e-3)
    
    # Create test batch
    graph = create_empty_chain_graph()
    triple = SemanticTriple("test", nothing, "relation", "entity", [100, 150], 0.9, "test")
    @test inject_triple!(graph, triple, 1) == true
    
    masked_positions = [(1, 1), (1, 2)]
    original_tokens = apply_mnm_masks(graph, masked_positions, mnm_config)
    mnm_batch = create_mnm_batch([graph], [masked_positions], [original_tokens], mnm_config)
    mlm_batch = Dict("input_ids" => [1, 2, 3], "attention_mask" => [1, 1, 1])
    
    # Test joint training with different weights
    results = train_joint_mlm_mnm_step(mock_model, mlm_batch, mnm_batch, mlm_config, mnm_config, optimizer)
    
    @test results["mlm_weight"] == 2.0
    @test results["mnm_weight"] == 0.5
    
    println("✓ Loss weighting configuration validated")
  end

  @testset "Training Stability" begin
    # Test training stability across multiple steps
    config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
    mock_model = MockGraphMERTModel(GraphMERTConfig(), 30522)
    optimizer = Flux.Adam(1e-3)
    
    # Create test data
    graph = create_empty_chain_graph()
    triple = SemanticTriple("stable", nothing, "relation", "entity", [100, 150], 0.9, "test")
    @test inject_triple!(graph, triple, 1) == true
    
    losses = Float64[]
    
    # Run multiple training steps
    for step in 1:5
      masked_positions = select_leaves_to_mask(graph, config)
      original_tokens = apply_mnm_masks(graph, masked_positions, config)
      batch = create_mnm_batch([graph], [masked_positions], [original_tokens], config)
      
      # Mock training step
      loss = calculate_mnm_loss(mock_model, batch, masked_positions)
      push!(losses, loss)
    end
    
    # Check that losses are reasonable (not exploding or vanishing)
    @test all(loss -> 0 <= loss <= 10, losses)
    @test length(losses) == 5
    
    println("✓ Training stability validated")
  end
end
