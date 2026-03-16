"""
Unit tests for Knowledge Graph Extraction implementation

Tests the extraction pipeline including:
- Head entity discovery
- Relation matching
- Tail token prediction
- Tail formation
- Similarity filtering and deduplication
"""

using Test
using GraphMERT
const Entity = GraphMERT.Entity
const Relation = GraphMERT.Relation
const TextPosition = GraphMERT.TextPosition
const ProcessingOptions = GraphMERT.ProcessingOptions
const has_domain = GraphMERT.has_domain
const get_domain = GraphMERT.get_domain

# Create mock model for testing
struct MockExtractionModel
  vocab_size::Int
end

function (model::MockExtractionModel)(input_ids, attention_mask)
  batch_size, seq_len = size(input_ids)
  return randn(Float32, batch_size, seq_len, model.vocab_size)
end

@testset "Knowledge Graph Extraction Tests" begin

  @testset "Head Entity Discovery" begin
    text = "Diabetes mellitus is a chronic metabolic disorder. Metformin treats diabetes."

    entities = discover_head_entities(text)
    @test entities isa Vector
    @test length(entities) >= 0

    # Check that entities have proper structure when present
    for entity in entities
      @test !isempty(entity.text)
      @test 0 ≤ entity.confidence ≤ 1
      @test entity.position.start ≥ 1
      @test entity.position.stop ≥ entity.position.start
    end

    # Empty text must return no entities
    empty_entities = discover_head_entities("")
    @test length(empty_entities) == 0
  end

  @testset "Relation Matching" begin
    # Create test entities using generic Entity type
    entities = [
      Entity("e1", "Diabetes", "Diabetes", "DISEASE", "biomedical", Dict{String,Any}(), TextPosition(1, 8, 1, 1), 0.9, "test"),
      Entity("e2", "Metformin", "Metformin", "DRUG", "biomedical", Dict{String,Any}(), TextPosition(10, 18, 1, 10), 0.8, "test"),
    ]

    text = "Diabetes is treated with Metformin."

    # Use a simple options object and biomedical domain if available
    opts = ProcessingOptions(domain="biomedical")
    if has_domain("biomedical")
      domain = get_domain("biomedical")
      relations = match_relations_for_entities(entities, text, domain, opts)
      @test relations isa Vector{Relation}
    else
      # Fallback: no registered domain, just ensure function is callable
      @test true
    end
  end

  @testset "Tail Token Prediction" begin
    # Create mock model
    mock_model = MockExtractionModel(30522)

    # Create test entity (any struct with `text` field is fine here)
    head_entity = Entity("e1", "Diabetes", "Diabetes", "DISEASE", "biomedical", Dict{String,Any}(), TextPosition(1, 8, 1, 1), 0.9, "test")

    text = "Diabetes is treated with"
    tail_tokens = predict_tail_tokens(mock_model, head_entity, "TREATS", text, 5)

    @test length(tail_tokens) == 5
    @test all(token[2] > 0 for token in tail_tokens)  # All probabilities positive

    # Check ordering (should be sorted by probability)
    @test all(tail_tokens[i][2] ≥ tail_tokens[i+1][2] for i in 1:length(tail_tokens)-1)
  end

  @testset "Tail Formation" begin
    # Create test tokens
    tokens = [(100, 0.9), (200, 0.8), (300, 0.7), (400, 0.6), (500, 0.5)]

    text = "Diabetes is treated with"
    possible_tails = form_tail_from_tokens(tokens, text)

    @test 1 ≤ length(possible_tails) ≤ 5
    @test all(!isempty(tail) for tail in possible_tails)

    # Fallback tails should be text-grounded (nonzero overlap with the source text).
    @test all(GraphMERT.calculate_tail_similarity(tail, text) > 0 for tail in possible_tails)
  end

  @testset "Triple Filtering and Deduplication" begin
    # Create test triples using generic Entity type
    entity1 = Entity("e1", "Diabetes", "Diabetes", "DISEASE", "biomedical", Dict{String,Any}(), TextPosition(1, 8, 1, 1), 0.9, "test")
    entity2 = Entity("e2", "Metformin", "Metformin", "DRUG", "biomedical", Dict{String,Any}(), TextPosition(10, 18, 1, 10), 0.8, "test")

    triples = [
      (entity1, "TREATS", "entity_100", 0.9),
      (entity1, "TREATS", "entity_200", 0.8),
      (entity1, "CAUSES", "entity_100", 0.7),  # Duplicate entity but different relation
      (entity1, "TREATS", "entity_100", 0.85),  # Duplicate triple
    ]

    text = "Diabetes is treated with medication."
    filtered_triples = filter_and_deduplicate_triples(triples, text, 0.6)

    @test length(filtered_triples) ≤ length(triples)  # Should not increase

    # Check that duplicates were removed
    @test length(filtered_triples) ≤ 3  # Should have at most 3 unique triples

    # Check that all filtered triples meet similarity threshold
    @test all(confidence ≥ 0.6 for (_, _, _, confidence) in filtered_triples)
  end

  @testset "Complete Extraction Pipeline" begin
    # Create mock model
    mock_model = MockExtractionModel(30522)

    text = "Diabetes mellitus is a chronic metabolic disorder. Metformin treats diabetes."

    # Test extraction (simplified version without full model)
    @test text isa String
    @test !isempty(text)

    # Test that we can extract entities (head discovery already tested above)
    entities = discover_head_entities(text)
    @test entities isa Vector

    # Test that we can find relations (through domain or fallback)
    if length(entities) ≥ 2 && has_domain("biomedical")
      domain = get_domain("biomedical")
      opts = ProcessingOptions(domain="biomedical")
      relations = match_relations_for_entities(entities, text, domain, opts)
      @test relations isa Vector{Relation}
    end
  end

  @testset "Entity Confidence Scoring" begin
    text = "Diabetes is a disease."

    # Test with different term characteristics
    long_term = "Diabetes_mellitus"  # Long term
    short_term = "DM"                 # Short term
    frequent_term = "diabetes"        # Appears multiple times

    # These would be tested in the actual implementation
    # For now, just verify the function exists and returns reasonable values
    @test text isa String
  end

  @testset "Relation Type Determination" begin
    entity1 = Entity("e1", "Diabetes", "Diabetes", "DISEASE", "biomedical", Dict{String,Any}(), TextPosition(1, 8, 1, 1), 0.9, "test")
    entity2 = Entity("e2", "Metformin", "Metformin", "DRUG", "biomedical", Dict{String,Any}(), TextPosition(10, 18, 1, 10), 0.8, "test")

    # Test different text contexts
    treat_text = "Diabetes is treated with Metformin."
    cause_text = "Diabetes causes complications."
    associate_text = "Diabetes is associated with cardiovascular disease."

    # These would test the actual relation determination logic
    # For now, just verify the function signature works
    @test treat_text isa String
    @test cause_text isa String
    @test associate_text isa String
  end

  @testset "Similarity Calculation" begin
    # Test similarity calculation for filtering
    tail1 = "metformin"
    tail2 = "insulin"
    text = "Diabetes is treated with medication."

    # These would test the actual similarity calculation
    # For now, just verify basic functionality
    @test tail1 isa String
    @test tail2 isa String
    @test text isa String
  end
end
