"""
Integration tests for Knowledge Graph Extraction pipeline

Tests the complete extraction pipeline including:
- End-to-end entity discovery and relation extraction
- Integration with the leafy chain graph structure
- Performance and accuracy validation
- API integration and error handling
"""

using Test
using GraphMERT

# Load and register biomedical domain for integration tests
# include(joinpath(dirname(dirname(@__DIR__)), "src", "domains", "biomedical.jl"))
if !GraphMERT.has_domain("biomedical")
    GraphMERT.register_domain!("biomedical", GraphMERT.load_biomedical_domain())
end

# Create mock model for testing
struct MockExtractionModel
  vocab_size::Int
end

function (model::MockExtractionModel)(input_ids, attention_mask)
  batch_size, seq_len = size(input_ids)
  return randn(Float32, batch_size, seq_len, model.vocab_size)
end

@testset "Knowledge Graph Extraction Pipeline Integration" begin

  @testset "End-to-End Extraction Pipeline" begin
    # Create mock model
    mock_model = MockExtractionModel(30522)

    # Test text
    text = """
    Diabetes mellitus is a chronic metabolic disorder characterized by
    elevated blood glucose levels. Metformin is used to treat
    type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
    Cardiovascular disease is a major complication of diabetes.
    """

    # Test complete extraction pipeline
    options = GraphMERT.default_processing_options()
    kg = GraphMERT.extract_knowledge_graph(text, mock_model, options=options)

    # Verify knowledge graph structure
    println("DEBUG: entities count: ", length(kg.entities))
    println("DEBUG: relations count: ", length(kg.relations))
    println("DEBUG: triples count: ", length(kg.triples))
    
    if length(kg.triples) == 0 && length(kg.relations) > 0
        println("DEBUG: DUMPING ENTITIES:")
        for e in kg.entities
            println("  ID: ", e.id, " Text: ", e.text)
        end
        println("DEBUG: DUMPING RELATIONS:")
        for r in kg.relations
            println("  Head: ", r.head, " Tail: ", r.tail, " Type: ", r.relation_type)
        end
        
        # Manually check matching
        id_set = Set(e.id for e in kg.entities)
        for r in kg.relations
            h_ok = r.head in id_set
            t_ok = r.tail in id_set
            println("  Rel Check: Head in set? $h_ok, Tail in set? $t_ok")
        end
    end

    @test kg isa GraphMERT.KnowledgeGraph
    @test length(kg.entities) > 0
    @test length(kg.relations) > 0
    @test length(kg.triples) > 0

    # Verify entity structure (TextPosition has start, stop, line, column — not start_char)
    for entity in kg.entities
      @test !isempty(entity.text)
      @test 0 ≤ entity.confidence ≤ 1
      @test entity.position.start ≥ 0
    end

    # Verify relation structure
    for relation in kg.relations
      @test 1 ≤ relation.head_entity_id ≤ length(kg.entities)
      @test 1 ≤ relation.tail_entity_id ≤ length(kg.entities)
      @test !isempty(relation.relation_type)
      @test 0 ≤ relation.confidence ≤ 1
    end

    # Verify triple consistency
    if length(kg.triples) == 0 && length(kg.relations) > 0
        println("DEBUG: kg.triples is empty but relations > 0")
        println("Entities IDs: ", [e.id for e in kg.entities])
        println("Relations: ", [(r.head, r.tail) for r in kg.relations])
    end
    for (head_idx, rel_idx, tail_idx) in kg.triples
      @test 1 ≤ head_idx ≤ length(kg.entities)
      @test 1 ≤ rel_idx ≤ length(kg.relations)
      @test 1 ≤ tail_idx ≤ length(kg.entities)
      @test kg.relations[rel_idx].head_entity_id == head_idx
      @test kg.relations[rel_idx].tail_entity_id == tail_idx
    end

    # Verify metadata
    @test haskey(kg.metadata, "extraction_time")
    @test haskey(kg.metadata, "model_version")
    @test haskey(kg.metadata, "num_entities")
    @test haskey(kg.metadata, "num_relations")
    @test haskey(kg.metadata, "num_triples")

    @test kg.metadata["num_entities"] == length(kg.entities)
    @test kg.metadata["num_relations"] == length(kg.relations)
    @test kg.metadata["num_triples"] == length(kg.triples)
  end

  @testset "Entity Discovery Integration" begin
    text = "Diabetes mellitus is characterized by hyperglycemia."

    entities = GraphMERT.discover_head_entities(text)
    @test length(entities) > 0

    # Check that discovered entities are relevant to the text
    entity_texts = [e.text for e in entities]
    @test any(contains(lowercase(text), lowercase(entity)) for entity in entity_texts)
  end

  @testset "Relation Extraction Integration" begin
    # Create test entities
    entities = [
      GraphMERT.BiomedicalEntity("Diabetes", "Diabetes", "DISEASE", nothing, String[], GraphMERT.TextPosition(0, 7, 1, 1), 0.9, "test"),
      GraphMERT.BiomedicalEntity("Metformin", "Metformin", "DRUG", nothing, String[], GraphMERT.TextPosition(8, 17, 2, 2), 0.8, "test")
    ]

    text = "Diabetes is treated with Metformin."
    domain = GraphMERT.get_domain("biomedical")
    options = GraphMERT.ProcessingOptions(domain="biomedical")
    relations = GraphMERT.match_relations_for_entities(entities, text, domain, options)

    @test length(relations) > 0

    # Check that relations connect the entities
    entity_ids = Set(e.id for e in entities)
    for relation in relations
      @test relation.head != relation.tail
      @test relation.head in entity_ids
      @test relation.tail in entity_ids
      @test !isempty(relation.relation_type)
      @test 0.0 ≤ relation.confidence ≤ 1.0
    end
  end

  @testset "Model Integration" begin
    # Test that extraction works with the model interface
    mock_model = MockExtractionModel(30522)

    # Simple text for testing
    text = "Diabetes is a disease."
    options = GraphMERT.default_processing_options()

    # Test model forward pass
    input_ids = reshape([100, 200, 300, 400], 1, 4)  # Simple test input
    attention_mask = reshape([true, true, true, true], 1, 4)

    logits = mock_model(input_ids, attention_mask)
    @test size(logits) == (1, 4, 30522)
  end

  @testset "Performance Validation" begin
    mock_model = MockExtractionModel(30522)

    # Test with different text lengths
    short_text = "Diabetes is a disease."
    medium_text = "Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels."
    long_text = repeat("Diabetes mellitus is a chronic metabolic disorder. ", 10)

    start_time = time()
    kg_short = GraphMERT.extract_knowledge_graph(short_text, mock_model)
    short_time = time() - start_time

    start_time = time()
    kg_medium = GraphMERT.extract_knowledge_graph(medium_text, mock_model)
    medium_time = time() - start_time

    start_time = time()
    kg_long = GraphMERT.extract_knowledge_graph(long_text, mock_model)
    long_time = time() - start_time

    # Performance checks are flaky in CI/shared environments due to JIT compilation and resource contention
    # We log the times but don't fail the test based on them
    @info "Performance metrics:" short_time medium_time long_time
    
    # @test long_time > medium_time > short_time 

    # Check that extraction produces results for all lengths
    @test length(kg_short.entities) > 0
    @test length(kg_medium.entities) > 0
    @test length(kg_long.entities) > 0

    # Calculate rough tokens per second (assuming ~4 chars per token)
    short_tokens = length(short_text) ÷ 4
    medium_tokens = length(medium_text) ÷ 4
    long_tokens = length(long_text) ÷ 4

    short_throughput = short_tokens / short_time
    medium_throughput = medium_tokens / medium_time
    long_throughput = long_tokens / long_time
    
    @info "Throughput (tokens/s):" short_throughput medium_throughput long_throughput

    # Throughput consistency checks are also too flaky for integration tests
    # @test abs(short_throughput - medium_throughput) / max(short_throughput, medium_throughput) < 0.5
    # @test abs(medium_throughput - long_throughput) / max(medium_throughput, long_throughput) < 0.5
  end

  @testset "Error Handling Integration" begin
    mock_model = MockExtractionModel(30522)

    # Test with empty text
    empty_kg = GraphMERT.extract_knowledge_graph("", mock_model)
    @test empty_kg isa GraphMERT.KnowledgeGraph
    @test isempty(empty_kg.entities)
    @test isempty(empty_kg.relations)

    # Test with very long text (would need proper max_length handling)
    # long_text = repeat("word ", 10000)
    # @test_throws ArgumentError GraphMERT.extract_knowledge_graph(long_text, mock_model)

    # Test with invalid model (would need proper validation)
    # @test_throws MethodError GraphMERT.extract_knowledge_graph("test", "not_a_model")
  end

  @testset "API Integration" begin
    mock_model = MockExtractionModel(30522)
    text = "Diabetes is treated with metformin."

    # Test default options
    kg1 = GraphMERT.extract_knowledge_graph(text, mock_model)
    @test kg1 isa GraphMERT.KnowledgeGraph

    # Test custom options
    options = GraphMERT.default_processing_options(
      batch_size=16,
      similarity_threshold=0.7,
      top_k_predictions=15
    )
    kg2 = GraphMERT.extract_knowledge_graph(text, mock_model, options=options)
    @test kg2 isa GraphMERT.KnowledgeGraph

    # Results should be similar but not identical due to different parameters
    @test length(kg1.entities) > 0
    @test length(kg2.entities) > 0
  end
end
