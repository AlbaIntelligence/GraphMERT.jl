"""
Integration tests for GraphMERT evaluation pipeline

This module tests the complete evaluation pipeline including FActScore,
ValidityScore, and GraphRAG evaluation functionality.
"""

using Test
using GraphMERT

@testset "Evaluation Pipeline Integration" begin

  @testset "Complete Evaluation Pipeline" begin
    # Create comprehensive test knowledge graph
    entities = [
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.95, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.88, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("glucose", "glucose", "C0017725", 0.92, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("metformin", "metformin", "C0025598", 0.90, TextPosition(1, 10, 1, 1))
    ]

    relations = [
      BiomedicalRelation("treats", "treats", "treats", 0.85),
      BiomedicalRelation("increases", "increases", "increases", 0.45),
      BiomedicalRelation("reduces", "reduces", "reduces", 0.80)
    ]

    metadata = Dict(
      "text" => "Diabetes is treated with insulin which increases glucose levels. Metformin reduces blood sugar.",
      "confidence" => 0.8
    )
    kg = KnowledgeGraph(entities, relations, metadata)

    # Test FActScore evaluation
    text = "Diabetes is treated with insulin which increases glucose levels. Metformin reduces blood sugar."
    factscore_result = evaluate_factscore(kg, text)
    @test factscore_result isa FActScoreResult
    @test 0.0 ≤ factscore_result.factscore ≤ 1.0
    @test factscore_result.total_triples > 0

    # Test ValidityScore evaluation
    validity_result = evaluate_validity(kg)
    @test validity_result isa ValidityScoreResult
    @test 0.0 ≤ validity_result.validity_score ≤ 1.0
    @test validity_result.total_triples > 0

    # Test GraphRAG evaluation
    query = "How is diabetes treated?"
    graphrag_score = evaluate_graphrag(kg, query)
    @test graphrag_score isa Float64
    @test 0.0 ≤ graphrag_score ≤ 1.0

    println("✓ Complete evaluation pipeline works")
    println("  FActScore: $(round(factscore_result.factscore, digits=3))")
    println("  ValidityScore: $(round(validity_result.validity_score, digits=3))")
    println("  GraphRAG Score: $(round(graphrag_score, digits=3))")
  end

  @testset "Evaluation Consistency" begin
    # Test that evaluation results are consistent across multiple runs
    entities = [
      BiomedicalEntity("cancer", "cancer", "C0006826", 0.95, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("chemotherapy", "chemotherapy", "C0007994", 0.90, TextPosition(1, 10, 1, 1))
    ]

    relations = [
      BiomedicalRelation("treats", "treats", "treats", 0.85)
    ]

    metadata = Dict("text" => "Chemotherapy treats cancer.", "confidence" => 0.8)
    kg = KnowledgeGraph(entities, relations, metadata)

    # Run evaluation multiple times
    results1 = evaluate_factscore(kg, "Chemotherapy treats cancer.")
    results2 = evaluate_factscore(kg, "Chemotherapy treats cancer.")

    @test results1.factscore == results2.factscore
    @test results1.total_triples == results2.total_triples
  end

  @testset "Edge Cases" begin
    # Test with empty knowledge graph
    empty_kg = KnowledgeGraph(BiomedicalEntity[], BiomedicalRelation[], Dict("text" => "", "confidence" => 0.0))

    factscore_result = evaluate_factscore(empty_kg, "")
    @test factscore_result.factscore == 0.0
    @test factscore_result.total_triples == 0

    validity_result = evaluate_validity(empty_kg)
    @test validity_result.validity_score == 0.0
    @test validity_result.total_triples == 0

    graphrag_score = evaluate_graphrag(empty_kg, "test query")
    @test graphrag_score == 0.0
  end

end
