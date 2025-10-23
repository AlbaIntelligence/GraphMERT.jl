"""
Unit tests for GraphMERT evaluation metrics

This module tests FActScore, ValidityScore, and GraphRAG evaluation functionality.
"""

using Test
using GraphMERT

@testset "FActScore Evaluation" begin

  @testset "Triple Confidence Filtering" begin
    # Create test knowledge graph with varying confidence scores
    entities = [
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.95, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.88, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("glucose", "glucose", "C0017725", 0.92, TextPosition(1, 10, 1, 1))
    ]

    relations = [
      BiomedicalRelation("treats", "treats", 0.85, entities[1], entities[2]),
      BiomedicalRelation("increases", "increases", 0.45, entities[2], entities[3])  # Low confidence
    ]

    kg = KnowledgeGraph(entities, relations, "test text", 0.8)

    # Test filtering with threshold
    filtered_kg = filter_triples_by_confidence(kg, 0.7)
    @test length(filtered_kg.relations) == 1  # Only high confidence relation should remain
    @test filtered_kg.relations[1].relation_type == "treats"
  end

  @testset "FActScore Calculation" begin
    # Create simple test case
    entities = [
      BiomedicalEntity("COVID-19", "COVID-19", "C5203670", 0.95),
      BiomedicalEntity("fever", "fever", "C0015967", 0.88)
    ]

    relations = [
      BiomedicalRelation("causes", "causes", 0.85, entities[1], entities[2])
    ]

    kg = KnowledgeGraph(entities, relations, "COVID-19 causes fever in patients", 0.9)
    text = "COVID-19 causes fever in patients with respiratory symptoms."

    # Test FActScore calculation
    factscore = evaluate_factscore(kg, text)
    @test factscore ≥ 0.0 && factscore ≤ 1.0  # Should be between 0 and 1
    @test typeof(factscore) == Float64
  end

  @testset "Triple Context Extraction" begin
    text = "Diabetes is a chronic condition that affects blood sugar levels. Insulin helps regulate glucose metabolism."
    head_entity = BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.95, TextPosition(1, 10, 1, 1))

    context = get_triple_context(text, head_entity, nothing)
    @test length(context) > 0  # Should find relevant context sentences
    @test typeof(context) == Vector{String}
  end

end

@testset "ValidityScore Evaluation" begin

  @testset "Triple Ontology Validation" begin
    # Test valid biomedical relationship
    head_entity = BiomedicalEntity("insulin", "insulin", "C0021641", 0.95)
    tail_entity = BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.88)
    relation = BiomedicalRelation("treats", "treats", 0.85, head_entity, tail_entity)

    # This should be ontologically valid (insulin treats diabetes)
    validity = validate_triple_ontology(relation)
    @test validity ≥ 0.0 && validity ≤ 1.0
  end

  @testset "Knowledge Graph Validity" begin
    entities = [
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.95),
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.88)
    ]

    relations = [
      BiomedicalRelation("treats", "treats", 0.85, entities[1], entities[2])
    ]

    kg = KnowledgeGraph(entities, relations, "test text", 0.9)

    validity_score = evaluate_validity(kg)
    @test validity_score ≥ 0.0 && validity_score ≤ 1.0
    @test typeof(validity_score) == Float64
  end

end

@testset "GraphRAG Evaluation" begin

  @testset "Local Search Implementation" begin
    # Create test knowledge graph
    entities = [
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.95, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.88, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("glucose", "glucose", "C0017725", 0.92, TextPosition(1, 10, 1, 1))
    ]

    relations = [
      BiomedicalRelation("treats", "treats", 0.85, entities[2], entities[1]),  # insulin treats diabetes
      BiomedicalRelation("increases", "increases", 0.75, entities[1], entities[3])  # diabetes increases glucose
    ]

    kg = KnowledgeGraph(entities, relations, "test biomedical text", 0.9)

    # Test local search functionality
    query = "How does insulin help with diabetes?"
    results = perform_local_search(kg, query)
    @test typeof(results) == Vector{Tuple{BiomedicalEntity,BiomedicalRelation,BiomedicalEntity,Float64}}
  end

  @testset "GraphRAG Evaluation Score" begin
    entities = [
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.95, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.88, TextPosition(1, 10, 1, 1))
    ]

    relations = [
      BiomedicalRelation("treats", "treats", 0.85, entities[2], entities[1])
    ]

    kg = KnowledgeGraph(entities, relations, "test text", 0.9)

    graphrag_score = evaluate_graphrag(kg, "test query")
    @test graphrag_score ≥ 0.0 && graphrag_score ≤ 1.0
    @test typeof(graphrag_score) == Float64
  end

end

@testset "GraphRAG Local Search" begin

  @testset "Entity Relevance Calculation" begin
    entity = BiomedicalEntity("diabetes mellitus", "diabetes", "C0011849", 0.95)
    query_terms = ["diabetes", "blood", "sugar"]

    relevance = calculate_entity_relevance(entity, query_terms)
    @test relevance ≥ 0.0 && relevance ≤ 1.0
    @test relevance > 0.0  # Should match "diabetes"
  end

  @testset "Relation Relevance Calculation" begin
    head_entity = BiomedicalEntity("insulin", "insulin", "C0021641", 0.95)
    tail_entity = BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.88)
    relation = BiomedicalRelation("treats", "treats", 0.85, head_entity, tail_entity)

    query_terms = ["insulin", "diabetes", "treatment"]
    relevance = calculate_relation_relevance(relation, query_terms)
    @test relevance ≥ 0.0 && relevance ≤ 1.0
  end

  @testset "Local Search Functionality" begin
    # Create test knowledge graph
    entities = [
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.95, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.88, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("glucose", "glucose", "C0017725", 0.92, TextPosition(1, 10, 1, 1))
    ]

    relations = [
      BiomedicalRelation("treats", "treats", 0.85, entities[2], entities[1]),  # insulin treats diabetes
      BiomedicalRelation("increases", "increases", 0.75, entities[1], entities[3])  # diabetes increases glucose
    ]

    kg = KnowledgeGraph(entities, relations, "test biomedical text", 0.9)

    # Test local search
    query = "How does insulin help with diabetes?"
    results = perform_local_search(kg, query, max_entities=5)
    @test typeof(results) == Vector{Tuple{BiomedicalEntity,BiomedicalRelation,BiomedicalEntity,Float64}}
    @test length(results) ≤ 5  # Should respect max_entities limit
  end

  @testset "Answer Generation" begin
    entities = [
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.95),
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.88)
    ]

    relations = [
      BiomedicalRelation("treats", "treats", 0.85, entities[1], entities[2])
    ]

    kg = KnowledgeGraph(entities, relations, "test text", 0.9)

    # Test answer generation
    query = "What treats diabetes?"
    results = perform_local_search(kg, query, max_entities=1)
    answer = generate_answer_from_context(results, query)

    @test typeof(answer) == String
    @test length(answer) > 0
    @test occursin("insulin", lowercase(answer)) || occursin("treats", lowercase(answer))
  end

  @testset "GraphRAG Evaluation Score" begin
    entities = [
      BiomedicalEntity("diabetes", "diabetes mellitus", "C0011849", 0.95, TextPosition(1, 10, 1, 1)),
      BiomedicalEntity("insulin", "insulin", "C0021641", 0.88, TextPosition(1, 10, 1, 1))
    ]

    relations = [
      BiomedicalRelation("treats", "treats", 0.85, entities[2], entities[1])
    ]

    kg = KnowledgeGraph(entities, relations, "test text", 0.9)

    score = evaluate_graphrag(kg, "What treats diabetes?")
    @test score ≥ 0.0 && score ≤ 1.0
    @test typeof(score) == Float64
  end

end

# Integration tests would go in a separate integration test file
# Performance tests would go in a separate performance test file
