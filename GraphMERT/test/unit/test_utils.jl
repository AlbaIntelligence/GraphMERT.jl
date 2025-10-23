using Test
using GraphMERT
using Dates

@testset "Utility Functions Tests" begin
  @testset "merge_knowledge_graphs" begin
    # Create test entities
    entity1 = BiomedicalEntity("diabetes", "DISEASE", "C001", 0.8, TextPosition(1, 8, 1, 8))
    entity2 = BiomedicalEntity("insulin", "DRUG", "C002", 0.9, TextPosition(10, 16, 10, 16))
    entity3 = BiomedicalEntity("glucose", "CHEMICAL", "C003", 0.7, TextPosition(20, 26, 20, 26))

    # Create test relations
    relation1 = BiomedicalRelation("diabetes", "insulin", "treats", 0.8)
    relation2 = BiomedicalRelation("diabetes", "glucose", "affects", 0.7)

    # Create test knowledge graphs
    kg1 = KnowledgeGraph([entity1], [relation1], Dict{String,Any}("source" => "doc1"), now())
    kg2 = KnowledgeGraph([entity2, entity3], [relation2], Dict{String,Any}("source" => "doc2"), now())

    # Test merging
    merged_kg = merge_knowledge_graphs([kg1, kg2])

    @test merged_kg !== nothing
    @test length(merged_kg.entities) == 3
    @test length(merged_kg.relations) == 2
    @test haskey(merged_kg.metadata, "merged")
    @test merged_kg.metadata["source_graphs"] == 2

    # Test empty input
    empty_kg = merge_knowledge_graphs([])
    @test empty_kg !== nothing
    @test length(empty_kg.entities) == 0
    @test length(empty_kg.relations) == 0
    @test haskey(empty_kg.metadata, "merged")

    # Test single graph
    single_kg = merge_knowledge_graphs([kg1])
    @test single_kg !== nothing
    @test length(single_kg.entities) == 1
    @test length(single_kg.relations) == 1
  end

  @testset "filter_knowledge_graph" begin
    # Create test entities with different confidences
    entity1 = BiomedicalEntity("diabetes", "DISEASE", "C001", 0.9, TextPosition(1, 8, 1, 8))
    entity2 = BiomedicalEntity("insulin", "DRUG", "C002", 0.5, TextPosition(10, 16, 10, 16))
    entity3 = BiomedicalEntity("glucose", "CHEMICAL", "C003", 0.8, TextPosition(20, 26, 20, 26))

    # Create test relations
    relation1 = BiomedicalRelation("diabetes", "insulin", "treats", 0.9)
    relation2 = BiomedicalRelation("diabetes", "glucose", "affects", 0.6)

    # Create test knowledge graph
    kg = KnowledgeGraph([entity1, entity2, entity3], [relation1, relation2], Dict{String,Any}("source" => "test"), now())

    # Test confidence filtering
    filtered_kg = filter_knowledge_graph(kg, min_confidence=0.7)
    @test filtered_kg !== nothing
    @test length(filtered_kg.entities) == 2  # Only high confidence entities
    @test length(filtered_kg.relations) == 1  # Only high confidence relations
    @test haskey(filtered_kg.metadata, "filtered")
    @test filtered_kg.metadata["min_confidence"] == 0.7

    # Test entity type filtering
    type_filtered_kg = filter_knowledge_graph(kg, entity_types=["DISEASE", "CHEMICAL"])
    @test length(type_filtered_kg.entities) == 2  # Only DISEASE and CHEMICAL
    @test all(e.label in ["DISEASE", "CHEMICAL"] for e in type_filtered_kg.entities)

    # Test relation type filtering
    rel_filtered_kg = filter_knowledge_graph(kg, relation_types=["treats"])
    @test length(rel_filtered_kg.relations) == 1  # Only treats relations
    @test all(r.relation_type == "treats" for r in rel_filtered_kg.relations)

    # Test combined filtering
    combined_filtered_kg = filter_knowledge_graph(kg, min_confidence=0.7, entity_types=["DISEASE"])
    @test length(combined_filtered_kg.entities) == 1  # Only high confidence DISEASE
    @test combined_filtered_kg.entities[1].label == "DISEASE"
  end

  @testset "Edge Cases" begin
    # Test with empty knowledge graph
    empty_kg = KnowledgeGraph(BiomedicalEntity[], BiomedicalRelation[], Dict{String,Any}("empty" => true), now())

    # Test merging empty graphs
    merged_empty = merge_knowledge_graphs([empty_kg])
    @test merged_empty !== nothing
    @test length(merged_empty.entities) == 0
    @test length(merged_empty.relations) == 0

    # Test filtering empty graph
    filtered_empty = filter_knowledge_graph(empty_kg, min_confidence=0.5)
    @test filtered_empty !== nothing
    @test length(filtered_empty.entities) == 0
    @test length(filtered_empty.relations) == 0
  end
end

println("✅ Utility Functions Tests Complete!")
