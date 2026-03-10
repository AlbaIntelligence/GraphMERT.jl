"""
Unit tests for GraphMERT visualization module.
"""

using Test
using GraphMERT
using GraphMERT: KnowledgeGraph, KnowledgeEntity, KnowledgeRelation, TextPosition

@testset "Visualization Module" begin

    @testset "kg_to_graphs_format" begin
        # Create a simple knowledge graph
        entities = [
            KnowledgeEntity("e1", "Alice", "PERSON", 0.9, TextPosition(0,5,1,1)),
            KnowledgeEntity("e2", "Bob", "PERSON", 0.8, TextPosition(10,13,1,11)),
            KnowledgeEntity("e3", "Google", "ORGANIZATION", 0.7, TextPosition(20,26,1,21))
        ]

        relations = [
            KnowledgeRelation("e1", "e2", "knows", 0.8),
            KnowledgeRelation("e2", "e3", "works_at", 0.6)
        ]

        kg = KnowledgeGraph(entities, relations)

        # Convert to MetaGraph
        mg = kg_to_graphs_format(kg)

        @test nv(mg) == 3  # 3 entities
        @test ne(mg) == 2  # 2 relations

        # Check node properties
        @test props(mg, 1)["id"] == "e1"
        @test props(mg, 1)["text"] == "Alice"
        @test props(mg, 1)["entity_type"] == "PERSON"
        @test props(mg, 1)["confidence"] == 0.9

        # Check edge properties
        edge1 = Edge(1, 2)
        @test has_edge(mg, edge1)
        @test props(mg, edge1)["relation_type"] == "knows"
        @test props(mg, edge1)["confidence"] == 0.8
    end

    @testset "filter_by_confidence" begin
        entities = [
            KnowledgeEntity("e1", "Alice", "PERSON", 0.9, TextPosition(0,5,1,1)),
            KnowledgeEntity("e2", "Bob", "PERSON", 0.5, TextPosition(10,13,1,11)),
            KnowledgeEntity("e3", "Google", "ORGANIZATION", 0.3, TextPosition(20,26,1,21))
        ]

        relations = [
            KnowledgeRelation("e1", "e2", "knows", 0.8),
            KnowledgeRelation("e2", "e3", "works_at", 0.4)
        ]

        kg = KnowledgeGraph(entities, relations)

        # Filter by confidence 0.6
        filtered = filter_by_confidence(kg, 0.6)

        @test length(filtered.entities) == 2  # Alice and Bob
        @test length(filtered.relations) == 1  # Only knows relation
        @test filtered.entities[1].text == "Alice"
        @test filtered.entities[2].text == "Bob"
    end

    @testset "filter_by_entity_type" begin
        entities = [
            KnowledgeEntity("e1", "Alice", "PERSON", 0.9, TextPosition(0,5,1,1),
                           Dict("entity_type" => "PERSON")),
            KnowledgeEntity("e2", "Bob", "PERSON", 0.8, TextPosition(10,13,1,11),
                           Dict("entity_type" => "PERSON")),
            KnowledgeEntity("e3", "Google", "ORGANIZATION", 0.7, TextPosition(20,26,1,21),
                           Dict("entity_type" => "ORGANIZATION"))
        ]

        relations = [
            KnowledgeRelation("e1", "e2", "knows", 0.8),
            KnowledgeRelation("e2", "e3", "works_at", 0.6)
        ]

        kg = KnowledgeGraph(entities, relations)

        # Filter for only PERSON entities
        person_kg = filter_by_entity_type(kg, ["PERSON"])

        @test length(person_kg.entities) == 2
        @test length(person_kg.relations) == 1  # Only the knows relation between persons
        @test person_kg.relations[1].relation_type == "knows"
    end

    @testset "simplify_graph" begin
        # Create a larger graph
        entities = [KnowledgeEntity("e$i", "Entity$i", "TYPE", 0.1 * i, TextPosition(0,5,1,1))
                   for i in 1:10]
        relations = [KnowledgeRelation("e$i", "e$(i+1)", "relates", 0.1 * i)
                    for i in 1:9]

        kg = KnowledgeGraph(entities, relations)

        # Simplify to max 5 nodes
        simple_kg = simplify_graph(kg, max_nodes=5, min_confidence=0.0)

        @test length(simple_kg.entities) == 5
        @test length(simple_kg.relations) <= 4  # May have fewer relations due to entity filtering

        # Check that highest confidence entities are selected
        confidences = [e.confidence for e in simple_kg.entities]
        @test all(c -> c >= 0.6, confidences)  # Should have top 5 entities (conf 1.0, 0.9, 0.8, 0.7, 0.6)
    end

    @testset "create_visualization_summary" begin
        entities = [
            KnowledgeEntity("e1", "Alice", "PERSON", 0.9, TextPosition(0,5,1,1),
                           Dict("entity_type" => "PERSON")),
            KnowledgeEntity("e2", "Bob", "PERSON", 0.8, TextPosition(10,13,1,11),
                           Dict("entity_type" => "PERSON")),
            KnowledgeEntity("e3", "Google", "ORGANIZATION", 0.7, TextPosition(20,26,1,21),
                           Dict("entity_type" => "ORGANIZATION"))
        ]

        relations = [
            KnowledgeRelation("e1", "e2", "knows", 0.8),
            KnowledgeRelation("e2", "e3", "works_at", 0.6)
        ]

        kg = KnowledgeGraph(entities, relations)

        summary = create_visualization_summary(kg)

        @test summary["num_entities"] == 3
        @test summary["num_relations"] == 2
        @test haskey(summary, "entity_types")
        @test haskey(summary, "relation_types")
        @test haskey(summary, "recommended_layout")
        @test summary["recommended_layout"] == :circular  # Small graph
    end

    @testset "cluster_entities" begin
        entities = [
            KnowledgeEntity("e1", "Alice", "PERSON", 0.9, TextPosition(0,5,1,1),
                           Dict("entity_type" => "PERSON")),
            KnowledgeEntity("e2", "Bob", "PERSON", 0.8, TextPosition(10,13,1,11),
                           Dict("entity_type" => "PERSON")),
            KnowledgeEntity("e3", "Google", "ORGANIZATION", 0.7, TextPosition(20,26,1,21),
                           Dict("entity_type" => "ORGANIZATION"))
        ]

        relations = [
            KnowledgeRelation("e1", "e2", "knows", 0.8),
            KnowledgeRelation("e2", "e3", "works_at", 0.6)
        ]

        kg = KnowledgeGraph(entities, relations)

        # Cluster by entity type
        clusters = cluster_entities(kg, :entity_type)

        @test haskey(clusters, "PERSON")
        @test haskey(clusters, "ORGANIZATION")
        @test length(clusters["PERSON"]) == 2
        @test length(clusters["ORGANIZATION"]) == 1
    end

    @testset "validate_visualization_input" begin
        # Valid graph
        entities = [KnowledgeEntity("e1", "Alice", "PERSON", 0.9, TextPosition(0,5,1,1))]
        relations = KnowledgeRelation[]
        kg = KnowledgeGraph(entities, relations)

        @test validate_visualization_input(kg) == true

        # Empty graph should error
        empty_kg = KnowledgeGraph(KnowledgeEntity[], KnowledgeRelation[])
        @test_throws ErrorException validate_visualization_input(empty_kg)
    end

end
