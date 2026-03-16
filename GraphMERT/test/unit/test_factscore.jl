using Test
using GraphMERT
using GraphMERT: KnowledgeGraph, Entity, Relation, filter_triples_by_confidence

@testset "FActScore Implementation" begin
    
    @testset "Confidence Filtering Efficiency" begin
        # Verify that filtering happens over relations (O(M)), not cartesian entities (O(N^2))
        
        num_entities = 100
        num_relations = 500
        
        entities = [
            Entity(
                string(i),           # id
                "Entity $i",         # text
                "Entity $i",         # label
                "Type",              # type
                "Source",            # domain
                Dict{String,Any}(),  # attributes
                GraphMERT.TextPosition(1, 5, 0, 0), # position
                1.0,                 # confidence
                ""                   # provenance
            ) for i in 1:num_entities
        ]
        
        relations = Relation[]
        
        for i in 1:num_relations
            head_id = string(rand(1:num_entities))
            tail_id = string(rand(1:num_entities))
            while tail_id == head_id
                tail_id = string(rand(1:num_entities))
            end
            
            # Relation(head, tail, relation_type, confidence, domain, provenance, evidence, attributes, id)
            push!(relations, Relation(
                head_id, 
                tail_id,
                "REL", 
                0.9, 
                "Source",      # domain
                "Provenance",  # provenance
                "Evidence",    # evidence
                Dict{String,Any}() # attributes
            ))
        end
        
        kg = KnowledgeGraph(entities, relations, Dict{String,Any}())
        
        # Test filtering with valid threshold
        filtered = filter_triples_by_confidence(kg, 0.5)
        @test length(filtered) == num_relations
        
        # Test filtering with high threshold (should filter everything)
        empty_filtered = filter_triples_by_confidence(kg, 0.95)
        @test length(empty_filtered) == 0
        
        # Check structure of filtered triples
        if !isempty(filtered)
            (h, r, t) = filtered[1]
            @test 1 <= h <= num_entities
            @test 1 <= r <= num_relations
            @test 1 <= t <= num_entities
            @test h != t
        end
    end

    @testset "Small Graph Verification" begin
        # Create small graph with known structure
        e1 = Entity("1", "A", "A", "T", "S", Dict{String,Any}(), GraphMERT.TextPosition(0,0,0,0), 1.0, "")
        e2 = Entity("2", "B", "B", "T", "S", Dict{String,Any}(), GraphMERT.TextPosition(0,0,0,0), 1.0, "")
        e3 = Entity("3", "C", "C", "T", "S", Dict{String,Any}(), GraphMERT.TextPosition(0,0,0,0), 1.0, "")
        
        # R1: A -> B (0.9)
        r1 = Relation("1", "2", "REL", 0.9, "S", "P", "E", Dict{String,Any}())
        # R2: B -> C (0.4)
        r2 = Relation("2", "3", "REL", 0.4, "S", "P", "E", Dict{String,Any}())
        
        kg = KnowledgeGraph([e1, e2, e3], [r1, r2], Dict{String,Any}())
        
        # Threshold 0.5 -> Should keep R1 only
        filtered_05 = filter_triples_by_confidence(kg, 0.5)
        @test length(filtered_05) == 1
        @test filtered_05[1][2] == 1 # relation index 1
        
        # Threshold 0.3 -> Should keep R1 and R2
        filtered_03 = filter_triples_by_confidence(kg, 0.3)
        @test length(filtered_03) == 2
    end
end
