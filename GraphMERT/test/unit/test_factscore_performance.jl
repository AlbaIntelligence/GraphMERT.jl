using Test
using GraphMERT

@testset "FActScore Performance Verification" begin
    @info "Starting FActScore performance test"
    
    # Create a large synthetic knowledge graph
    n_entities = 5000
    n_relations = 50000
    
    @info "Generating KG with $n_entities entities and $n_relations relations..."
    
    entities = [GraphMERT.KnowledgeEntity("e_$i", "Entity $i", "TYPE", 1.0, GraphMERT.TextPosition(0,0,0,0)) for i in 1:n_entities]
    
    relations = Vector{GraphMERT.KnowledgeRelation}()
    # Pre-allocate relations
    sizehint!(relations, n_relations)
    
    for i in 1:n_relations
        head_idx = rand(1:n_entities)
        tail_idx = rand(1:n_entities)
        while head_idx == tail_idx
            tail_idx = rand(1:n_entities)
        end
        
        push!(relations, GraphMERT.KnowledgeRelation(
            "e_$(head_idx)",     # head
            "e_$(tail_idx)",     # tail
            "RELATION",          # relation_type
            0.8,                 # confidence
            Dict{String,Any}()   # attributes
        ))
    end
    
    kg = GraphMERT.KnowledgeGraph(entities, relations, Dict{String,Any}())
    
    # Measure time for filter_triples_by_confidence
    @info "Running filter_triples_by_confidence..."
    t_start = time()
    filtered = GraphMERT.filter_triples_by_confidence(kg, 0.5)
    t_end = time()
    
    elapsed = t_end - t_start
    @info "Filtering $(n_relations) relations with $(n_entities) entities took $(elapsed) seconds"
    
    # Assert it's fast (O(M+N)) - should be sub-second for this size
    @test elapsed < 2.0  # Generous threshold, typical should be < 0.1s
    @test length(filtered) == n_relations # All should pass threshold 0.5
    
    # Test valid indices
    for (h, r, t) in filtered
        @test 1 <= h <= n_entities
        @test 1 <= t <= n_entities
        @test 1 <= r <= n_relations
        @test h != t
    end
end
