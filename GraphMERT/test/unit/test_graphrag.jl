using Test
using GraphMERT

@testset "GraphRAG Evaluation Tests" begin
    # Setup test data
    head_ent = GraphMERT.KnowledgeEntity(
        "1", "aspirin", "DRUG", 0.9, 
        GraphMERT.TextPosition(1, 1, 0, 7),
        Dict{String,Any}()
    )
    tail_ent = GraphMERT.KnowledgeEntity(
        "2", "headache", "DISEASE", 0.9,
        GraphMERT.TextPosition(1, 15, 15, 23),
        Dict{String,Any}()
    )
    relation = GraphMERT.KnowledgeRelation(
        "1", "2", "TREATS", 0.8,
        Dict{String,Any}()
    )
    
    kg = GraphMERT.KnowledgeGraph(
        [head_ent, tail_ent],
        [relation],
        Dict{String,Any}()
    )

    @testset "Local Search" begin
        # Test basic retrieval
        results = GraphMERT.perform_local_search(kg, "aspirin headache")
        @test !isempty(results)
        @test length(results) <= 10
        
        # Check result structure
        (h, r, t, score) = results[1]
        @test h isa GraphMERT.KnowledgeEntity
        @test r isa GraphMERT.KnowledgeRelation
        @test t isa GraphMERT.KnowledgeEntity
        @test score > 0.0
    end

    @testset "Answer Generation" begin
        # Mock LLM Client
        struct MockRAGLLM <: GraphMERT.AbstractLLMClient
            response::String
        end
        
        function GraphMERT.make_llm_request(client::MockRAGLLM, prompt::String)
            return GraphMERT.HelperLLMResponse(true, client.response, nothing, Dict{String,Any}(), 200)
        end

        client = MockRAGLLM("Aspirin treats headache.")
        config = GraphMERT.GraphRAGConfig(llm_client=client)
        
        # Test with LLM
        answer = GraphMERT.evaluate_graphrag(
            kg, "What treats headache?",
            config=config
        )
        @test answer isa Float64 # Current implementation returns a score
    end
end
