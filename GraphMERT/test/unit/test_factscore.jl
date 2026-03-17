using Test
using GraphMERT
using GraphMERT: KnowledgeEntity, KnowledgeRelation, KnowledgeGraph

@testset "FActScore Tests" begin

    # Setup entities and relation
    head_ent = KnowledgeEntity(
        "1", "aspirin", "DRUG", 0.9, 
        GraphMERT.TextPosition(1, 1, 0, 7),
        Dict{String,Any}()
    )
    tail_ent = KnowledgeEntity(
        "2", "pain", "SYMPTOM", 0.9,
        GraphMERT.TextPosition(1, 20, 20, 24),
        Dict{String,Any}()
    )
    relation = KnowledgeRelation(
        "1", "2", "TREATS", 0.8,
        Dict{String,Any}()
    )
    
    kg = KnowledgeGraph(
        [head_ent, tail_ent],
        [relation],
        Dict{String,Any}()
    )

    source_text = "The doctor prescribed aspirin to treat the patient's pain. It was effective."

    @testset "Context Extraction" begin
        # Test get_triple_context (not exported, so using GraphMERT.get_triple_context)
        context = GraphMERT.get_triple_context(source_text, head_ent, tail_ent, 1)
        @test !isempty(context)
        @test length(context) == 1
        @test occursin("aspirin", lowercase(context[1]))
        @test occursin("pain", lowercase(context[1]))
    end

    @testset "Heuristic Evaluation" begin
        # Test heuristic support
        is_supported = GraphMERT.evaluate_triple_heuristic(head_ent, relation, tail_ent, [source_text])
        @test is_supported == true
        
        # Test unsupported
        unrelated_text = ["The sun is shining today."]
        is_supported = GraphMERT.evaluate_triple_heuristic(head_ent, relation, tail_ent, unrelated_text)
        @test is_supported == false
    end

    @testset "LLM Evaluation" begin
        # Mock LLM Client
        struct MockFactScoreLLM <: GraphMERT.AbstractLLMClient
            response::String
        end
        
        function GraphMERT.make_llm_request(client::MockFactScoreLLM, prompt::String)
            return GraphMERT.HelperLLMResponse(true, client.response, nothing, Dict{String,Any}(), 200)
        end

        # Case 1: Supported
        client_supported = MockFactScoreLLM("SUPPORTED")
        result = GraphMERT.evaluate_factscore(kg, source_text, llm_client=client_supported)
        @test result.factscore == 1.0
        @test result.supported_triples == 1
        @test result.triple_scores[1] == :supported

        # Case 2: Contradicted
        client_contradicted = MockFactScoreLLM("CONTRADICTED")
        result = GraphMERT.evaluate_factscore(kg, source_text, llm_client=client_contradicted)
        @test result.factscore == 0.0
        @test result.contradicted_triples == 1
        @test result.triple_scores[1] == :contradicted

        # Case 3: Not Mentioned
        client_not_mentioned = MockFactScoreLLM("NOT_MENTIONED")
        result = GraphMERT.evaluate_factscore(kg, source_text, llm_client=client_not_mentioned)
        @test result.factscore == 0.0
        @test result.not_supported_triples == 1
        @test result.triple_scores[1] == :not_supported
    end

    @testset "Reference-based Evaluation" begin
        # Create a reference KG (identical to kg)
        ref_kg = KnowledgeGraph(
            [head_ent, tail_ent],
            [relation],
            Dict{String,Any}()
        )
        
        score = GraphMERT.evaluate_factscore(kg, ref_kg)
        @test score.score == 1.0
        @test score.correct_count == 1
        
        # Create a disjoint reference KG
        other_relation = KnowledgeRelation(
            "1", "2", "CAUSES", 0.8,
            Dict{String,Any}()
        )
        diff_ref_kg = KnowledgeGraph(
            [head_ent, tail_ent],
            [other_relation],
            Dict{String,Any}()
        )
        
        score = GraphMERT.evaluate_factscore(kg, diff_ref_kg)
        @test score.score == 0.0
    end

    @testset "Confidence Filtering Efficiency" begin
        # Legacy test adapted to KnowledgeEntity
        
        num_entities = 100
        num_relations = 500
        
        entities = [
            KnowledgeEntity(
                string(i),           # id
                "Entity $i",         # text
                "Entity $i",         # label
                1.0,                 # confidence
                GraphMERT.TextPosition(1, 5, 0, 0), # position
                Dict{String,Any}()   # attributes
            ) for i in 1:num_entities
        ]
        
        relations = KnowledgeRelation[]
        
        for i in 1:num_relations
            head_id = string(rand(1:num_entities))
            tail_id = string(rand(1:num_entities))
            while tail_id == head_id
                tail_id = string(rand(1:num_entities))
            end
            
            push!(relations, KnowledgeRelation(
                head_id, 
                tail_id,
                "REL", 
                0.9, 
                Dict{String,Any}() # attributes
            ))
        end
        
        kg = KnowledgeGraph(entities, relations, Dict{String,Any}())
        
        # Test filtering with valid threshold
        filtered = GraphMERT.filter_triples_by_confidence(kg, 0.5)
        @test length(filtered) == num_relations
        
        # Test filtering with high threshold (should filter everything)
        empty_filtered = GraphMERT.filter_triples_by_confidence(kg, 0.95)
        @test length(empty_filtered) == 0
    end
end
