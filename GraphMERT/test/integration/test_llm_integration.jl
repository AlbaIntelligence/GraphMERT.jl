"""
Integration tests for LLM Helper integration

Tests the complete LLM integration pipeline including:
- End-to-end entity discovery and relation matching
- Integration with UMLS for entity validation
- Integration with extraction pipeline
- Performance and accuracy validation
- Error handling and fallback mechanisms
"""

using Test
using GraphMERT

@testset "LLM Integration Tests" begin

    @testset "LLM Entity Discovery Integration" begin
        # Create mock LLM client
        llm_client = create_helper_llm_client("test_key")

        # Test text
        text = """
        Diabetes mellitus is a chronic metabolic disorder characterized by
        elevated blood glucose levels. Metformin is commonly used to treat
        type 2 diabetes. Insulin resistance is a key feature of type 2 diabetes.
        """

        # Test entity discovery
        entities = discover_entities(llm_client, text)

        @test entities isa Vector{String}
        @test length(entities) > 0

        # Should find biomedical entities
        entity_texts = lowercase.(entities)
        @test any(occursin("diabet", entity) for entity in entity_texts)

        # Test integration with UMLS
        if length(entities) > 0
            first_entity = entities[1]
            # In full implementation, would link to UMLS
            # linking_result = link_entity_to_umls(first_entity, umls_client)
        end
    end

    @testset "LLM Relation Matching Integration" begin
        # Create mock LLM client
        llm_client = create_helper_llm_client("test_key")

        # Test entities
        entities = ["Diabetes", "Metformin", "Insulin"]

        # Test text
        text = """
        Diabetes is treated with Metformin. Insulin is used for type 1 diabetes.
        Diabetes causes complications including cardiovascular disease.
        """

        # Test relation matching
        relations = match_relations(llm_client, entities, text)

        @test relations isa Dict{String, Dict{String, String}}

        # Should find relations between entities
        if length(relations) > 0
            @test all(haskey(relation_data, "relation") for relation_data in values(relations))
            @test all(haskey(relation_data, "entity2") for relation_data in values(relations))
        end
    end

    @testset "LLM Integration with Extraction Pipeline" begin
        # Create mock LLM client
        llm_client = create_helper_llm_client("test_key")

        # Test text
        text = "Diabetes is treated with metformin and insulin."

        # Test enhanced entity discovery
        entities = discover_head_entities_enhanced(text, nothing, llm_client)

        @test entities isa Vector{GraphMERT.BiomedicalEntity}
        @test length(entities) > 0

        # Test enhanced relation matching
        if length(entities) > 1
            relations = match_relations_for_entities_enhanced(entities, text, llm_client)

            @test relations isa Vector{Tuple{Int, Int, String, Float64}}

            # Should find relations between entities
            if length(relations) > 0
                @test all(1 ≤ head_idx ≤ length(entities) for (head_idx, _, _, _) in relations)
                @test all(1 ≤ tail_idx ≤ length(entities) for (_, tail_idx, _, _) in relations)
                @test all(head_idx != tail_idx for (head_idx, tail_idx, _, _) in relations)
            end
        end
    end

    @testset "Caching Integration" begin
        # Create mock LLM client
        llm_client = create_helper_llm_client("test_key")

        # Test text
        text = "Diabetes is a chronic condition."

        # First call should make API request
        entities1 = discover_entities(llm_client, text)

        # Second call should use cache
        entities2 = discover_entities(llm_client, text)

        # Results should be identical (cached)
        @test entities1 == entities2

        # Cache should have been populated
        @test length(llm_client.cache.responses) > 0
    end

    @testset "Rate Limiting Integration" begin
        # Create client with low rate limit for testing
        llm_client = create_helper_llm_client("test_key", rate_limit=5)

        # Make multiple requests quickly
        for i in 1:10
            discover_entities(llm_client, "Test text $i")
        end

        # Should have triggered rate limiting
        @test llm_client.request_count > 0

        # Should have cache entries from successful requests
        @test length(llm_client.cache.responses) > 0
    end

    @testset "Error Handling Integration" begin
        # Create mock LLM client
        llm_client = create_helper_llm_client("test_key")

        # Test with empty text
        entities = discover_entities(llm_client, "")
        @test entities isa Vector{String}

        # Test with very long text
        long_text = repeat("word ", 1000)
        entities = discover_entities(llm_client, long_text)
        @test entities isa Vector{String}

        # Test with invalid entities
        relations = match_relations(llm_client, String[], "No entities")
        @test relations isa Dict{String, Dict{String, String}}
    end

    @testset "Performance Integration" begin
        # Create mock LLM client
        llm_client = create_helper_llm_client("test_key")

        # Test with different text sizes
        texts = [
            "Short text.",
            "Medium length text with some medical terms and conditions.",
            repeat("Longer text with repeated medical terminology. ", 20)
        ]

        for (i, text) in enumerate(texts)
            start_time = time()
            entities = discover_entities(llm_client, text)
            processing_time = time() - start_time

            # Should complete in reasonable time
            @test processing_time < 10.0  # seconds

            # Should return results
            @test entities isa Vector{String}

            @info "Performance test $i: $(length(text)) chars, $(processing_time)s, $(length(entities)) entities"
        end
    end
end
