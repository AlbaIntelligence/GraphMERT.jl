using Test
using GraphMERT

@testset "External Service Integration (Requires Keys)" begin
    
    # Check if we should run external tests
    run_external = get(ENV, "RUN_EXTERNAL_TESTS", "false") == "true"
    
    if !run_external
        @info "Skipping external service tests (RUN_EXTERNAL_TESTS not set)"
        return
    end

    @testset "Real UMLS Integration" begin
        api_key = get(ENV, "UMLS_API_KEY", "")
        if isempty(api_key)
            @warn "Skipping Real UMLS test: UMLS_API_KEY not set"
        else
            @info "Running Real UMLS test..."
            client = create_umls_client(api_key; mock_mode=false)
            
            # Simple query: "diabetes" -> C0011849
            result = link_entity_to_umls("diabetes", client)
            @test result !== nothing
            @test result.cui == "C0011849"
            @test result.entity_text == "diabetes"
            @test result.source == "umls_search"
            
            # Check semantic types
            types = get_entity_semantic_types(client, "C0011849")
            @test !isempty(types)
            @test "T047" in types # Disease or Syndrome
        end
    end

    @testset "Real OpenAI Integration" begin
        api_key = get(ENV, "OPENAI_API_KEY", "")
        if isempty(api_key)
            @warn "Skipping Real OpenAI test: OPENAI_API_KEY not set"
        else
            @info "Running Real OpenAI test..."
            client = create_helper_llm_client(api_key; model="gpt-3.5-turbo")
            
            # Simple entity discovery
            text = "Diabetes is a chronic disease."
            entities = discover_entities(client, text)
            @test !isempty(entities)
            # Expect "Diabetes" in some form (case-insensitive check)
            @test any(occursin("iabet", lowercase(e)) for e in entities)
        end
    end
end
