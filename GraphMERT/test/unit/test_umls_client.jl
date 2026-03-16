using Test
using GraphMERT
using Dates
using HTTP
using JSON

# Include the file to test (it defines UMLSClient, UMLSConfig, etc.)
# We need to include it in the current scope.
# Note: In a real app, this would be loaded via the domain loader.
include(joinpath(@__DIR__, "../../src/domains/biomedical/umls.jl"))

@testset "UMLS Client" begin
    @testset "Configuration" begin
        client = create_umls_client("mock-key"; mock_mode=true)
        @test client.config.api_key == "mock-key"
        @test client.config.mock_mode == true
        @test client.config.base_url == "https://uts-ws.nlm.nih.gov/rest"
        
        client_real = create_umls_client("real-key"; mock_mode=false)
        @test client_real.config.mock_mode == false
    end

    @testset "Mock Linking" begin
        client = create_umls_client("mock"; mock_mode=true)
        
        # Test linking
        result = link_entity_to_umls("diabetes", client)
        @test result isa EntityLinkingResult
        @test result.entity_text == "diabetes"
        @test result.cui != ""
        @test result.source == "UMLS-MOCK"
        @test result.semantic_types == ["T047"] # Disease or Syndrome
        
        # Test CUI retrieval
        cui = get_entity_cui(client, "diabetes")
        @test cui == result.cui
        
        # Test empty input
        @test link_entity_to_umls("", client) === nothing
    end

    @testset "Mock Relations" begin
        client = create_umls_client("mock"; mock_mode=true)
        cui = "C0011849" # Diabetes
        
        response = get_relations(client, cui)
        @test response.success == true
        @test haskey(response.data, "result")
        
        # Check caching
        @test haskey(client.cache.relations, "relations:$cui")
    end

    @testset "Mock Semantic Types" begin
        client = create_umls_client("mock"; mock_mode=true)
        cui = "C0011849"
        
        types = get_entity_semantic_types(client, cui)
        @test types == ["T047"]
        
        # Check caching
        @test haskey(client.cache.concepts, "semantic_types:$cui")
    end
    
    @testset "Batch Linking" begin
        client = create_umls_client("mock"; mock_mode=true)
        entities = ["diabetes", "cancer", "flu"]
        
        results = link_entities_batch(client, entities)
        @test length(results) == 3
        @test haskey(results, "diabetes")
        @test haskey(results, "cancer")
        @test haskey(results, "flu")
    end
    
    @testset "Retrieve Triples" begin
        client = create_umls_client("mock"; mock_mode=true)
        cui = "C0011849"
        
        triples = retrieve_triples(client, cui)
        @test length(triples) == 2
        @test triples[1] isa UMLSTriple
        @test triples[1].cui == cui
        @test triples[1].relation_label == "TREATS"
        @test triples[1].related_cui == "C12345"
        
        # Test filtering
        filtered_triples = retrieve_triples(client, cui, ["TREATS"])
        @test length(filtered_triples) == 1
        @test filtered_triples[1].relation_label == "TREATS"
        
        filtered_empty = retrieve_triples(client, cui, ["INVALID"])
        @test isempty(filtered_empty)
    end
end
