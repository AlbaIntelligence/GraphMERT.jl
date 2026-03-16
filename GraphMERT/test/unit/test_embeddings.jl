using Test
using GraphMERT
using LinearAlgebra
using Statistics

@testset "Embedding Clients" begin
    
    @testset "Cosine Similarity" begin
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        v3 = [0.0, 1.0]
        v4 = [-1.0, 0.0]
        
        @test cosine_similarity(v1, v2) ≈ 1.0f0
        @test cosine_similarity(v1, v3) ≈ 0.0f0
        @test cosine_similarity(v1, v4) ≈ -1.0f0
        
        # Zero vector handling
        @test cosine_similarity(v1, [0.0, 0.0]) == 0.0f0
    end
    
    @testset "MockEmbeddingClient" begin
        client = create_mock_embedding_client(128; deterministic=true)
        
        vec1 = embed(client, "Hello World")
        @test length(vec1) == 128
        @test norm(vec1) ≈ 1.0f0 atol=1e-5
        
        # Determinism check
        vec2 = embed(client, "Hello World")
        @test vec1 == vec2
        
        vec3 = embed(client, "Different Text")
        @test vec1 != vec3
        
        # Batch embedding
        batch = embed_batch(client, ["A", "B"])
        @test length(batch) == 2
        @test length(batch[1]) == 128
    end
    
    # We can add Gemini test here if we mock the request_fn
    # But MockEmbeddingClient is sufficient for now to test the interface
end
