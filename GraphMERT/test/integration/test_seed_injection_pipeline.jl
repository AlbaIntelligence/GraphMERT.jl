using Test
using GraphMERT
using Dates

# Include the biomedical domain module directly since it's not exported by default
# Resolve path relative to this file
# include(joinpath(dirname(dirname(dirname(@__FILE__))), "src", "domains", "biomedical.jl"))

@testset "Seed Injection Pipeline (Stream D1)" begin
    # 1. Setup Mock Services
    
    # Mock UMLS Client
    umls_client = create_umls_client("mock"; mock_mode=true)
    
    # Mock Entity Linker (simulating SapBERT)
    # Define a local mock linker for control over mappings
    struct TestMockLinker <: AbstractEntityLinker end
    
    function GraphMERT.link_entity(linker::TestMockLinker, text::String; top_k::Int=1)
        text_lower = lowercase(text)
        if text_lower == "diabetes mellitus" || text_lower == "diabetes"
            return [("C0011849", 1.0, "Diabetes Mellitus")]
        elseif text_lower == "cancer"
            return [("C0006826", 1.0, "Cancer")]
        end
        return []
    end
    
    entity_linker = TestMockLinker()
    
    # Mock Embedding Client
    # Deterministic embeddings for consistent testing
    embedding_client = create_mock_embedding_client(768; deterministic=true)
    
    # Load Domain
    domain = load_biomedical_domain(umls_client, entity_linker, embedding_client)
    register_domain!("biomedical", domain)
    
    # 2. Setup Config
    config = SeedInjectionConfig(
        entity_linking_threshold=0.5,
        top_k_candidates=5,
        top_n_triples_per_entity=10,
        alpha_score_threshold=0.5,
        use_contextual_filtering=true,
        contextual_similarity_threshold=-1.0, # Disable filtering for test (random embeddings are orthogonal ~0.0)
        injection_ratio=0.99 # Must be < 1.0 according to assertions
    )
    
    # 3. Create Test Data
    # Use "Diabetes mellitus" to ensure regex extraction works (it has a specific pattern)
    sequences = [
        "Diabetes mellitus is a chronic disease.",
        "Metformin treats diabetes."
    ]
    
    # Create some mock seed triples in the "Seed KG"
    # These represent the universe of available triples
    seed_kg = [
        SemanticTriple("Diabetes Mellitus", "C0011849", "TREATS", "Metformin", [1,2,3], 0.9, "UMLS"),
        SemanticTriple("Diabetes Mellitus", "C0011849", "CAUSES", "Retinopathy", [4,5,6], 0.8, "UMLS"),
        SemanticTriple("Diabetes Mellitus", "C0011849", "ASSOCIATED_WITH", "Obesity", [7,8,9], 0.7, "UMLS"),
        SemanticTriple("Cancer", "C0006826", "TREATS", "Chemotherapy", [10,11], 0.9, "UMLS"), # Irrelevant
    ]
    
    # 4. Run Injection
    injected = inject_seed_kg(sequences, seed_kg, config, domain)
    
    @test length(injected) == 2
    
    # Check first sequence: "Diabetes mellitus is a chronic disease."
    seq1, triples1 = injected[1]
    @test seq1 == sequences[1]
    
    # Should contain triples related to Diabetes Mellitus (C0011849)
    # "Cancer" triples should be filtered out
    @test !isempty(triples1)
    for t in triples1
        @test t.head_cui == "C0011849"
    end
    
    # Check context filtering
    # Since we used mock embeddings, similarity is random but deterministic.
    # We set a low threshold (-1.0) so all should pass.
    # The scores should now be the similarity scores (Float64)
    if !isempty(triples1)
        @test triples1[1].score <= 1.0
        @test triples1[1].score >= -1.0
    end
    
    # 5. Verify Embeddings Integration
    # Explicitly test select_triples_for_injection with embeddings
    
    # Create a linking result
    linked = [
        EntityLinkingResult("diabetes", "C0011849", "Diabetes", ["T047"], 1.0, "Mock")
    ]
    
    # Compute embedding for context
    seq_embedding = embed(embedding_client, "Diabetes context")
    
    # Select
    selected = select_triples_for_injection(
        linked,
        seed_kg,
        config;
        sequence_embedding=seq_embedding,
        embedding_client=embedding_client
    )
    
    @test length(selected) > 0
    # Verify scores are updated to similarity
    for t in selected
        # Check that score is consistent with cosine similarity
        triple_text = "$(t.head) $(t.relation) $(t.tail)"
        triple_emb = embed(embedding_client, triple_text)
        sim = cosine_similarity(seq_embedding, triple_emb)
        @test isapprox(t.score, Float64(sim), atol=1e-5)
    end
    
    println("Stream D1 (Seed KG Injection) verification passed!")
end
