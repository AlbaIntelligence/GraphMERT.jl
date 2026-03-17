using Test
using GraphMERT

# Access internal function if not exported
# Depending on module structure, it might be GraphMERT.calculate_character_3gram_jaccard
# or GraphMERT.BiomedicalDomain.calculate_character_3gram_jaccard if namespaced.
# Based on file structure, it seems likely to be directly in GraphMERT if included there.
# Let's try to access it from Main.GraphMERT.

@testset "Character 3-gram Jaccard Similarity" begin
    # Test cases
    # 1. Exact match
    @test GraphMERT.calculate_character_3gram_jaccard("diabetes", "diabetes") ≈ 1.0
    
    # 2. Case insensitivity
    @test GraphMERT.calculate_character_3gram_jaccard("Diabetes", "diabetes") ≈ 1.0
    
    # 3. Partial overlap
    # "diabetes" -> dia, iab, abe, bet, ete, tes (6 grams)
    # "diabetic" -> dia, iab, abe, bet, eti, tic (6 grams)
    # Intersection: dia, iab, abe, bet (4 grams)
    # Union: dia, iab, abe, bet, ete, tes, eti, tic (8 grams)
    # Jaccard: 4/8 = 0.5
    score = GraphMERT.calculate_character_3gram_jaccard("diabetes", "diabetic")
    @test score ≈ 0.5
    
    # 4. No overlap
    @test GraphMERT.calculate_character_3gram_jaccard("diabetes", "cancer") == 0.0
    
    # 5. Short strings fallback
    @test GraphMERT.calculate_character_3gram_jaccard("ab", "ab") == 1.0
    @test GraphMERT.calculate_character_3gram_jaccard("ab", "ac") == 0.0
end

@testset "SapBERT Linker Reranking" begin
    # Create linker with mock behavior enabled by default implementation
    # Set reranking weight to 0.5 for easy calculation
    linker = SapBERTLinker("mock_model", "mock_index", reranking_weight=0.5, top_n_candidates=10)
    
    # Query: "diabetes"
    # Mock candidates will be:
    # 1. "diabetes" (Exact match) -> BERT=0.85, Jaccard=1.0 -> Final = 0.5*0.85 + 0.5*1.0 = 0.925
    # 2. "diabetes type 2" -> BERT=0.75, Jaccard<1.0 -> Final < 0.925
    # 3. "metabolic syndrome" -> BERT=0.70, Jaccard=0.0 -> Final = 0.35
    
    results = GraphMERT.link_entity(linker, "diabetes", top_k=3)
    
    @test length(results) == 3
    
    # Check top result
    (cui, score, name) = results[1]
    @test name == "diabetes"
    @test score ≈ 0.925
    
    # Check ranking order
    @test results[1][2] > results[2][2]
    @test results[2][2] > results[3][2]
    
    # Test with different weights
    # Weight=0.0 (Only BERT)
    linker_bert = SapBERTLinker("mock", "mock", reranking_weight=0.0)
    results_bert = GraphMERT.link_entity(linker_bert, "diabetes", top_k=3)
    @test results_bert[1][2] ≈ 0.85
    
    # Weight=1.0 (Only Jaccard)
    linker_jaccard = SapBERTLinker("mock", "mock", reranking_weight=1.0)
    results_jaccard = GraphMERT.link_entity(linker_jaccard, "diabetes", top_k=3)
    @test results_jaccard[1][2] ≈ 1.0
end

@testset "SapBERT In-Memory Index Search" begin
    # 1. Create dummy linker to access encode_text
    dummy_linker = SapBERTLinker("mock", "mock")
    
    # 2. Generate embeddings for known concepts
    # Since encode_text is deterministic based on text hash
    vec_diabetes = GraphMERT.encode_text(dummy_linker, "diabetes")
    vec_cancer = GraphMERT.encode_text(dummy_linker, "cancer")
    vec_metformin = GraphMERT.encode_text(dummy_linker, "metformin")
    
    # 3. Build index
    # Stack vectors into matrix (dim x N)
    embeddings = hcat(vec_diabetes, vec_cancer, vec_metformin)
    cui_list = ["C0011849", "C0006826", "C0025598"]
    name_list = ["diabetes", "cancer", "metformin"]
    
    # 4. Create linker with index
    linker = SapBERTLinker("mock", "mock", 
        embeddings=embeddings, 
        cui_list=cui_list, 
        name_list=name_list,
        reranking_weight=0.0 # Disable Jaccard for this test to check BERT retrieval pure
    )
    
    # 5. Test exact retrieval
    # "diabetes" should match index 1 (similarity 1.0)
    results = GraphMERT.link_entity(linker, "diabetes", top_k=1)
    @test length(results) == 1
    @test results[1][1] == "C0011849"
    @test isapprox(results[1][2], 1.0, atol=1e-5)
    
    # 6. Test retrieval of another concept
    results = GraphMERT.link_entity(linker, "cancer", top_k=1)
    @test results[1][1] == "C0006826"
    @test isapprox(results[1][2], 1.0, atol=1e-5)
    
    # 7. Test integration with Jaccard
    # Create linker with Jaccard enabled
    linker_hybrid = SapBERTLinker("mock", "mock", 
        embeddings=embeddings, 
        cui_list=cui_list, 
        name_list=name_list,
        reranking_weight=0.5
    )
    
    # Search for "diabetis" (typo)
    # BERT might give high score to "diabetes" (if vectors are close, but here they are random so maybe not)
    # Jaccard will give high score to "diabetes"
    
    # Note: Random vectors are orthogonal in high dimensions, so "diabetis" vector will be far from "diabetes" vector
    # So BERT score will be low (~0).
    # Jaccard score will be high (~0.8).
    # Final score should favor "diabetes".
    
    results = GraphMERT.link_entity(linker_hybrid, "diabetis", top_k=3)
    # We expect "diabetes" to be in top results due to Jaccard
    # Find "diabetes" in results
    found = false
    for (cui, score, name) in results
        if name == "diabetes"
            found = true
            @test score > 0.3 # At least some score from Jaccard (0.5 * ~0.8 = 0.4)
        end
    end
    @test found
end
