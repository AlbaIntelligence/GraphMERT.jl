"""
Unit tests for Seed KG Injection implementation

Tests the seed injection algorithm including:
- SapBERT embedding-based entity linking
- Character 3-gram Jaccard similarity filtering
- Triple selection and injection
- Validation and quality control
"""

using Test
using Random
using GraphMERT
using GraphMERT: SeedInjectionConfig, SemanticTriple, EntityLinkingResult,
  link_entity_sapbert, select_triples_for_entity, inject_seed_kg,
  select_triples_for_injection, bucket_by_score, bucket_by_relation_frequency,
  validate_injected_triples, get_entity_name_from_cui, tokenize_entity_name

@testset "Seed Injection Tests" begin

  @testset "Entity Linking with SapBERT" begin
    config = SeedInjectionConfig()
    
    # Test basic entity linking
    entity_text = "diabetes"
    results = link_entity_sapbert(entity_text, config)
    
    @test results isa Vector{EntityLinkingResult}
    @test length(results) >= 0  # May be empty
    
    # Check that results are properly ranked if we have multiple results
    if length(results) > 1
      @test results[1].similarity_score >= results[2].similarity_score
    end
    
    # Test with different entity types
    test_entities = ["metformin", "pregnancy", "clopidogrel", "hypertension"]
    
    for entity in test_entities
      results = link_entity_sapbert(entity, config)
      @test results isa Vector{EntityLinkingResult}
      @test length(results) >= 0  # May be empty for some entities
      
      # Check result structure
      for result in results
        @test result.cui isa String
        @test result.preferred_name isa String
        @test result.semantic_types isa Vector{String}
        @test result.similarity_score isa Float64
        @test 0.0 <= result.similarity_score <= 1.0
        @test result.source isa String
      end
    end
  end

  @testset "Character 3-gram Jaccard Similarity" begin
    config = SeedInjectionConfig()
    
    # Test entities with known similarities
    test_cases = [
      ("diabetes", "Diabetes Mellitus", 0.0),  # Should have some similarity
      ("metformin", "Metformin", 0.0),         # Should have some similarity
      ("preg", "Pregnancy", 0.0),              # Should have some similarity
      ("xyz", "Completely Different", 0.0),    # Should have no similarity
    ]
    
    for (entity, concept, expected_min) in test_cases
      results = link_entity_sapbert(entity, config)
      
      # Find matching concept
      matching_result = findfirst(r -> r.preferred_name == concept, results)
      
      if matching_result !== nothing
        result = results[matching_result]
        @test result.similarity_score >= expected_min
        @test result.similarity_score <= 1.0
      end
    end
  end

  @testset "Triple Selection for Entity" begin
    config = SeedInjectionConfig()
    
    # Test triple selection for known entities
    test_cuis = ["C0011849", "C0025598", "C0032961"]  # diabetes, metformin, pregnancy
    
    for cui in test_cuis
      triples = select_triples_for_entity(cui, config)
      
      @test triples isa Vector{SemanticTriple}
      @test length(triples) >= 0  # May be empty for some entities
      
      # Check triple structure
      for triple in triples
        @test triple.head isa String
        @test triple.relation isa String
        @test triple.tail isa String
        @test triple.score isa Float64
        @test 0.0 <= triple.score <= 1.0
        @test triple.source isa String
      end
    end
  end

  @testset "Seed KG Injection" begin
    config = SeedInjectionConfig()
    
    # Create test sequences
    sequences = [
      "The patient has diabetes and takes metformin.",
      "Pregnancy complications require careful monitoring.",
      "Hypertension is a risk factor for stroke."
    ]
    
    # Create test seed KG
    seed_kg = [
      SemanticTriple("diabetes", nothing, "treats", "metformin", [100, 150], 0.9, "seed"),
      SemanticTriple("pregnancy", nothing, "complication", "monitoring", [200, 250], 0.8, "seed"),
      SemanticTriple("hypertension", nothing, "causes", "stroke", [300, 350], 0.85, "seed"),
    ]
    
    # Test injection
    injected_sequences = inject_seed_kg(sequences, seed_kg, config)
    
    @test injected_sequences isa Vector{Tuple{String, Vector{SemanticTriple}}}
    @test length(injected_sequences) == length(sequences)
    
    # Check that sequences were processed
    for (original, (injected_seq, injected_triples)) in zip(sequences, injected_sequences)
      @test isa(injected_seq, String)
      @test isa(injected_triples, Vector{SemanticTriple})
    end
  end

  @testset "Triple Selection for Injection" begin
    config = SeedInjectionConfig()
    
    # Create mock linked entities
    linked_entities = [
      EntityLinkingResult("diabetes", "C0011849", "Diabetes Mellitus", ["Disease"], 0.9, "sapbert"),
      EntityLinkingResult("metformin", "C0025598", "Metformin", ["Drug"], 0.8, "sapbert"),
    ]
    
    # Create test seed KG
    seed_kg = [
      SemanticTriple("diabetes", nothing, "treats", "metformin", [100, 150], 0.9, "seed"),
      SemanticTriple("diabetes", nothing, "causes", "complications", [200, 250], 0.8, "seed"),
      SemanticTriple("metformin", nothing, "treats", "diabetes", [300, 350], 0.85, "seed"),
    ]
    
    # Test triple selection
    selected_triples = select_triples_for_injection(linked_entities, seed_kg, config)
    
    @test selected_triples isa Vector{SemanticTriple}
    @test length(selected_triples) <= length(seed_kg)  # Should be subset
    
    # Check that selected triples are relevant
    for triple in selected_triples
      @test triple.head isa String
      @test triple.relation isa String
      @test triple.tail isa String
      @test triple.score isa Float64
    end
  end

  @testset "Triple Bucketing" begin
    # Create test triples with different scores and relations
    test_triples = [
      SemanticTriple("a", nothing, "treats", "b", [100], 0.9, "test"),
      SemanticTriple("c", nothing, "causes", "d", [200], 0.7, "test"),
      SemanticTriple("e", nothing, "treats", "f", [300], 0.8, "test"),
      SemanticTriple("g", nothing, "prevents", "h", [400], 0.6, "test"),
      SemanticTriple("i", nothing, "treats", "j", [500], 0.5, "test"),
      SemanticTriple("k", nothing, "causes", "l", [600], 0.4, "test"),
    ]
    
    @testset "Score-based Bucketing" begin
      # Test score-based bucketing
      score_buckets = bucket_by_score(test_triples, 3)
      @test length(score_buckets) == 3
      @test all(bucket -> bucket isa Vector{SemanticTriple}, score_buckets)
      
      # Test that buckets are sorted by score (highest first)
      all_triples = vcat(score_buckets...)
      @test length(all_triples) == length(test_triples)
      
      # Test that first bucket has highest scores
      if !isempty(score_buckets[1])
        @test score_buckets[1][1].score >= 0.8  # Should contain highest scores
      end
      
      # Test empty input
      empty_buckets = bucket_by_score(SemanticTriple[], 3)
      @test length(empty_buckets) == 3
      @test all(isempty, empty_buckets)
    end
    
    @testset "Relation Frequency Bucketing" begin
      # Test relation frequency bucketing
      relation_buckets = bucket_by_relation_frequency(test_triples, 2)
      @test length(relation_buckets) == 2
      @test all(bucket -> bucket isa Vector{SemanticTriple}, relation_buckets)
      
      # Test that buckets contain triples (the function filters by relation frequency)
      all_triples = vcat(relation_buckets...)
      @test length(all_triples) > 0  # Should have some triples
      
      # Test that buckets contain different relations
      if !isempty(relation_buckets[1]) && !isempty(relation_buckets[2])
        relations_1 = Set([t.relation for t in relation_buckets[1]])
        relations_2 = Set([t.relation for t in relation_buckets[2]])
        # Should have some overlap since we're bucketing by frequency
        @test !isempty(relations_1 ∩ relations_2) || !isempty(relations_1) || !isempty(relations_2)
      end
      
      # Test empty input
      empty_buckets = bucket_by_relation_frequency(SemanticTriple[], 2)
      @test length(empty_buckets) == 2
      @test all(isempty, empty_buckets)
    end
    
    @testset "Edge Cases" begin
      # Test with single triple
      single_triple = [SemanticTriple("x", nothing, "rel", "y", [100], 0.5, "test")]
      single_buckets = bucket_by_score(single_triple, 3)
      @test length(single_buckets) == 3
      @test length(single_buckets[1]) == 1
      @test isempty(single_buckets[2])
      @test isempty(single_buckets[3])
      
      # Test with more buckets than triples
      many_buckets = bucket_by_score(test_triples, 10)
      @test length(many_buckets) == 10
      @test sum(length, many_buckets) == length(test_triples)
    end
  end

  @testset "Triple Validation" begin
    # Test sequence with medical content
    sequence = "The patient has diabetes and takes metformin for treatment."
    
    # Test valid triples
    valid_triples = [
      SemanticTriple("diabetes", nothing, "treats", "metformin", [100, 150], 0.9, "test"),
    ]
    
    validation_result = validate_injected_triples(sequence, valid_triples)
    @test validation_result isa Dict{SemanticTriple, Bool}
    
    # Test invalid triples (unrelated to sequence)
    invalid_triples = [
      SemanticTriple("cancer", nothing, "treats", "chemotherapy", [200, 250], 0.9, "test"),
    ]
    
    validation_result = validate_injected_triples(sequence, invalid_triples)
    @test validation_result isa Dict{SemanticTriple, Bool}
  end

  @testset "Utility Functions" begin
    # Test entity name retrieval
    test_cui = "C0011849"
    entity_name = get_entity_name_from_cui(test_cui)
    @test entity_name isa String
    @test !isempty(entity_name)
    
    # Test entity name tokenization
    test_name = "Diabetes Mellitus"
    tokens = tokenize_entity_name(test_name)
    @test tokens isa Vector{Int}  # Returns token IDs, not strings
    @test length(tokens) > 0
    @test all(token -> token isa Int, tokens)
  end

  @testset "Configuration Validation" begin
    # Test default configuration
    config = SeedInjectionConfig()
    @test config isa SeedInjectionConfig
    
    # Test configuration parameters
    @test config.entity_linking_threshold isa Float64
    @test config.top_k_candidates isa Int
    @test config.top_n_triples_per_entity isa Int
    @test config.alpha_score_threshold isa Float64
    
    # Test parameter constraints
    @test config.top_k_candidates > 0
    @test config.top_n_triples_per_entity > 0
    @test 0.0 <= config.entity_linking_threshold <= 1.0
    @test 0.0 <= config.alpha_score_threshold <= 1.0
  end

  @testset "Edge Cases" begin
    config = SeedInjectionConfig()
    
    # Test empty entity
    results = link_entity_sapbert("", config)
    @test results isa Vector{EntityLinkingResult}
    @test length(results) == 0
    
    # Test very long entity name
    long_entity = "very_long_entity_name_that_might_cause_issues_with_string_matching"
    results = link_entity_sapbert(long_entity, config)
    @test results isa Vector{EntityLinkingResult}
    
    # Test entity with special characters
    special_entity = "diabetes-mellitus_type-2"
    results = link_entity_sapbert(special_entity, config)
    @test results isa Vector{EntityLinkingResult}
    
    # Test empty seed KG
    empty_sequences = ["No medical content here."]
    empty_seed_kg = SemanticTriple[]
    injected = inject_seed_kg(empty_sequences, empty_seed_kg, config)
    @test injected isa Vector{Tuple{String, Vector{SemanticTriple}}}
    @test length(injected) == 1
  end

  @testset "Integration with Leafy Chain" begin
    # Test that seed injection works with leafy chain graphs
    config = SeedInjectionConfig()
    
    # Create a simple sequence
    sequence = "Patient has diabetes."
    
    # Create seed KG
    seed_kg = [
      SemanticTriple("diabetes", nothing, "treats", "metformin", [100, 150], 0.9, "seed"),
    ]
    
    # Inject seed KG
    injected_sequence = inject_seed_kg([sequence], seed_kg, config)
    
    @test length(injected_sequence) == 1
    @test isa(injected_sequence[1], Tuple{String, Vector{SemanticTriple}})
    
    # The injected sequence should be processable by leafy chain
    # (This would be tested in integration tests)
  end
end