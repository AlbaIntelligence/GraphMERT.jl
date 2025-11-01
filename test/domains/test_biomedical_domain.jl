"""
Test suite for biomedical domain extraction
============================================

This test suite verifies that the biomedical domain provider correctly
extracts entities, relations, validates them, and calculates confidence scores.
"""

using Test
using Pkg
Pkg.activate("../../")

using GraphMERT
using Logging

# Configure logging for tests
# Use global logger to suppress verbose output
global_logger(Logging.ConsoleLogger(stderr, Logging.Warn))

# Load biomedical domain
include("../../GraphMERT/src/domains/biomedical.jl")

@testset "Biomedical Domain Tests" begin
    @testset "Domain Creation and Registration" begin
        # Create domain instance
        bio_domain = load_biomedical_domain()
        @test bio_domain isa BiomedicalDomain
        @test bio_domain isa DomainProvider
        
        # Register domain
        register_domain!("biomedical", bio_domain)
        @test has_domain("biomedical")
        
        # Test domain name
        @test get_domain_name(bio_domain) == "biomedical"
    end
    
    @testset "Entity Type Registration" begin
        bio_domain = get_domain("biomedical")
        entity_types = register_entity_types(bio_domain)
        
        # Check expected biomedical entity types
        expected_types = ["DISEASE", "DRUG", "PROTEIN", "GENE", "ANATOMY", 
                         "SYMPTOM", "PROCEDURE", "ORGANISM", "CHEMICAL"]
        
        for entity_type in expected_types
            @test haskey(entity_types, entity_type)
        end
        
        # Verify entity type metadata
        disease_meta = entity_types["DISEASE"]
        @test haskey(disease_meta, "domain")
        @test disease_meta["domain"] == "biomedical"
    end
    
    @testset "Relation Type Registration" begin
        bio_domain = get_domain("biomedical")
        relation_types = register_relation_types(bio_domain)
        
        # Check expected biomedical relation types
        expected_types = ["TREATS", "CAUSES", "ASSOCIATED_WITH", "PREVENTS",
                         "INHIBITS", "ACTIVATES", "BINDS_TO"]
        
        for relation_type in expected_types
            @test haskey(relation_types, relation_type)
        end
    end
    
    @testset "Entity Extraction" begin
        bio_domain = get_domain("biomedical")
        options = ProcessingOptions(domain="biomedical")
        
        # Test extraction from biomedical text
        text1 = "Diabetes is treated with metformin."
        entities1 = extract_entities(bio_domain, text1, options)
        
        @test isa(entities1, Vector)
        @test length(entities1) > 0
        @test all(e -> e isa Entity, entities1)
        
        # Check domain field
        @test all(e -> e.domain == "biomedical", entities1)
        
        # Test extraction from longer text
        text2 = """
        Alzheimer's disease is a neurodegenerative disorder. 
        Aspirin is a common drug used to treat pain. 
        Insulin is a protein hormone produced by the pancreas.
        """
        entities2 = extract_entities(bio_domain, text2, options)
        
        @test length(entities2) > 0
        
        # Verify entity types are valid
        entity_types = register_entity_types(bio_domain)
        for entity in entities2
            @test haskey(entity_types, entity.entity_type) || entity.entity_type == "UNKNOWN"
        end
    end
    
    @testset "Entity Validation" begin
        bio_domain = get_domain("biomedical")
        
        # Test valid entities
        @test validate_entity(bio_domain, "diabetes", "DISEASE", Dict{String, Any}())
        @test validate_entity(bio_domain, "aspirin", "DRUG", Dict{String, Any}())
        @test validate_entity(bio_domain, "insulin", "PROTEIN", Dict{String, Any}())
        
        # Test invalid entities
        @test !validate_entity(bio_domain, "", "DISEASE", Dict{String, Any}())
        @test !validate_entity(bio_domain, "a", "DISEASE", Dict{String, Any}())  # Too short
        
        # Test wrong type
        @test !validate_entity(bio_domain, "random text", "DISEASE", Dict{String, Any}())
    end
    
    @testset "Entity Confidence Calculation" begin
        bio_domain = get_domain("biomedical")
        
        # Test confidence calculation
        conf1 = calculate_entity_confidence(bio_domain, "diabetes", "DISEASE", Dict{String, Any}())
        @test 0.0 <= conf1 <= 1.0
        
        conf2 = calculate_entity_confidence(bio_domain, "aspirin", "DRUG", Dict{String, Any}())
        @test 0.0 <= conf2 <= 1.0
        
        # Test invalid entity confidence
        conf3 = calculate_entity_confidence(bio_domain, "", "DISEASE", Dict{String, Any}())
        @test conf3 == 0.0 || conf3 < 0.5  # Should be low or zero
    end
    
    @testset "Relation Extraction" begin
        bio_domain = get_domain("biomedical")
        options = ProcessingOptions(domain="biomedical")
        
        # Create sample entities
        text = "Diabetes is treated with metformin."
        entities = extract_entities(bio_domain, text, options)
        
        # Extract relations
        relations = extract_relations(bio_domain, entities, text, options)
        
        @test isa(relations, Vector)
        @test all(r -> r isa Relation, relations)
        
        # Check domain field
        @test all(r -> r.domain == "biomedical", relations)
        
        # Verify relation types are valid
        relation_types = register_relation_types(bio_domain)
        for relation in relations
            @test haskey(relation_types, relation.relation_type)
        end
    end
    
    @testset "Relation Validation" begin
        bio_domain = get_domain("biomedical")
        
        # Test valid relations
        @test validate_relation(bio_domain, "metformin", "TREATS", "diabetes", Dict{String, Any}())
        @test validate_relation(bio_domain, "smoking", "CAUSES", "lung cancer", Dict{String, Any}())
        
        # Test invalid relations
        @test !validate_relation(bio_domain, "", "TREATS", "diabetes", Dict{String, Any}())
        @test !validate_relation(bio_domain, "metformin", "", "diabetes", Dict{String, Any}())
    end
    
    @testset "Relation Confidence Calculation" begin
        bio_domain = get_domain("biomedical")
        
        # Test confidence calculation
        conf1 = calculate_relation_confidence(bio_domain, "metformin", "TREATS", "diabetes", Dict{String, Any}())
        @test 0.0 <= conf1 <= 1.0
        
        conf2 = calculate_relation_confidence(bio_domain, "smoking", "CAUSES", "lung cancer", Dict{String, Any}())
        @test 0.0 <= conf2 <= 1.0
    end
    
    @testset "LLM Prompt Generation" begin
        bio_domain = get_domain("biomedical")
        
        # Test entity discovery prompt
        prompt1 = create_prompt(bio_domain, :entity_discovery, Dict("text" => "Sample text"))
        @test isa(prompt1, String)
        @test length(prompt1) > 0
        
        # Test relation matching prompt
        prompt2 = create_prompt(bio_domain, :relation_matching, Dict("entities" => ["entity1", "entity2"]))
        @test isa(prompt2, String)
        @test length(prompt2) > 0
        
        # Test tail formation prompt
        prompt3 = create_prompt(bio_domain, :tail_formation, Dict("head" => "head", "relation" => "TREATS"))
        @test isa(prompt3, String)
        @test length(prompt3) > 0
    end
    
    @testset "Entity Linking (UMLS)" begin
        bio_domain = get_domain("biomedical")
        
        # Test linking without UMLS client (should return nothing)
        result = link_entity(bio_domain, "diabetes", Dict{String, Any}())
        @test result === nothing || isa(result, Dict)
        
        # Note: Full UMLS linking tests would require a UMLS client instance
    end
    
    @testset "Seed Triple Creation" begin
        bio_domain = get_domain("biomedical")
        
        # Test seed triple creation without UMLS client
        triples = create_seed_triples(bio_domain, "diabetes", Dict{String, Any}())
        @test isa(triples, Vector)
        
        # Note: Full seed triple tests would require a UMLS client instance
    end
    
    @testset "Edge Cases" begin
        bio_domain = get_domain("biomedical")
        options = ProcessingOptions(domain="biomedical")
        
        # Test empty text
        entities_empty = extract_entities(bio_domain, "", options)
        @test isa(entities_empty, Vector)
        
        # Test very short text
        entities_short = extract_entities(bio_domain, "a", options)
        @test isa(entities_short, Vector)
        
        # Test text with no biomedical entities
        entities_none = extract_entities(bio_domain, "This is a test sentence with no biomedical terms.", options)
        @test isa(entities_none, Vector)
    end
end

println("✅ All biomedical domain tests passed!")
