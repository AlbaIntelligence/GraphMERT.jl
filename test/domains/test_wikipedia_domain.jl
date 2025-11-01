"""
Test suite for Wikipedia domain extraction
===========================================

This test suite verifies that the Wikipedia domain provider correctly
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

# Load Wikipedia domain
include("../../GraphMERT/src/domains/wikipedia.jl")

@testset "Wikipedia Domain Tests" begin
    @testset "Domain Creation and Registration" begin
        # Create domain instance
        wiki_domain = load_wikipedia_domain()
        @test wiki_domain isa WikipediaDomain
        @test wiki_domain isa DomainProvider
        
        # Register domain
        register_domain!("wikipedia", wiki_domain)
        @test has_domain("wikipedia")
        
        # Test domain name
        @test get_domain_name(wiki_domain) == "wikipedia"
    end
    
    @testset "Entity Type Registration" begin
        wiki_domain = get_domain("wikipedia")
        entity_types = register_entity_types(wiki_domain)
        
        # Check expected Wikipedia entity types
        expected_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", 
                         "EVENT", "TECHNOLOGY", "ARTWORK", "PERIOD"]
        
        for entity_type in expected_types
            @test haskey(entity_types, entity_type)
        end
        
        # Verify entity type metadata
        person_meta = entity_types["PERSON"]
        @test haskey(person_meta, "domain")
        @test person_meta["domain"] == "wikipedia"
    end
    
    @testset "Relation Type Registration" begin
        wiki_domain = get_domain("wikipedia")
        relation_types = register_relation_types(wiki_domain)
        
        # Check expected Wikipedia relation types
        expected_types = ["CREATED_BY", "WORKED_AT", "BORN_IN", "DIED_IN",
                         "FOUNDED", "INVENTED", "DISCOVERED", "WROTE"]
        
        for relation_type in expected_types
            @test haskey(relation_types, relation_type)
        end
    end
    
    @testset "Entity Extraction" begin
        wiki_domain = get_domain("wikipedia")
        options = ProcessingOptions(domain="wikipedia")
        
        # Test extraction from Wikipedia-style text
        text1 = "Leonardo da Vinci was born in Vinci, Italy."
        entities1 = extract_entities(wiki_domain, text1, options)
        
        @test isa(entities1, Vector)
        @test length(entities1) > 0
        @test all(e -> e isa Entity, entities1)
        
        # Check domain field
        @test all(e -> e.domain == "wikipedia", entities1)
        
        # Test extraction from longer text
        text2 = """
        Albert Einstein developed the theory of relativity.
        He worked at Princeton University and was born in Germany.
        The Mona Lisa was painted by Leonardo da Vinci.
        """
        entities2 = extract_entities(wiki_domain, text2, options)
        
        @test length(entities2) > 0
        
        # Verify entity types are valid
        entity_types = register_entity_types(wiki_domain)
        for entity in entities2
            @test haskey(entity_types, entity.entity_type) || entity.entity_type == "UNKNOWN"
        end
    end
    
    @testset "Entity Validation" begin
        wiki_domain = get_domain("wikipedia")
        
        # Test valid entities
        @test validate_entity(wiki_domain, "Leonardo da Vinci", "PERSON", Dict())
        @test validate_entity(wiki_domain, "Italy", "LOCATION", Dict())
        @test validate_entity(wiki_domain, "Princeton University", "ORGANIZATION", Dict())
        
        # Test invalid entities
        @test !validate_entity(wiki_domain, "", "PERSON", Dict())
        @test !validate_entity(wiki_domain, "a", "PERSON", Dict())  # Too short
        
        # Test wrong type
        @test !validate_entity(wiki_domain, "random text", "PERSON", Dict())
    end
    
    @testset "Entity Confidence Calculation" begin
        wiki_domain = get_domain("wikipedia")
        
        # Test confidence calculation
        conf1 = calculate_entity_confidence(wiki_domain, "Leonardo da Vinci", "PERSON", Dict())
        @test 0.0 <= conf1 <= 1.0
        
        conf2 = calculate_entity_confidence(wiki_domain, "Italy", "LOCATION", Dict())
        @test 0.0 <= conf2 <= 1.0
        
        # Test invalid entity confidence
        conf3 = calculate_entity_confidence(wiki_domain, "", "PERSON", Dict())
        @test conf3 == 0.0 || conf3 < 0.5  # Should be low or zero
    end
    
    @testset "Relation Extraction" begin
        wiki_domain = get_domain("wikipedia")
        options = ProcessingOptions(domain="wikipedia")
        
        # Create sample entities
        text = "Leonardo da Vinci was born in Vinci, Italy."
        entities = extract_entities(wiki_domain, text, options)
        
        # Extract relations
        relations = extract_relations(wiki_domain, entities, text, options)
        
        @test isa(relations, Vector)
        @test all(r -> r isa Relation, relations)
        
        # Check domain field
        @test all(r -> r.domain == "wikipedia", relations)
        
        # Verify relation types are valid
        relation_types = register_relation_types(wiki_domain)
        for relation in relations
            @test haskey(relation_types, relation.relation_type)
        end
    end
    
    @testset "Relation Validation" begin
        wiki_domain = get_domain("wikipedia")
        
        # Test valid relations
        @test validate_relation(wiki_domain, "Leonardo da Vinci", "BORN_IN", "Italy", Dict())
        @test validate_relation(wiki_domain, "Einstein", "WORKED_AT", "Princeton University", Dict())
        
        # Test invalid relations
        @test !validate_relation(wiki_domain, "", "BORN_IN", "Italy", Dict())
        @test !validate_relation(wiki_domain, "Leonardo da Vinci", "", "Italy", Dict())
    end
    
    @testset "Relation Confidence Calculation" begin
        wiki_domain = get_domain("wikipedia")
        
        # Test confidence calculation
        conf1 = calculate_relation_confidence(wiki_domain, "Leonardo da Vinci", "BORN_IN", "Italy", Dict())
        @test 0.0 <= conf1 <= 1.0
        
        conf2 = calculate_relation_confidence(wiki_domain, "Einstein", "WORKED_AT", "Princeton", Dict())
        @test 0.0 <= conf2 <= 1.0
    end
    
    @testset "LLM Prompt Generation" begin
        wiki_domain = get_domain("wikipedia")
        
        # Test entity discovery prompt
        prompt1 = create_prompt(wiki_domain, :entity_discovery, Dict("text" => "Sample text"))
        @test isa(prompt1, String)
        @test length(prompt1) > 0
        
        # Test relation matching prompt
        prompt2 = create_prompt(wiki_domain, :relation_matching, Dict("entities" => ["entity1", "entity2"]))
        @test isa(prompt2, String)
        @test length(prompt2) > 0
        
        # Test tail formation prompt
        prompt3 = create_prompt(wiki_domain, :tail_formation, Dict("head" => "head", "relation" => "BORN_IN"))
        @test isa(prompt3, String)
        @test length(prompt3) > 0
    end
    
    @testset "Entity Linking (Wikidata)" begin
        wiki_domain = get_domain("wikipedia")
        
        # Test linking without Wikidata client (should return nothing)
        result = link_entity(wiki_domain, "Leonardo da Vinci", Dict())
        @test result === nothing || isa(result, Dict)
        
        # Note: Full Wikidata linking tests would require a Wikidata client instance
    end
    
    @testset "Seed Triple Creation" begin
        wiki_domain = get_domain("wikipedia")
        
        # Test seed triple creation without Wikidata client
        triples = create_seed_triples(wiki_domain, "Leonardo da Vinci", Dict())
        @test isa(triples, Vector)
        
        # Note: Full seed triple tests would require a Wikidata client instance
    end
    
    @testset "Edge Cases" begin
        wiki_domain = get_domain("wikipedia")
        options = ProcessingOptions(domain="wikipedia")
        
        # Test empty text
        entities_empty = extract_entities(wiki_domain, "", options)
        @test isa(entities_empty, Vector)
        
        # Test very short text
        entities_short = extract_entities(wiki_domain, "a", options)
        @test isa(entities_short, Vector)
        
        # Test text with no Wikipedia entities
        entities_none = extract_entities(wiki_domain, "This is a test sentence with no proper nouns.", options)
        @test isa(entities_none, Vector)
    end
end

println("✅ All Wikipedia domain tests passed!")
