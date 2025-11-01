"""
Integration test for domain system
===================================

This test verifies that the full extraction pipeline works correctly
with both biomedical and Wikipedia domains, including domain switching.
"""

using Test
using Pkg
Pkg.activate("../../")

using GraphMERT
using Logging

# Configure logging for tests
# Use global logger to suppress verbose output
global_logger(Logging.ConsoleLogger(stderr, Logging.Warn))

# Load domains
include("../../GraphMERT/src/domains/biomedical.jl")
include("../../GraphMERT/src/domains/wikipedia.jl")

@testset "Domain Integration Tests" begin
    @testset "Domain Loading and Registration" begin
        # Load both domains
        bio_domain = load_biomedical_domain()
        wiki_domain = load_wikipedia_domain()
        
        # Register both domains
        register_domain!("biomedical", bio_domain)
        register_domain!("wikipedia", wiki_domain)
        
        # Verify both are registered
        @test has_domain("biomedical")
        @test has_domain("wikipedia")
        @test length(list_domains()) >= 2
        
        # Verify both can be retrieved
        @test get_domain("biomedical") !== nothing
        @test get_domain("wikipedia") !== nothing
    end
    
    @testset "Biomedical Domain Extraction" begin
        bio_domain = get_domain("biomedical")
        options = ProcessingOptions(domain="biomedical")
        
        # Test biomedical text
        text = "Diabetes is treated with metformin. Aspirin is used for pain relief."
        
        # Extract entities
        entities = extract_entities(bio_domain, text, options)
        @test length(entities) > 0
        @test all(e -> e.domain == "biomedical", entities)
        
        # Extract relations
        relations = extract_relations(bio_domain, entities, text, options)
        @test isa(relations, Vector)
        @test all(r -> r.domain == "biomedical", relations)
        
        # Verify entity types are biomedical
        bio_entity_types = register_entity_types(bio_domain)
        for entity in entities
            @test haskey(bio_entity_types, entity.entity_type) || entity.entity_type == "UNKNOWN"
        end
    end
    
    @testset "Wikipedia Domain Extraction" begin
        wiki_domain = get_domain("wikipedia")
        options = ProcessingOptions(domain="wikipedia")
        
        # Test Wikipedia-style text
        text = "Leonardo da Vinci was born in Vinci, Italy. He painted the Mona Lisa."
        
        # Extract entities
        entities = extract_entities(wiki_domain, text, options)
        @test length(entities) > 0
        @test all(e -> e.domain == "wikipedia", entities)
        
        # Extract relations
        relations = extract_relations(wiki_domain, entities, text, options)
        @test isa(relations, Vector)
        @test all(r -> r.domain == "wikipedia", relations)
        
        # Verify entity types are Wikipedia
        wiki_entity_types = register_entity_types(wiki_domain)
        for entity in entities
            @test haskey(wiki_entity_types, entity.entity_type) || entity.entity_type == "UNKNOWN"
        end
    end
    
    @testset "Domain Switching" begin
        # Test switching between domains
        bio_domain = get_domain("biomedical")
        wiki_domain = get_domain("wikipedia")
        
        bio_options = ProcessingOptions(domain="biomedical")
        wiki_options = ProcessingOptions(domain="wikipedia")
        
        # Extract with biomedical domain
        bio_text = "Diabetes is treated with metformin."
        bio_entities = extract_entities(bio_domain, bio_text, bio_options)
        
        # Extract with Wikipedia domain (should find different entities)
        wiki_text = "Leonardo da Vinci was born in Italy."
        wiki_entities = extract_entities(wiki_domain, wiki_text, wiki_options)
        
        # Verify domains are different
        @test all(e -> e.domain == "biomedical", bio_entities)
        @test all(e -> e.domain == "wikipedia", wiki_entities)
        
        # Verify entity types are domain-appropriate
        bio_entity_types = register_entity_types(bio_domain)
        wiki_entity_types = register_entity_types(wiki_domain)
        
        # Biomedical entities should have biomedical types
        for entity in bio_entities
            if entity.entity_type != "UNKNOWN"
                @test haskey(bio_entity_types, entity.entity_type)
            end
        end
        
        # Wikipedia entities should have Wikipedia types
        for entity in wiki_entities
            if entity.entity_type != "UNKNOWN"
                @test haskey(wiki_entity_types, entity.entity_type)
            end
        end
    end
    
    @testset "Cross-Domain Validation" begin
        # Test that domains correctly validate only their own entity/relation types
        bio_domain = get_domain("biomedical")
        wiki_domain = get_domain("wikipedia")
        
        # Biomedical domain should validate biomedical entities
        @test validate_entity(bio_domain, "diabetes", "DISEASE", Dict())
        @test !validate_entity(bio_domain, "Leonardo da Vinci", "PERSON", Dict())
        
        # Wikipedia domain should validate Wikipedia entities
        @test validate_entity(wiki_domain, "Leonardo da Vinci", "PERSON", Dict())
        @test !validate_entity(wiki_domain, "diabetes", "DISEASE", Dict())
        
        # Test relation validation
        @test validate_relation(bio_domain, "metformin", "TREATS", "diabetes", Dict())
        @test !validate_relation(bio_domain, "Leonardo da Vinci", "BORN_IN", "Italy", Dict())
        
        @test validate_relation(wiki_domain, "Leonardo da Vinci", "BORN_IN", "Italy", Dict())
        @test !validate_relation(wiki_domain, "metformin", "TREATS", "diabetes", Dict())
    end
    
    @testset "Default Domain Behavior" begin
        # Test default domain functionality
        default_domain = get_default_domain()
        @test default_domain !== nothing
        
        # Test that operations work with default domain
        options = ProcessingOptions(domain="biomedical")  # Explicitly set
        text = "Diabetes is treated with metformin."
        entities = extract_entities(default_domain, text, options)
        @test length(entities) > 0
    end
    
    @testset "Domain-Specific Prompts" begin
        # Test that prompts are domain-specific
        bio_domain = get_domain("biomedical")
        wiki_domain = get_domain("wikipedia")
        
        bio_prompt = create_prompt(bio_domain, :entity_discovery, Dict("text" => "sample"))
        wiki_prompt = create_prompt(wiki_domain, :entity_discovery, Dict("text" => "sample"))
        
        # Prompts should be different (domain-specific)
        @test bio_prompt != wiki_prompt
        
        # Both should be valid strings
        @test isa(bio_prompt, String)
        @test isa(wiki_prompt, String)
        @test length(bio_prompt) > 0
        @test length(wiki_prompt) > 0
    end
    
    @testset "Backward Compatibility" begin
        # Test that default domain is set (biomedical for backward compatibility)
        default_domain = get_default_domain()
        @test default_domain !== nothing
        
        # Test that ProcessingOptions defaults to biomedical domain
        # (This is tested in config.jl, but verify here)
        options = default_processing_options()
        @test options.domain == "biomedical"
    end
end

println("✅ All integration tests passed!")
