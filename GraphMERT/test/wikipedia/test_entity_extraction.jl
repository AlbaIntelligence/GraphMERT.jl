"""
Unit tests for Wikipedia domain entity extraction.

Tests the Wikipedia domain's ability to extract entities from French monarchy articles.
"""

using Test
using GraphMERT

# Include test fixtures
include("fixtures.jl")

@testset "Wikipedia Entity Extraction" begin
    # Setup
    setup_wikipedia_tests()
    
    @testset "Louis XIV Entity Extraction" begin
        # Test entity extraction from Louis XIV article
        text = LOUIS_XIV_ARTICLE
        options = create_test_options()
        
        # Try to extract entities
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            entities = Base.invokelatest(GraphMERT.extract_entities, domain, text, options)
            
            # Basic validation
            @test length(entities) > 0
            
            # Check for key entities
            entity_texts = [e.text for e in entities]
            @test "Louis XIV" in entity_texts
            
            # Check entity types are assigned
            for ent in entities
                @test !isempty(ent.entity_type)
            end
            
            println("Extracted $(length(entities)) entities from Louis XIV article")
            println("Entity types: $(unique([e.entity_type for e in entities]))")
        else
            @test_skip true
        end
    end
    
    @testset "Henry IV Entity Extraction" begin
        text = HENRY_IV_ARTICLE
        options = create_test_options()
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            entities = Base.invokelatest(GraphMERT.extract_entities, domain, text, options)
            
            @test length(entities) > 0
            
            entity_texts = [e.text for e in entities]
            @test "Henry IV" in entity_texts
            
            println("Extracted $(length(entities)) entities from Henry IV article")
        else
            @test_skip true
        end
    end
    
    @testset "Marie Antoinette Entity Extraction" begin
        text = MARIE_ANTOINETTE_ARTICLE
        options = create_test_options()
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            entities = Base.invokelatest(GraphMERT.extract_entities, domain, text, options)
            
            @test length(entities) > 0
            
            entity_texts = [e.text for e in entities]
            @test "Marie Antoinette" in entity_texts
            
            @test "Louis XVI" in entity_texts
            println("Extracted $(length(entities)) entities from Marie Antoinette article")
        else
            @test_skip true
        end
    end
    
    @testset "Confidence Scores" begin
        text = LOUIS_XIV_ARTICLE
        options = create_test_options()
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            entities = Base.invokelatest(GraphMERT.extract_entities, domain, text, options)
            
            # Check confidence scores are in valid range
            for ent in entities
                @test 0.0 <= ent.confidence <= 1.0
            end
            
            # Check filtering works
            high_conf = filter(e -> e.confidence >= 0.8, entities)
            println("High confidence entities (>= 0.8): $(length(high_conf))")
        else
            @test_skip true
        end
    end
    
    @testset "Entity Type Classification" begin
        text = LOUIS_XIV_ARTICLE * " " * HENRY_IV_ARTICLE
        options = create_test_options()
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            entities = Base.invokelatest(GraphMERT.extract_entities, domain, text, options)
            
            # Check for different entity types
            types = unique([e.entity_type for e in entities])
            
            # Should have person entities
            has_person = any(e -> e.entity_type == "PERSON", entities)
            @test has_person
            
            println("Found entity types: $types")
        else
            @test_skip true
        end
    end
end

println("Entity extraction tests completed!")
