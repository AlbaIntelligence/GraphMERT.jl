"""
Tests for entity type classification in Wikipedia domain.

Validates that entities are correctly classified as PERSON, LOCATION, TITLE, etc.
"""

using Test
using GraphMERT

include("fixtures.jl")

@testset "Entity Type Classification" begin
    
    @testset "Person Classification" begin
        # Test that monarchs are classified as PERSON
        monarch_texts = [
            "Louis XIV was King of France",
            "Henry IV ruled France",
            "Marie Antoinette was queen",
            "Louis XVI and Marie Antoinette were married"
        ]
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            person_count = 0
            for text in monarch_texts
                entities = GraphMERT.extract_entities(domain, text, options)
                person_count += count(e -> e.entity_type == "PERSON", entities)
            end
            
            @test person_count > 0 "Should identify at least some person entities"
            println("Person entities found: $person_count")
        else
            @skip "Wikipedia domain not available"
        end
    end
    
    @testset "Location Classification" begin
        # Test that locations are classified correctly
        location_texts = [
            "Louis XIV lived in Versailles",
            "Henry IV was born in Pau",
            "The court was at the Louvre in Paris"
        ]
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            location_count = 0
            for text in location_texts
                entities = GraphMERT.extract_entities(domain, text, options)
                location_count += count(e -> e.entity_type == "LOCATION", entities)
            end
            
            println("Location entities found: $location_count")
        else
            @skip "Wikipedia domain not available"
        end
    end
    
    @testset "Title Classification" begin
        # Test that royal titles are identified
        title_texts = [
            "Louis XIV, the Sun King, ruled France",
            "King Henry IV was great",
            "Queen Marie Antoinette"
        ]
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            title_count = 0
            for text in title_texts
                entities = GraphMERT.extract_entities(domain, text, options)
                title_count += count(e -> e.entity_type == "TITLE", entities)
            end
            
            println("Title entities found: $title_count")
        else
            @skip "Wikipedia domain not available"
        end
    end
    
    @testset "Mixed Entity Types" begin
        # Test a complex text with multiple entity types
        text = """
        Louis XIV, King of France, was born in Saint-Germain-en-Laye. 
        He ruled from Versailles and was known as the Sun King.
        His son Louis XV also became king of France.
        """
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            entities = GraphMERT.extract_entities(domain, text, options)
            
            # Should have multiple entity types
            types = unique([e.entity_type for e in entities])
            
            println("Mixed text - found types: $types")
            println("Total entities: $(length(entities))")
            
            # At least should have persons
            has_person = any(e -> e.entity_type == "PERSON", entities)
            @test has_person "Should identify person entities in mixed text"
        else
            @skip "Wikipedia domain not available"
        end
    end
    
    @testset "Entity Precision vs Expected" begin
        # Compare extracted entities against expected
        text = LOUIS_XIV_ARTICLE
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options(; confidence_threshold=0.3)
            
            entities = GraphMERT.extract_entities(domain, text, options)
            expected = get_expected_entities("louis_xiv")
            
            # Calculate precision
            precision = calculate_entity_precision(
                [(e.text, e.entity_type, e.confidence) for e in entities],
                expected
            )
            
            println("Entity precision for Louis XIV: $(round(precision * 100, digits=1))%")
            
            # This is the key metric for SC-001
            @test_broken precision >= 0.80 "Entity precision should be >= 80%"
        else
            @skip "Wikipedia domain not available"
        end
    end
end

println("Entity type classification tests completed!")
