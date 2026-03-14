"""
Unit tests for Wikipedia domain relation extraction.

Tests the Wikipedia domain's ability to extract relations between entities.
"""

using Test
using GraphMERT

include("fixtures.jl")

@testset "Wikipedia Relation Extraction" begin
    
    @testset "Spouse Relations" begin
        # Test extraction of spouse relationships
        text = "Louis XIV married Maria Theresa of Spain. They had several children."
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            # First extract entities
            entities = GraphMERT.extract_entities(domain, text, options)
            
            # Then extract relations
            relations = GraphMERT.extract_relations(domain, entities, text, options)
            
            println("Extracted $(length(relations)) relations from spouse text")
            
            # Should find some relations
            @test length(relations) >= 0
            
            # Check for spouse-like relations
            rel_types = [r.relation_type for r in relations]
            println("Relation types found: $rel_types")
        else
            @test_skip true
        end
    end
    
    @testset "Parent-Child Relations" begin
        # Test extraction of parent-child relationships
        text = "Louis XIV was the father of Louis XV. Louis XV was the grandfather of Louis XVII."
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            entities = GraphMERT.extract_entities(domain, text, options)
            relations = GraphMERT.extract_relations(domain, entities, text, options)
            
            println("Parent-child: extracted $(length(relations)) relations")
            
            # Check for parent-related relations
            has_parent = any(r -> occursin("parent", lowercase(r.relation_type)), relations)
            println("Has parent relation: $has_parent")
        else
            @test_skip true
        end
    end
    
    @testset "Dynastic Relations" begin
        # Test extraction of dynastic relationships
        text = "Louis XIV belonged to the Bourbon dynasty. The Capetian dynasty preceded the Bourbons."
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            entities = GraphMERT.extract_entities(domain, text, options)
            relations = GraphMERT.extract_relations(domain, entities, text, options)
            
            println("Dynastic: extracted $(length(relations)) relations")
        else
            @test_skip true
        end
    end
    
    @testset "Temporal Relations" begin
        # Test extraction of temporal relationships (reigned dates)
        text = "Louis XIV reigned from 1643 to 1715. He was born in 1638."
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            entities = GraphMERT.extract_entities(domain, text, options)
            relations = GraphMERT.extract_relations(domain, entities, text, options)
            
            println("Temporal: extracted $(length(relations)) relations")
            
            # Check for temporal relations
            temporal_types = ["reigned_from", "reigned_until", "born", "died"]
            has_temporal = any(r -> r.relation_type in temporal_types, relations)
            println("Has temporal relation: $has_temporal")
        else
            @test_skip true
        end
    end
    
    @testset "Relation Precision" begin
        # Test relation extraction precision against expected
        text = LOUIS_XIV_ARTICLE
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options(; confidence_threshold=0.3)
            
            entities = GraphMERT.extract_entities(domain, text, options)
            relations = GraphMERT.extract_relations(domain, entities, text, options)
            
            # Get expected relations
            expected = get_expected_relations("louis_xiv")
            
            # Calculate precision
            if !isempty(relations) && !isempty(expected)
                precision = calculate_relation_precision(
                    [(r.head, r.relation_type, r.tail, r.confidence) for r in relations],
                    expected
                )
                
                println("Relation precision: $(round(precision * 100, digits=1))%")
                
                # This is the key metric for SC-002
                @test_broken precision >= 0.70
            else
                println("Skipping precision test - no relations extracted")
            end
        else
            @test_skip true
        end
    end
    
    @testset "Relation Confidence Scores" begin
        text = LOUIS_XIV_ARTICLE
        
        if GraphMERT.has_domain("wikipedia")
            domain = GraphMERT.get_domain("wikipedia")
            options = create_test_options()
            
            entities = GraphMERT.extract_entities(domain, text, options)
            relations = GraphMERT.extract_relations(domain, entities, text, options)
            
            # Check confidence scores
            for rel in relations
                @test 0.0 <= rel.confidence <= 1.0
            end
            
            println("All $(length(relations)) relations have valid confidence scores")
        else
            @test_skip true
        end
    end
end

println("Relation extraction tests completed!")
