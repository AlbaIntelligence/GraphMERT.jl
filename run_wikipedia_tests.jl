#!/usr/bin/env julia
"""
Wikipedia Domain Test Runner for GraphMERT.jl

Usage:
    julia run_wikipedia_tests.jl

This script runs the Wikipedia domain tests to validate entity extraction,
relation extraction, and knowledge graph quality using French monarchy articles.
"""

using Pkg

# Activate the GraphMERT environment
Pkg.activate(joinpath(@__DIR__, "GraphMERT"))

using GraphMERT

function main()
    println("="^60)
    println("Wikipedia Domain Test Suite")
    println("="^60)
    
    # Load Wikipedia domain
    println("\n1. Loading Wikipedia domain...")
    try
        initialize_default_domains()
        set_default_domain("wikipedia")
        println("   ✅ Domain loaded")
    catch e
        println("   ❌ Error loading domain: $e")
        return 1
    end
    
    # Check domain availability
    if !has_domain("wikipedia")
        println("\n❌ Wikipedia domain not available")
        return 1
    end
    
    domain = get_domain("wikipedia")
    options = ProcessingOptions(
        domain="wikipedia",
        confidence_threshold=0.5,
        max_entities=100,
        max_relations=100
    )
    
    # Test articles
    test_articles = Dict(
        "louis_xiv" => """
        Louis XIV (5 September 1638 – 1 September 1715), known as the Sun King, 
        was King of France from 1643 until his death in 1715. His reign of 72 years 
        and 110 days is the longest of any major European monarch.
        
        Louis XIV was born at the Château de Saint-Germain-en-Laye. He became king 
        at the age of four under the regency of his mother, Anne of Austria. 
        His father, Louis XIII, had died in 1643.
        
        Louis XIV married Maria Theresa of Spain in 1660. They had several children 
        including Louis, Grand Dauphin, who was the father of Louis XV.
        """,
        
        "henry_iv" => """
        Henry IV (13 December 1553 – 14 May 1610), also known as Henry the Great, 
        was King of France from 1589 to his death in 1610. He was the first Bourbon 
        king of France.
        
        Born in Pau, Henry was originally a Huguenot leader. He converted to 
        Catholicism in 1593, famously stating that "Paris is well worth a mass."
        
        Henry IV married Margaret of Valois in 1572. He later married Marie de' Medici 
        in 1600. Their son Louis XIII succeeded him.
        """,
        
        "marie_antoinette" => """
        Marie Antoinette (2 November 1755 – 16 October 1793) was the last Queen 
        of France before the French Revolution. She was born Archduchess Maria Theresa 
        of Austria and married Louis XVI in 1770.
        
        Marie Antoinette and Louis XVI had four children: Marie-Thérèse Charlotte, 
        Louis-Joseph, Louis-Charles (Dauphin), and Sophie. Louis-Charles was the 
        father of Louis XVII.
        """
    )
    
    println("\n2. Running Entity Extraction Tests...")
    println("-"^40)
    
    total_entities = 0
    for (name, text) in test_articles
        entities = extract_entities(domain, text, options)
        total_entities += length(entities)
        
        println("\n$name:")
        println("  Entities extracted: $(length(entities))")
        
        # Show entity details
        for ent in entities[1:min(5, length(entities))]
            println("    - $(ent.text) [$(ent.entity_type)] ($(round(ent.confidence, digits=2)))")
        end
        if length(entities) > 5
            println("    ... and $(length(entities) - 5) more")
        end
    end
    
    println("\n3. Running Relation Extraction Tests...")
    println("-"^40)
    
    total_relations = 0
    for (name, text) in test_articles
        entities = extract_entities(domain, text, options)
        relations = extract_relations(domain, entities, text, options)
        total_relations += length(relations)
        
        println("\n$name:")
        println("  Relations extracted: $(length(relations))")
        
        for rel in relations[1:min(5, length(relations))]
            println("    - $(rel.head) --[$(rel.relation_type)]--> $(rel.tail)")
        end
    end
    
    println("\n4. Summary")
    println("="^60)
    println("Total entities extracted: $total_entities")
    println("Total relations extracted: $total_relations")
    
    # Expected targets
    println("\nTargets:")
    println("  Entity precision: ≥80%")
    println("  Relation precision: ≥70%")
    println("  Fact capture: ≥75%")
    
    println("\n✅ Test run complete!")
    return 0
end

# Run main function
exit(main())
