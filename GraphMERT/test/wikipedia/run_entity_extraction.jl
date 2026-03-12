#!/usr/bin/env julia
"""
Wikipedia Entity Extraction Test Runner

Runs entity extraction on French monarchy articles and validates results.
Tasks: T009, T010, T011, T012
"""

# Use parent project
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GraphMERT
using Random

const TEST_RANDOM_SEED = 42

function run_entity_extraction_tests()
    println("="^60)
    println("Wikipedia Entity Extraction Tests")
    println("="^60)
    
    Random.seed!(TEST_RANDOM_SEED)
    
    println("\n[Setup] Loading Wikipedia domain...")
    
    if !GraphMERT.has_domain("wikipedia")
        try
            domain = GraphMERT.load_wikipedia_domain()
            GraphMERT.register_domain!("wikipedia", domain)
            println("[Setup] Wikipedia domain registered successfully")
        catch e
            println("[ERROR] Could not load Wikipedia domain: $e")
            return false
        end
    end
    
    domain = GraphMERT.get_domain("wikipedia")
    
    test_articles = [
        ("Louis XIV", """
        Louis XIV (5 September 1638 – 1 September 1715), known as the Sun King, 
        was King of France from 1643 until his death in 1715. His reign of 72 years 
        and 110 days is the longest of any major European monarch.
        
        Louis XIV was born at the Château de Saint-Germain-en-Laye. He became king 
        at the age of four under the regency of his mother, Anne of Austria. 
        His father, Louis XIII, had died in 1643.
        
        Louis XIV married Maria Theresa of Spain in 1660. They had several children 
        including Louis, Grand Dauphin, who was the father of Louis XV.
        """),
        
        ("Henry IV", """
        Henry IV (13 December 1553 – 14 May 1610), also known as Henry the Great, 
        was King of France from 1589 to his death in 1610. He was the first Bourbon 
        king of France.
        
        Born in Pau, Henry was originally a Huguenot leader. He converted to 
        Catholicism in 1593, famously stating that "Paris is well worth a mass."
        
        Henry IV married Margaret of Valois in 1572. He later married Marie de' Medici 
        in 1600. Their son Louis XIII succeeded him.
        """),
        
        ("Marie Antoinette", """
        Marie Antoinette (2 November 1755 – 16 October 1793) was the last Queen 
        of France before the French Revolution. She was born Archduchess Maria Theresa 
        of Austria and married Louis XVI in 1770.
        
        Marie Antoinette and Louis XVI had four children: Marie-Thérèse Charlotte, 
        Louis-Joseph, Louis-Charles (Dauphin), and Sophie.
        
        Marie Antoinette was executed by guillotine in Paris.
        """)
    ]
    
    expected_entities = Dict(
        "Louis XIV" => ["Louis XIV", "France", "Louis XIII", "Anne of Austria", 
                        "Maria Theresa of Spain", "Louis XV", "Palace of Versailles"],
        "Henry IV" => ["Henry IV", "France", "Bourbon", "Paris", "Margaret of Valois", 
                       "Marie de' Medici", "Louis XIII"],
        "Marie Antoinette" => ["Marie Antoinette", "Louis XVI", "Maria Theresa of Austria", 
                              "Marie-Thérèse Charlotte", "Louis-Charles", "Paris"]
    )
    
    options = GraphMERT.ProcessingOptions(
        domain="wikipedia",
        confidence_threshold=0.5,
        max_length=2048,
        batch_size=32,
        verbose=true
    )
    
    total_precision = 0.0
    total_recall = 0.0
    article_count = 0
    
    for (name, text) in test_articles
        println("\n" * "="^60)
        println("Testing: $name")
        println("="^60)
        
        try
            entities = Base.invokelatest(GraphMERT.extract_entities, domain, text, options)
            
            println("Extracted $(length(entities)) entities:")
            entity_texts = String[]
            for ent in entities
                println("  - $(ent.text) ($(ent.entity_type)) confidence: $(round(ent.confidence, digits=2))")
                push!(entity_texts, ent.text)
            end
            
            expected = get(expected_entities, name, String[])
            println("\nExpected entities: $expected")
            
            matches = count(e -> e in entity_texts, expected)
            precision = length(entities) > 0 ? matches / length(entities) : 0.0
            recall = length(expected) > 0 ? matches / length(expected) : 0.0
            
            println("\nResults:")
            println("  Precision: $(round(precision * 100, digits=1))%")
            println("  Recall: $(round(recall * 100, digits=1))%")
            
            total_precision += precision
            total_recall += recall
            article_count += 1
            
            if recall >= 0.80
                println("  ✓ Meets 80% recall threshold (SC-001)")
            else
                println("  ✗ Below 80% recall threshold")
            end
            
        catch e
            println("[ERROR] Extraction failed: $e")
        end
    end
    
    println("\n" * "="^60)
    println("Summary")
    println("="^60)
    
    if article_count > 0
        avg_precision = total_precision / article_count
        avg_recall = total_recall / article_count
        println("Average Precision: $(round(avg_precision * 100, digits=1))%")
        println("Average Recall: $(round(avg_recall * 100, digits=1))%")
        
        if avg_recall >= 0.80
            println("\n✓ SC-001 PASSED: Entity extraction achieves at least 80% recall")
            return true
        else
            println("\n✗ SC-001 FAILED: Entity extraction below 80% recall threshold")
            return false
        end
    else
        println("No articles tested")
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    success = run_entity_extraction_tests()
    exit(success ? 0 : 1)
end
