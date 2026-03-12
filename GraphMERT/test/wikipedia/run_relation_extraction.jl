#!/usr/bin/env julia
"""
Wikipedia Relation Extraction Test Runner

Runs relation extraction on French monarchy articles and validates results.
Tasks: T014, T015, T016
"""

# Use parent project
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GraphMERT
using Random

const TEST_RANDOM_SEED = 42

function run_relation_extraction_tests()
    println("="^60)
    println("Wikipedia Relation Extraction Tests")
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
    
    test_text = """
    Louis XIV (5 September 1638 – 1 September 1715), known as the Sun King, 
    was King of France from 1643 until his death in 1715. His reign of 72 years 
    and 110 days is the longest of any major European monarch.
    
    Louis XIV was born at the Château de Saint-Germain-en-Laye. He became king 
    at the age of four under the regency of his mother, Anne of Austria. 
    His father, Louis XIII, had died in 1643.
    
    Louis XIV married Maria Theresa of Spain in 1660. They had several children 
    including Louis, Grand Dauphin, who was the father of Louis XV.
    
    Maria Theresa of Spain (10 June 1638 – 30 July 1683) was Queen of France 
    as the wife of Louis XIV. She was a daughter of Philip IV of Spain.
    
    Louis XV (15 February 1710 – 10 May 1774) was King of France from 1715 
    until his death. He was the great-grandson of Louis XIV and succeeded him.
    """
    
    expected_relations = [
        ("Louis XIV", "parent_of", "Louis XV"),
        ("Louis XIV", "spouse_of", "Maria Theresa of Spain"),
        ("Louis XIV", "reigned_after", "Louis XIII"),
        ("Louis XIII", "parent_of", "Louis XIV"),
    ]
    
    options = GraphMERT.ProcessingOptions(
        domain="wikipedia",
        confidence_threshold=0.5,
        max_length=2048,
        batch_size=32,
        verbose=true
    )
    
    println("\n[Step 1] Extracting entities...")
    entities = Base.invokelatest(GraphMERT.extract_entities, domain, test_text, options)
    println("Extracted $(length(entities)) entities")
    
    println("\n[Step 2] Extracting relations...")
    relations = Base.invokelatest(GraphMERT.extract_relations, domain, entities, test_text, options)
    println("Extracted $(length(relations)) relations")
    
    # Build entity lookup by ID
    entity_by_id = Dict{String, String}()
    for ent in entities
        entity_by_id[ent.id] = ent.text
    end
    
    println("\nExtracted relations:")
    for rel in relations
        head_text = get(entity_by_id, rel.head, rel.head)
        tail_text = get(entity_by_id, rel.tail, rel.tail)
        println("  - $head_text --$(rel.relation_type)--> $tail_text (confidence: $(round(rel.confidence, digits=2)))")
    end
    
    println("\nExpected relations:")
    for (head, rel_type, tail) in expected_relations
        println("  - $head --$rel_type--> $tail")
    end
    
    # Build entity lookup by ID (case-insensitive)
    entity_by_id = Dict{String, String}()
    for ent in entities
        entity_by_id[ent.id] = ent.text
        entity_by_id[lowercase(ent.text)] = ent.id
    end
    
    # Check for matching relations using entity texts
    matches = 0
    for (exp_head, exp_rel, exp_tail) in expected_relations
        for rel in relations
            head_text = get(entity_by_id, rel.head, "")
            tail_text = get(entity_by_id, rel.tail, "")
            # Case-insensitive comparison
            if lowercase(head_text) == lowercase(exp_head) && 
               lowercase(rel.relation_type) == lowercase(exp_rel) && 
               lowercase(tail_text) == lowercase(exp_tail)
                matches += 1
                break
            end
        end
    end
    
    precision = length(relations) > 0 ? matches / length(relations) : 0.0
    recall = length(expected_relations) > 0 ? matches / length(expected_relations) : 0.0
    
    println("\n" * "="^60)
    println("Results")
    println("="^60)
    println("Matches: $matches / $(length(expected_relations))")
    println("Precision: $(round(precision * 100, digits=1))%")
    println("Recall: $(round(recall * 100, digits=1))%")
    
    if precision >= 0.70
        println("\n✓ SC-002 PASSED: Relation extraction achieves at least 70% precision")
        return true
    else
        println("\n✗ SC-002 FAILED: Relation extraction below 70% precision threshold")
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    success = run_relation_extraction_tests()
    exit(success ? 0 : 1)
end
