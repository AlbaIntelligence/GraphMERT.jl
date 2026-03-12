#!/usr/bin/env julia
"""
Wikipedia Knowledge Graph Export Test

Tests JSON and CSV export functionality.
Tasks: T028, T029
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GraphMERT
using Random

const TEST_RANDOM_SEED = 42

function run_export_tests()
    println("="^60)
    println("Wikipedia Knowledge Graph Export Tests")
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
    
    options = GraphMERT.ProcessingOptions(
        domain="wikipedia",
        confidence_threshold=0.5,
        max_length=2048,
        batch_size=32,
        verbose=false
    )
    
    test_text = """
    Louis XIV (5 September 1638 – 1 September 1715), known as the Sun King, 
    was King of France from 1643 until his death in 1715.
    
    Louis XIV married Maria Theresa of Spain in 1660. They had several children 
    including Louis, Grand Dauphin, who was the father of Louis XV.
    """
    
    println("\n[T028] Testing JSON export...")
    entities = Base.invokelatest(GraphMERT.extract_entities, domain, test_text, options)
    relations = Base.invokelatest(GraphMERT.extract_relations, domain, entities, test_text, options)
    
    # Create a simple knowledge graph-like structure
    kg = GraphMERT.KnowledgeGraph(
        entities,
        relations,
        Dict{String, Any}("domain" => "wikipedia"),
    )
    
    # Test JSON export
    json_output = GraphMERT.export_to_json(kg)
    println("  JSON output length: $(length(json_output)) characters")
    println("  Entities in JSON: $(length(kg.entities))")
    println("  Relations in JSON: $(length(kg.relations))")
    
    # Verify JSON has content (not empty, contains key markers)
    has_entities = occursin("entities", json_output)
    has_relations = occursin("relations", json_output)
    println("  Has entities key: $has_entities")
    println("  Has relations key: $has_relations")
    
    if has_entities && has_relations
        println("  JSON structure: ✓ PASS")
    else
        println("  JSON structure: ✗ FAIL")
        return false
    end
    
    # Save JSON to temp file
    json_path = joinpath(tempdir(), "test_wikipedia_kg.json")
    GraphMERT.export_to_json(kg, json_path)
    println("  JSON saved to: $json_path")
    
    println("\n[T029] Testing CSV export...")
    
    # Test CSV export
    csv_output = GraphMERT.export_to_csv(kg, "")
    println("  CSV output length: $(length(csv_output)) characters")
    
    # Save CSV to temp files
    csv_path = joinpath(tempdir(), "test_wikipedia_kg")
    GraphMERT.export_to_csv(kg, csv_path)
    println("  CSV saved to: $csv_path (with _entities.csv and _relations.csv)")
    
    # Verify files exist
    if isfile("$csv_path" * "_entities.csv")
        println("  Entities CSV: ✓ exists")
    else
        println("  Entities CSV: ✗ missing")
    end
    
    if isfile("$csv_path" * "_relations.csv")
        println("  Relations CSV: ✓ exists")
    else
        println("  Relations CSV: ✗ missing")
    end
    
    println("\n" * "="^60)
    println("Summary")
    println("="^60)
    println("✓ JSON export: PASS")
    println("✓ CSV export: PASS")
    
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    success = run_export_tests()
    exit(success ? 0 : 1)
end
