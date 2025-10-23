"""
UMLS (Unified Medical Language System) Integration Demo

This example demonstrates the GraphMERT UMLS integration for biomedical
entity linking and validation. UMLS provides standardized medical terminology
and relationships for accurate biomedical knowledge graph construction.

Key concepts demonstrated:
1. UMLS client creation with authentication
2. Entity linking using search and similarity matching
3. Concept details retrieval (semantic types, definitions)
4. Rate limiting and caching for API efficiency
5. Error handling and fallback mechanisms
"""

using GraphMERT

println("=== UMLS Integration Demo ===")

# 1. Create UMLS client (requires API key)
println("\n1. Creating UMLS client...")
try
    # Note: In practice, you would use a real API key from NLM UTS
    # client = create_umls_client(ENV["UMLS_API_KEY"])

    # For demo, we'll simulate the client creation
    println("UMLS client creation requires a valid API key from NLM UTS")
    println("Example: client = create_umls_client(ENV[\"UMLS_API_KEY\"])")
    println("API key can be obtained from: https://uts.nlm.nih.gov/uts/")

    # Simulate client for demonstration
    println("Simulating UMLS client with demo data...")

    # 2. Demonstrate entity linking
    println("\n2. Demonstrating entity linking...")

    # Sample biomedical entities from diabetes text
    entities = ["diabetes", "metformin", "insulin", "cardiovascular disease"]

    println("Sample entities: $entities")

    # Simulate linking results using the implemented functions
    simulated_results = []
    for entity in entities
        # Simulate the linking process
        if entity == "diabetes"
            push!(simulated_results, ("diabetes", "C0011849", "Diabetes Mellitus", ["Disease", "Endocrine System Disease"], 0.95, "umls_search"))
        elseif entity == "metformin"
            push!(simulated_results, ("metformin", "C0025598", "Metformin", ["Pharmacologic Substance", "Organic Chemical"], 0.92, "umls_search"))
        elseif entity == "insulin"
            push!(simulated_results, ("insulin", "C0021641", "Insulin", ["Amino Acid, Peptide, or Protein", "Hormone", "Pharmacologic Substance"], 0.88, "umls_search"))
        elseif entity == "cardiovascular disease"
            push!(simulated_results, ("cardiovascular disease", "C0007222", "Cardiovascular Diseases", ["Disease"], 0.85, "umls_search"))
        end
    end

    println("Simulated UMLS linking results:")
    for (entity, cui, name, semantic_types, score, source) in simulated_results
        println("  '$entity' → CUI: $cui")
        println("    Preferred name: $name")
        println("    Semantic types: $(join(semantic_types, ", "))")
        println("    Confidence: $(round(score, digits=3))")
        println("    Source: $source")
        println()
    end

    # 3. Demonstrate concept details
    println("3. Demonstrating concept details...")

    # Focus on diabetes concept
    diabetes_cui = "C0011849"
    println("Diabetes concept (CUI: $diabetes_cui):")
    println("  In real implementation, would retrieve:")
    println("  - Definitions and descriptions")
    println("  - Related concepts and synonyms")
    println("  - Hierarchical relationships")
    println("  - Associated procedures and findings")

    # 4. Demonstrate semantic type classification
    println("\n4. Semantic type classification:")

    semantic_type_counts = Dict{String, Int}()
    for (_, _, _, semantic_types, _, _) in simulated_results
        for st in semantic_types
            semantic_type_counts[st] = get(semantic_type_counts, st, 0) + 1
        end
    end

    println("Semantic type distribution:")
    for (st, count) in sort(collect(semantic_type_counts), by=x->x[2], rev=true)
        println("  $st: $count entities")
    end

    # 5. Demonstrate caching and rate limiting
    println("\n5. Caching and rate limiting:")
    println("  Cache TTL: 3600 seconds (1 hour)")
    println("  Rate limit: 100 requests/minute")
    println("  Automatic retry with exponential backoff")
    println("  Local caching reduces API calls and improves performance")

    # 6. Integration with knowledge graph extraction
    println("\n6. Integration with knowledge graph extraction:")
    println("  UMLS linking enhances entity extraction by:")
    println("  • Providing standardized concept identifiers (CUIs)")
    println("  • Adding semantic type classification")
    println("  • Enabling concept hierarchy traversal")
    println("  • Supporting relation validation")
    println("  • Improving extraction accuracy and consistency")

    # 7. Error handling and fallbacks
    println("\n7. Error handling and fallbacks:")
    println("  • Graceful degradation when API unavailable")
    println("  • Fallback to local entity recognition")
    println("  • Rate limit handling with automatic retry")
    println("  • Timeout handling with configurable limits")
    println("  • Cache-based offline operation")

    println("\n✅ UMLS integration demo complete!")

    println("\nTo use UMLS in practice:")
    println("1. Get API key from: https://uts.nlm.nih.gov/uts/")
    println("2. Create client: client = create_umls_client(api_key)")
    println("3. Link entities: result = link_entity(client, \"diabetes\")")
    println("4. Get details: details = get_concept_details(client, result.cui)")
    println("5. Use in extraction: kg = extract_knowledge_graph(text, model, umls_client=client)")

    println("\nNext steps for complete implementation:")
    println("• Implement actual HTTP API integration")
    println("• Add comprehensive UMLS relation retrieval")
    println("• Create semantic type-based filtering")
    println("• Optimize caching for large-scale processing")
    println("• Add batch processing for multiple entities")

catch e
    println("UMLS demo completed with simulated functionality")
    println("Error: $e")
    println("This is expected without a real UMLS API key")
    println("To run with real UMLS:")
    println("1. Get API key from NLM UTS: https://uts.nlm.nih.gov/uts/")
    println("2. Set environment variable: ENV[\"UMLS_API_KEY\"] = \"your_key\"")
    println("3. Create client: client = create_umls_client(ENV[\"UMLS_API_KEY\"])")
end
