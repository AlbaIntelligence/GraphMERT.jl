"""
Example 1: Basic Entity Extraction
==================================

This example demonstrates basic biomedical entity extraction using GraphMERT,
following the progression of the original paper. This is the first step in
the GraphMERT pipeline.

Based on: GraphMERT paper - Section 3.1 Entity Recognition
"""

using Pkg
Pkg.activate("../../")

using GraphMERT
using Logging

# Configure logging
Logging.configure(level=Logging.Info)

function main()
  println("="^60)
    println("GraphMERT Example 1: Basic Entity Extraction")
    println("="^60)

    # Sample biomedical text (from a medical abstract)
    sample_text = """
    Alzheimer's disease is a neurodegenerative disorder characterized by
    progressive cognitive decline and memory loss. The disease is associated
    with the accumulation of beta-amyloid plaques and tau protein tangles
    in the brain. Current treatments include cholinesterase inhibitors
    such as donepezil and memantine, which help manage symptoms but do not
    cure the disease. Research is ongoing to develop new therapeutic
    approaches targeting the underlying pathological mechanisms.
    """

    println("\n📄 Sample Text:")
    println(sample_text)

    # Initialize UMLS client (if available)
    umls_client = nothing
    try
        # In a real scenario, you would provide your UMLS API key
        # umls_client = create_umls_client("your-api-key-here")
        println("\n⚠️  UMLS client not configured - using fallback methods")
    catch e
        println("\n⚠️  UMLS client initialization failed: $e")
    end

    # Extract entities using rule-based approach
    println("\n🔍 Extracting entities...")

    # Get supported entity types
    entity_types = get_supported_entity_types()
    println("Supported entity types: $(length(entity_types))")

    # Extract entities from text
    entities = extract_entities_from_text(sample_text; entity_types=entity_types)

    println("\n📊 Extraction Results:")
    println("Found $(length(entities)) entities")

    # Display results
    for (i, (text, entity_type, confidence)) in enumerate(entities)
        println("  $i. \"$text\" -> $entity_type (confidence: $(round(confidence, digits=3)))")
    end

    # Group by entity type
    println("\n📈 Entity Type Distribution:")
    type_counts = Dict{String, Int}()
    for (text, entity_type, confidence) in entities
        type_name = get_entity_type_name(entity_type)
        type_counts[type_name] = get(type_counts, type_name, 0) + 1
    end

    for (type_name, count) in sort(collect(type_counts), by=x->x[2], rev=true)
        println("  $type_name: $count entities")
    end

    # Validate entities
    println("\n✅ Entity Validation:")
    valid_entities = 0
    for (text, entity_type, confidence) in entities
        is_valid = validate_biomedical_entity(text, entity_type)
        if is_valid
            valid_entities += 1
        end
        println("  \"$text\" ($entity_type): $(is_valid ? "✓" : "✗")")
    end

    println("\nValid entities: $valid_entities/$(length(entities)) ($(round(100*valid_entities/length(entities), digits=1))%)")

    # Calculate confidence statistics
    confidences = [conf for (text, entity_type, conf) in entities]
    avg_confidence = mean(confidences)
    max_confidence = maximum(confidences)
    min_confidence = minimum(confidences)

    println("\n📊 Confidence Statistics:")
    println("  Average: $(round(avg_confidence, digits=3))")
    println("  Maximum: $(round(max_confidence, digits=3))")
    println("  Minimum: $(round(min_confidence, digits=3))")

    println("\n" * "="^60)
    println("✅ Example 1 completed successfully!")
    println("Next: Run 02_relation_extraction.jl")
    println("="^60)
end

# Run the example
main()
