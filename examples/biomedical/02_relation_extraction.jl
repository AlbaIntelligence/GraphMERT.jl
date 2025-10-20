"""
Example 2: Relation Extraction
==============================

This example demonstrates relation extraction between biomedical entities,
following the progression of the original GraphMERT paper. This builds on
the entity extraction from Example 1.

Based on: GraphMERT paper - Section 3.2 Relation Extraction
"""

using Pkg
Pkg.activate(; temp = true)
Pkg.add(["Revise", "Logging"])
using Revise
using Logging

Pkg.develop(path = "./GraphMERT")
using GraphMERT

# Configure logging
global_logger(ConsoleLogger(stderr, Logging.Info))

function main()
    println("="^60)
    println("GraphMERT Example 2: Relation Extraction")
    println("="^60)

    # Sample biomedical text with clear relations
    sample_text = """
    Donepezil is a cholinesterase inhibitor that treats Alzheimer's disease
    by preventing the breakdown of acetylcholine in the brain. The drug
    binds to acetylcholinesterase enzyme and inhibits its activity, which
    leads to increased acetylcholine levels. This medication is commonly
    prescribed for patients with mild to moderate Alzheimer's disease and
    has been shown to improve cognitive function and reduce symptoms.
    However, donepezil does not cure the disease and only provides
    symptomatic relief.
    """

    println("\n📄 Sample Text:")
    println(sample_text)

    # Initialize clients
    umls_client = nothing
    llm_client = nothing

    try
        # In a real scenario, you would provide your API keys
        # umls_client = create_umls_client("your-umls-api-key")
        # llm_client = create_helper_llm_client("your-llm-api-key")
        println("\n⚠️  External clients not configured - using fallback methods")
    catch e
        println("\n⚠️  Client initialization failed: $e")
    end

    # Step 1: Extract entities
    println("\n🔍 Step 1: Extracting entities...")
    entity_types = get_supported_entity_types()
    entities = extract_entities_from_text(sample_text; entity_types = entity_types)

    println("Found $(length(entities)) entities:")
    for (i, (text, entity_type, confidence)) in enumerate(entities)
        println("  $i. \"$text\" -> $entity_type")
    end

    # Step 2: Extract relations
    println("\n🔗 Step 2: Extracting relations...")

    # Convert entities to the format expected by relation extraction
    entity_list = [Dict("text" => text, "type" => get_entity_type_name(entity_type),
        "confidence" => confidence) for (text, entity_type, confidence) in entities]

    # Extract relations using rule-based approach
    relations = Vector{Dict{String, Any}}()

    # Find relations between entities
    for i in 1:length(entity_list)
        for j in (i+1):length(entity_list)
            head_entity = entity_list[i]["text"]
            tail_entity = entity_list[j]["text"]

            # Classify relation using rule-based approach
            relation_type = classify_relation(head_entity, tail_entity, sample_text; umls_client = umls_client)

            if relation_type != UNKNOWN_RELATION
                # Calculate confidence
                confidence = calculate_relation_confidence(head_entity, tail_entity, relation_type, sample_text)

                # Validate relation
                is_valid = validate_biomedical_relation(head_entity, tail_entity, relation_type)

                if is_valid
                    push!(relations, Dict(
                        "head_entity" => head_entity,
                        "tail_entity" => tail_entity,
                        "relation_type" => get_relation_type_name(relation_type),
                        "confidence" => confidence,
                        "context" => sample_text,
                    ))
                end
            end
        end
    end

    println("\n📊 Relation Extraction Results:")
    println("Found $(length(relations)) relations")

    # Display results
    for (i, relation) in enumerate(relations)
        println("  $i. $(relation["head_entity"]) --[$(relation["relation_type"])]--> $(relation["tail_entity"])")
        println("     Confidence: $(round(relation["confidence"], digits=3))")
    end

    # Group by relation type
    println("\n📈 Relation Type Distribution:")
    relation_counts = Dict{String, Int}()
    for relation in relations
        rel_type = relation["relation_type"]
        relation_counts[rel_type] = get(relation_counts, rel_type, 0) + 1
    end

    for (rel_type, count) in sort(collect(relation_counts), by = x->x[2], rev = true)
        println("  $rel_type: $count relations")
    end

    # Analyze relation confidence
    if !isempty(relations)
        confidences = [rel["confidence"] for rel in relations]
        avg_confidence = mean(confidences)
        max_confidence = maximum(confidences)
        min_confidence = minimum(confidences)

        println("\n📊 Relation Confidence Statistics:")
        println("  Average: $(round(avg_confidence, digits=3))")
        println("  Maximum: $(round(max_confidence, digits=3))")
        println("  Minimum: $(round(min_confidence, digits=3))")
    end

    # Create a simple knowledge graph representation
    println("\n🕸️  Knowledge Graph Representation:")
    println("Entities: $(length(entities))")
    println("Relations: $(length(relations))")

    # Show entity-relation network
    println("\nNetwork Structure:")
    for relation in relations
        head = relation["head_entity"]
        tail = relation["tail_entity"]
        rel_type = relation["relation_type"]
        println("  $head --[$rel_type]--> $tail")
    end

    # Calculate graph density
    num_entities = length(entities)
    num_relations = length(relations)
    max_possible_relations = num_entities * (num_entities - 1)
    density = max_possible_relations > 0 ? num_relations / max_possible_relations : 0.0

    println("\n📊 Graph Statistics:")
    println("  Entities: $num_entities")
    println("  Relations: $num_relations")
    println("  Density: $(round(density, digits=3))")

    println("\n" * "="^60)
    println("✅ Example 2 completed successfully!")
    println("Next: Run 03_knowledge_graph_construction.jl")
    println("="^60)
end

# Run the example
main()
