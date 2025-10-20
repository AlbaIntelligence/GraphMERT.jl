"""
Example 3: Knowledge Graph Construction
=======================================

This example demonstrates the construction of a biomedical knowledge graph
using the entities and relations extracted in previous examples, following
the progression of the original GraphMERT paper.

Based on: GraphMERT paper - Section 3.3 Knowledge Graph Construction
"""

using Pkg
Pkg.activate(; temp = true)
Pkg.add(["Revise", "Logging", "Dates", "HTTP", "JSON"])
using Revise
using Dates, HTTP, JSON, Logging

Pkg.develop(path = "./GraphMERT")
using GraphMERT

# Configure logging
global_logger(ConsoleLogger(stderr, Logging.Info))

function main()
    println("="^60)
    println("GraphMERT Example 3: Knowledge Graph Construction")
    println("="^60)

    # Sample biomedical texts for knowledge graph construction
    texts = [
        """
        Alzheimer's disease is a neurodegenerative disorder characterized by
        progressive cognitive decline and memory loss. The disease is associated
        with the accumulation of beta-amyloid plaques and tau protein tangles
        in the brain.
        """,
        """
        Donepezil is a cholinesterase inhibitor that treats Alzheimer's disease
        by preventing the breakdown of acetylcholine in the brain. The drug
        binds to acetylcholinesterase enzyme and inhibits its activity.
        """,
        """
        Memantine is an NMDA receptor antagonist used to treat moderate to
        severe Alzheimer's disease. It works by blocking excessive glutamate
        activity in the brain, which can damage nerve cells.
        """,
        """
        Acetylcholine is a neurotransmitter that plays a crucial role in
        memory and learning. It is synthesized by choline acetyltransferase
        and broken down by acetylcholinesterase.
        """,
    ]

    println("\n📄 Processing $(length(texts)) biomedical texts...")

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

    # Step 1: Extract entities from all texts
    println("\n🔍 Step 1: Extracting entities from all texts...")
    all_entities = Vector{Tuple{String, BiomedicalEntityType, Float64}}()

    for (i, text) in enumerate(texts)
        println("  Processing text $i/$(length(texts))...")
        entities = extract_entities_from_text(text; entity_types = get_supported_entity_types())
        append!(all_entities, entities)
    end

    # Remove duplicates and merge entities
    println("\n🔄 Step 2: Merging and deduplicating entities...")
    unique_entities = merge_entities(all_entities)
    println("  Original entities: $(length(all_entities))")
    println("  Unique entities: $(length(unique_entities))")

    # Step 3: Extract relations
    println("\n🔗 Step 3: Extracting relations...")
    all_relations = Vector{Dict{String, Any}}()

    for (i, text) in enumerate(texts)
        println("  Processing text $i/$(length(texts))...")

        # Find entities in this text
        text_entities = extract_entities_from_text(text; entity_types = get_supported_entity_types())

        # Extract relations between entities in this text
        for j in 1:length(text_entities)
            for k in (j+1):length(text_entities)
                head_entity = text_entities[j][1]
                tail_entity = text_entities[k][1]

                # Classify relation
                relation_type = classify_relation(head_entity, tail_entity, text; umls_client = umls_client)

                if relation_type != UNKNOWN_RELATION
                    confidence = calculate_relation_confidence(head_entity, tail_entity, relation_type, text)

                    if validate_biomedical_relation(head_entity, tail_entity, relation_type)
                        push!(all_relations, Dict(
                            "head_entity" => head_entity,
                            "tail_entity" => tail_entity,
                            "relation_type" => get_relation_type_name(relation_type),
                            "confidence" => confidence,
                            "context" => text,
                        ))
                    end
                end
            end
        end
    end

    println("  Found $(length(all_relations)) relations")

    # Step 4: Create knowledge graph
    println("\n🕸️  Step 4: Creating knowledge graph...")

    # Convert entities to BiomedicalEntity objects
    biomedical_entities = Vector{BiomedicalEntity}()
    for (i, (text, entity_type, confidence)) in enumerate(unique_entities)
        entity = BiomedicalEntity(
            "entity_$i",
            text,
            get_entity_type_name(entity_type),
            confidence,
            TextPosition(1, length(text), 1, 1),
            Dict{String, Any}(),
            Dates.now(),
        )
        push!(biomedical_entities, entity)
    end

    # Convert relations to BiomedicalRelation objects
    biomedical_relations = Vector{BiomedicalRelation}()
    for (i, relation) in enumerate(all_relations)
        # Find entity IDs
        head_id = find_entity_id(biomedical_entities, relation["head_entity"])
        tail_id = find_entity_id(biomedical_entities, relation["tail_entity"])

        if head_id !== nothing && tail_id !== nothing
            rel = BiomedicalRelation(
                head = head_id,
                tail = tail_id,
                relation_type = relation["relation_type"],
                confidence = relation["confidence"],
                attributes = Dict{String, Any}("context" => relation["context"]),
                created_at = now(),
            )
            push!(biomedical_relations, rel)
        end
    end

    # Build the knowledge graph
    kg = build_biomedical_graph(biomedical_entities, biomedical_relations; umls_client = umls_client)

    println("  Knowledge graph created successfully!")

    # Step 5: Analyze the knowledge graph
    println("\n📊 Step 5: Analyzing knowledge graph...")

    # Basic statistics
    stats = analyze_biomedical_graph(kg)

    println("\n📈 Knowledge Graph Statistics:")
    println("  Total entities: $(stats["total_entities"])")
    println("  Total relations: $(stats["total_relations"])")
    println("  UMLS mapped entities: $(stats["umls_mapped_entities"])")
    println("  UMLS coverage: $(round(stats["umls_coverage"] * 100, digits=1))%")

    # Entity type distribution
    println("\n🏷️  Entity Type Distribution:")
    for (type_name, count) in sort(collect(stats["entity_types"]), by = x->x[2], rev = true)
        println("  $type_name: $count")
    end

    # Relation type distribution
    println("\n🔗 Relation Type Distribution:")
    for (rel_type, count) in sort(collect(stats["relation_types"]), by = x->x[2], rev = true)
        println("  $rel_type: $count")
    end

    # Confidence statistics
    println("\n📊 Confidence Statistics:")
    println("  Average entity confidence: $(round(stats["avg_entity_confidence"], digits=3))")
    println("  Average relation confidence: $(round(stats["avg_relation_confidence"], digits=3))")
    println("  Min entity confidence: $(round(stats["min_entity_confidence"], digits=3))")
    println("  Max entity confidence: $(round(stats["max_entity_confidence"], digits=3))")

    # Graph metrics
    println("\n📐 Graph Metrics:")
    metrics = calculate_graph_metrics(kg)
    println("  Density: $(round(metrics["density"], digits=3))")
    println("  Average out-degree: $(round(metrics["avg_out_degree"], digits=2))")
    println("  Average in-degree: $(round(metrics["avg_in_degree"], digits=2))")
    println("  Max out-degree: $(metrics["max_out_degree"])")
    println("  Max in-degree: $(metrics["max_in_degree"])")

    # Connected components
    println("\n🔗 Connected Components:")
    components = find_connected_components(kg)
    println("  Number of components: $(length(components))")
    println("  Largest component size: $(maximum(length.(components)))")

    # Step 6: Export knowledge graph
    println("\n💾 Step 6: Exporting knowledge graph...")

    # Export to JSON
    json_data = export_to_json(kg)
    println("  Knowledge graph exported to JSON format")
    println("  JSON size: $(length(JSON.json(json_data))) characters")
    println("  JSON Data: $(JSON.json(json_data)) characters")

    # Show sample of exported data
    println("\n📄 Sample Exported Data:")
    println("  Entities: $(length(json_data["entities"]))")
    println("  Relations: $(length(json_data["relations"]))")
    println("  UMLS mappings: $(length(json_data["umls_mappings"]))")

    println("\n" * "="^60)
    println("✅ Example 3 completed successfully!")
    println("Next: Run 04_training_pipeline.jl")
    println("="^60)
end

# Helper function to merge entities
function merge_entities(entities::Vector{Tuple{String, BiomedicalEntityType, Float64}})
    entity_map = Dict{String, Tuple{String, BiomedicalEntityType, Float64}}()

    for (text, entity_type, confidence) in entities
        normalized_text = normalize_entity_text(text)

        if haskey(entity_map, normalized_text)
            # Update confidence if higher
            existing_confidence = entity_map[normalized_text][3]
            if confidence > existing_confidence
                entity_map[normalized_text] = (text, entity_type, confidence)
            end
        else
            entity_map[normalized_text] = (text, entity_type, confidence)
        end
    end

    return collect(values(entity_map))
end

# Helper function to find entity ID
function find_entity_id(entities::Vector{BiomedicalEntity}, text::String)
    for entity in entities
        if entity.text == text
            return entity.id
        end
    end
    return nothing
end

# Run the example
main()
