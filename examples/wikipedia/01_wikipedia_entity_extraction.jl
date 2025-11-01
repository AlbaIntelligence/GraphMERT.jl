"""
Wikipedia Example 1: Entity Extraction from Wikipedia Pages - Domain System Version
==================================================================================

This example demonstrates entity extraction from Wikipedia pages using the domain system
to show the performance of GraphMERT methods on non-biomedical text. We'll use
Wikipedia articles about various topics to extract entities and relations.

This demonstrates the generalizability of the GraphMERT approach beyond
biomedical domains and shows how to use the domain system.
"""

using Pkg
Pkg.activate(; temp = true)
Pkg.add(["Revise", "Logging", "HTTP", "JSON"])
using Revise
using HTTP, JSON, Logging
using Statistics

Pkg.develop(path = "./GraphMERT")
using GraphMERT

# Configure logging
global_logger(ConsoleLogger(stderr, Logging.Info))

function main()
    println("="^60)
    println("GraphMERT Wikipedia Example 1: Entity Extraction (Domain System)")
    println("="^60)

    # Load and register the Wikipedia domain
    println("\n1. Loading Wikipedia domain...")
    include("../../GraphMERT/src/domains/wikipedia.jl")

    wiki_domain = load_wikipedia_domain()
    register_domain!("wikipedia", wiki_domain)
    set_default_domain("wikipedia")

    println("   ✅ Wikipedia domain loaded and registered\n")

    # Sample Wikipedia-style texts (simplified versions)
    wikipedia_texts = [
        """
        Artificial Intelligence is a branch of computer science that aims to
        create machines capable of intelligent behavior. Machine learning is
        a subset of AI that enables computers to learn without being explicitly
        programmed. Deep learning uses neural networks with multiple layers
        to process data and make predictions.
        """,
        """
        Climate change refers to long-term shifts in global temperatures and
        weather patterns. Greenhouse gases such as carbon dioxide and methane
        trap heat in the atmosphere, causing global warming. Renewable energy
        sources like solar and wind power can help reduce greenhouse gas
        emissions and mitigate climate change.
        """,
        """
        The Renaissance was a period of cultural, artistic, and intellectual
        rebirth in Europe from the 14th to 17th centuries. Artists like
        Leonardo da Vinci and Michelangelo created masterpieces that continue
        to influence art today. The period also saw advances in science,
        literature, and philosophy.
        """,
        """
        Quantum computing is a type of computation that uses quantum mechanical
        phenomena such as superposition and entanglement. Unlike classical
        computers that use bits, quantum computers use quantum bits or qubits.
        This technology has the potential to solve certain problems much faster
        than classical computers.
        """,
    ]

    println("\n📄 Processing $(length(wikipedia_texts)) Wikipedia-style texts...\n")

    # Process each text using domain system
    all_results = Vector{Dict{String, Any}}()

    options = ProcessingOptions(domain="wikipedia")

    for (i, text) in enumerate(wikipedia_texts)
        println("📖 Processing text $i/$(length(wikipedia_texts))...")
        println("   Text preview: $(strip(text[1:min(100, length(text))]))...\n")

        # Extract entities using Wikipedia domain
        entities = extract_entities(wiki_domain, text, options)

        # Extract relations using Wikipedia domain
        relations = extract_relations(wiki_domain, entities, text, options)

        # Store results
        result = Dict(
            "text_id" => i,
            "text_preview" => strip(text[1:min(100, length(text))]),
            "entities" => entities,
            "relations" => relations,
            "entity_count" => length(entities),
            "relation_count" => length(relations),
        )

        push!(all_results, result)

        # Display results for this text
        println("   Found $(length(entities)) entities and $(length(relations)) relations")

        # Show top entities
        if !isempty(entities)
            println("   Top entities:")
            for (j, entity) in enumerate(entities[1:min(5, length(entities))])
                println("     $j. \"$(entity.text)\" -> $(entity.entity_type) ($(round(entity.confidence, digits=2)))")
            end
        end

        # Show sample relations
        if !isempty(relations)
            println("   Sample relations:")
            for (j, relation) in enumerate(relations[1:min(3, length(relations))])
                head_text = get_entity_text(entities, relation.head)
                tail_text = get_entity_text(entities, relation.tail)
                println("     $j. \"$head_text\" --[$(relation.relation_type)]--> \"$tail_text\" ($(round(relation.confidence, digits=2)))")
            end
        end
        println()
    end

    # Aggregate results
    println("\n📊 Aggregated Results:")
    total_entities = sum(r["entity_count"] for r in all_results)
    total_relations = sum(r["relation_count"] for r in all_results)

    println("   Total entities across all texts: $total_entities")
    println("   Total relations across all texts: $total_relations")
    println("   Average entities per text: $(round(total_entities / length(wikipedia_texts), digits=1))")
    println("   Average relations per text: $(round(total_relations / length(wikipedia_texts), digits=1))")

    # Analyze entity types across all texts
    println("\n🏷️  Entity Type Distribution (All Texts):")
    all_entity_types = Dict{String, Int}()
    for result in all_results
        for entity in result["entities"]
            type_name = entity.entity_type
            all_entity_types[type_name] = get(all_entity_types, type_name, 0) + 1
        end
    end

    for (type_name, count) in sort(collect(all_entity_types), by = x->x[2], rev = true)
        println("   $type_name: $count")
    end

    # Analyze relation types
    println("\n🔗 Relation Type Distribution (All Texts):")
    all_relation_types = Dict{String, Int}()
    for result in all_results
        for relation in result["relations"]
            rel_type = relation.relation_type
            all_relation_types[rel_type] = get(all_relation_types, rel_type, 0) + 1
        end
    end

    for (rel_type, count) in sort(collect(all_relation_types), by = x->x[2], rev = true)
        println("   $rel_type: $count")
    end

    # Calculate confidence statistics
    all_confidences = Float64[]
    for result in all_results
        for entity in result["entities"]
            push!(all_confidences, entity.confidence)
        end
    end

    if !isempty(all_confidences)
        avg_confidence = mean(all_confidences)
        max_confidence = maximum(all_confidences)
        min_confidence = minimum(all_confidences)

        println("\n📊 Confidence Statistics:")
        println("   Average: $(round(avg_confidence, digits=3))")
        println("   Maximum: $(round(max_confidence, digits=3))")
        println("   Minimum: $(round(min_confidence, digits=3))")
    end

    println("\n" * "="^60)
    println("✅ Wikipedia Example 1 completed successfully!")
    println("   Domain system allows easy switching between biomedical and Wikipedia domains")
    println("="^60)
end

# Helper function to get entity text by ID
function get_entity_text(entities::Vector{GraphMERT.Entity}, entity_id::String)
    for entity in entities
        if entity.id == entity_id
            return entity.text
        end
    end
    return "Unknown"
end

# Run the example
main()
