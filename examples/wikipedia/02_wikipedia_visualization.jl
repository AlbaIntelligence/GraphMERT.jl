"""
Wikipedia Example 2: Knowledge Graph Visualization

This example demonstrates visualizing knowledge graphs extracted from Wikipedia-style texts.
It builds on the entity extraction from Example 1 and creates visual representations
of the extracted knowledge.

This shows how to convert extracted entities and relations into KnowledgeGraph objects
and visualize them using GraphMERT's visualization capabilities.
"""

using Pkg
Pkg.activate(; temp = true)
Pkg.add(["Revise", "Logging"])
using Revise
using Logging

# Activate the main project
Pkg.develop(path = joinpath(@__DIR__, "..", "..", "GraphMERT"))
using GraphMERT
using GraphMERT:Entity

# Configure logging
global_logger(ConsoleLogger(stderr, Logging.Info))

function create_knowledge_graph_from_results(entities::Vector, relations::Vector, text_id::Int)
    """
    Convert extracted entities and relations into a KnowledgeGraph object.
    """
    # Convert entities to KnowledgeEntity format
    kg_entities = Vector{KnowledgeEntity}()
    for (i, entity) in enumerate(entities)
        # Create a unique ID for each entity
        entity_id = "text$(text_id)_entity$(i)"

        # Convert position if available, otherwise use defaults
        position = haskey(entity, :position) ? entity.position : TextPosition(0, 0, 1, 1)

        # Create KnowledgeEntity
        kg_entity = KnowledgeEntity(
            entity_id,
            entity.text,
            entity.label,
            entity.confidence,
            position,
            Dict{String, Any}("entity_type" => entity.entity_type),
        )
        push!(kg_entities, kg_entity)
    end

    # Convert relations to KnowledgeRelation format
    kg_relations = Vector{KnowledgeRelation}()
    for (i, relation) in enumerate(relations)
        # Find entity IDs by matching text
        head_id = ""
        tail_id = ""

        for entity in kg_entities
            if entity.text == relation.head
                head_id = entity.id
            elseif entity.text == relation.tail
                tail_id = entity.id
            end
        end

        # Skip if we can't find matching entities
        if isempty(head_id) || isempty(tail_id)
            continue
        end

        # Create KnowledgeRelation
        kg_relation = KnowledgeRelation(
            head_id,
            tail_id,
            relation.relation_type,
            relation.confidence,
        )
        push!(kg_relations, kg_relation)
    end

    return KnowledgeGraph(kg_entities, kg_relations)
end

function visualize_text_results(text_results::Dict{String, Any})
    """
    Visualize the results from a single text processing.
    """
    text_id = text_results["text_id"]
    entities = text_results["entities"]
    relations = text_results["relations"]
    text_preview = text_results["text_preview"]

    println("\n" * "="^50)
    println("📊 Visualizing Text $text_id")
    println("="^50)
    println("📝 Text: $text_preview...")
    println("📊 $(length(entities)) entities, $(length(relations)) relations")

    if isempty(entities) && isempty(relations)
        println("⚠️  No entities or relations found - skipping visualization")
        return
    end

    # Create KnowledgeGraph
    kg = create_knowledge_graph_from_results(entities, relations, text_id)

    # Display basic statistics
    println("\n📈 Knowledge Graph Statistics:")
    stats = create_visualization_summary(kg)
    println("   • Entities: $(stats["num_entities"])")
    println("   • Relations: $(stats["num_relations"])")
    println("   • Entity types: $(join(unique([e.entity_type for e in entities]), ", "))")
    println("   • Relation types: $(join(unique([r.relation_type for r in relations]), ", "))")
    println("   • Recommended layout: $(stats["recommended_layout"])")

    # Display entities
    if !isempty(kg.entities)
        println("\n🏷️  Entities:")
        for entity in kg.entities
            println("   • $(entity.text) ($(get(entity.attributes, "entity_type", "UNKNOWN")), conf: $(round(entity.confidence, digits=2))))")
        end
    end

    # Display relations
    if !isempty(kg.relations)
        println("\n🔗 Relations:")
        for relation in kg.relations
            # Find entity texts
            head_text = ""
            tail_text = ""
            for entity in kg.entities
                if entity.id == relation.head
                    head_text = entity.text
                elseif entity.id == relation.tail
                    tail_text = entity.text
                end
            end
            println("   • $head_text →[$(relation.relation_type)]→ $tail_text (conf: $(round(relation.confidence, digits=2))))")
        end
    end

    # Attempt visualization
    println("\n🎨 Creating Visualization...")
    try
        p = visualize_graph(kg,
            layout = :circular,
            node_size = :confidence,
            node_color = :entity_type,
            edge_width = :confidence,
            title = "Wikipedia Text $text_id: Knowledge Graph")

        println("   ✅ Static visualization created successfully!")

        # Export visualization
        export_filename = "wikipedia_text_$(text_id)_graph"
        try
            export_graph(p, export_filename, format = :png)
            println("   💾 Graph exported as $export_filename.png")
        catch e
            println("   ⚠️  Export failed: $e")
        end

    catch e
        println("   ℹ️  Static visualization requires GraphRecipes.jl and Plots.jl:")
        println("      $e")
        println("      Install with: using Pkg; Pkg.add([\"GraphRecipes\", \"Plots\"])")
    end

    # Attempt interactive visualization
    println("\n🎯 Creating Interactive Visualization...")
    try
        p_interactive = visualize_graph_interactive(kg,
            layout = :spring,
            node_color = :confidence,
            title = "Interactive: Wikipedia Text $text_id")

        println("   ✅ Interactive visualization created successfully!")

        # Export interactive version
        try
            export_graph_interactive(p_interactive, "wikipedia_text_$(text_id)_interactive")
            println("   💾 Interactive graph exported as wikipedia_text_$(text_id)_interactive.html")
        catch e
            println("   ⚠️  Interactive export failed: $e")
        end

    catch e
        println("   ℹ️  Interactive visualization requires PlotlyJS.jl:")
        println("      $e")
        println("      Install with: using Pkg; Pkg.add(\"PlotlyJS\")")
    end

    return kg
end

function main()
    println("="^70)
    println("GraphMERT Wikipedia Example 2: Knowledge Graph Visualization")
    println("="^70)

    # Load and register the Wikipedia domain
    println("\n1. Loading Wikipedia domain...")
    initialize_default_domains()
    set_default_domain("wikipedia")

    wiki_domain = get_domain("wikipedia")
    println("   ✅ Wikipedia domain loaded and registered")

    # Sample Wikipedia-style texts (same as Example 1)
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
    ]

    println("\n📄 Processing $(length(wikipedia_texts)) Wikipedia-style texts for visualization...")

    # Process each text using domain system (same as Example 1)
    all_results = Vector{Dict{String, Any}}()
    all_graphs = Vector{KnowledgeGraph}()

    options = ProcessingOptions(domain = "wikipedia")

    for (i, text) in enumerate(wikipedia_texts)
        println("\n📖 Processing text $i/$(length(wikipedia_texts))...")

        # Extract entities using Wikipedia domain
        entities = Base.invokelatest(extract_entities, wiki_domain, text, options)

        # Extract relations using Wikipedia domain
        relations = Base.invokelatest(extract_relations, wiki_domain, entities, text, options)

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

        # Create and visualize knowledge graph
        kg = visualize_text_results(result)
        push!(all_graphs, kg)
    end

    # Aggregate visualization results
    println("\n" * "="^70)
    println("📊 AGGREGATE VISUALIZATION RESULTS")
    println("="^70)

    total_entities = sum(length(kg.entities) for kg in all_graphs)
    total_relations = sum(length(kg.relations) for kg in all_graphs)

    println("🎯 Overall Statistics:")
    println("   • Total entities across all texts: $total_entities")
    println("   • Total relations across all texts: $total_relations")
    println("   • Average entities per text: $(round(total_entities / length(all_graphs), digits=1))")
    println("   • Average relations per text: $(round(total_relations / length(all_graphs), digits=1))")

    # Analyze entity types across all texts
    println("\n🏷️  Entity Type Distribution (All Texts):")
    all_entity_types = Dict{String, Int}()
    for result in all_results
        for entity in result["entities"]
            type_name = string(entity.entity_type)
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
            rel_type = string(relation.relation_type)
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
        for relation in result["relations"]
            push!(all_confidences, relation.confidence)
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

    println("\n" * "="^70)
    println("✅ Wikipedia Example 2 completed successfully!")
    println("   Visualization system allows exploring extracted knowledge graphs")
    println("   Both static and interactive visualizations are supported")
    println("="^70)

    println("\n📁 Generated Files:")
    println("   • PNG exports: wikipedia_text_[1-3]_graph.png")
    println("   • HTML exports: wikipedia_text_[1-3]_interactive.html")
    println("   (if GraphRecipes/Plots and PlotlyJS packages are installed)")

    println("\n🔧 To enable full visualization:")
    println("   using Pkg")
    println("   Pkg.add([\"GraphRecipes\", \"Plots\"])      # For static visualizations")
    println("   Pkg.add(\"PlotlyJS\")                     # For interactive visualizations")
end

# Helper function to get entity text by ID (from Example 1)
function get_entity_text(entities::Vector, entity_id::String)
    for entity in entities
        if entity.id == entity_id
            return entity.text
        end
    end
    return "Unknown"
end

# Run the example
main()
