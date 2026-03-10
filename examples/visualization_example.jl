"""
GraphMERT Visualization Example

This example demonstrates the visualization capabilities of GraphMERT.jl,
including static and interactive graph visualization of knowledge graphs.
"""

using Pkg
Pkg.activate(; temp = true)
Pkg.add(["Revise", "Logging"])
using Revise
using Logging

# Activate the main project
Pkg.develop(path = "/home/emmanuel/Sync/Development/julia/projects/_repos/GraphMERT.jl/GraphMERT")
using GraphMERT

# Configure logging
global_logger(ConsoleLogger(stderr, Logging.Info))

function main()
    println("="^70)
    println("GraphMERT Visualization Example")
    println("="^70)

    # Create a sample knowledge graph
    println("\n1. Creating Sample Knowledge Graph...")

    entities = [
        KnowledgeEntity("alice", "Alice Johnson", "PERSON", 0.95,
                       TextPosition(0, 13, 1, 1), Dict{String,Any}("occupation" => "Software Engineer")),
        KnowledgeEntity("bob", "Bob Smith", "PERSON", 0.90,
                       TextPosition(20, 29, 1, 21), Dict{String,Any}("occupation" => "Data Scientist")),
        KnowledgeEntity("google", "Google Inc.", "ORGANIZATION", 0.88,
                       TextPosition(40, 51, 1, 41), Dict{String,Any}("industry" => "Technology")),
        KnowledgeEntity("ai", "Artificial Intelligence", "CONCEPT", 0.85,
                       TextPosition(60, 82, 1, 61), Dict{String,Any}("field" => "Computer Science")),
        KnowledgeEntity("ml", "Machine Learning", "CONCEPT", 0.82,
                       TextPosition(90, 106, 1, 91), Dict{String,Any}("type" => "AI Subfield")),
        KnowledgeEntity("london", "London", "LOCATION", 0.78,
                       TextPosition(110, 116, 1, 111), Dict{String,Any}("country" => "UK"))
    ]

    relations = [
        KnowledgeRelation("alice", "bob", "colleague", 0.85),
        KnowledgeRelation("bob", "google", "employed_by", 0.80),
        KnowledgeRelation("alice", "google", "worked_at", 0.75),
        KnowledgeRelation("alice", "ai", "expertise", 0.70),
        KnowledgeRelation("bob", "ml", "specializes_in", 0.72),
        KnowledgeRelation("ml", "ai", "subfield_of", 0.90),
        KnowledgeRelation("alice", "london", "lives_in", 0.65),
        KnowledgeRelation("google", "london", "headquartered_in", 0.60)
    ]

    kg = KnowledgeGraph(entities, relations)
    println("   ✅ Created knowledge graph with $(length(kg.entities)) entities and $(length(kg.relations)) relations")

    # Test conversion to MetaGraph
    println("\n2. Testing Graph Conversion...")
    mg = kg_to_graphs_format(kg)
    println("   ✅ Converted to MetaGraph format")

    # Display graph statistics
    println("\n3. Graph Statistics:")
    stats = create_visualization_summary(kg)
    println("   • Number of entities: $(stats["num_entities"])")
    println("   • Number of relations: $(stats["num_relations"])")
    println("   • Entity types: $(join(keys(stats["entity_types"]), ", "))")
    println("   • Relation types: $(join(keys(stats["relation_types"]), ", "))")
    println("   • Recommended layout: $(stats["recommended_layout"])")
    println("   • Simplification needed: $(stats["recommended_simplification"])")

    # Test filtering
    println("\n4. Testing Graph Filtering...")

    # Filter by confidence
    high_conf_kg = filter_by_confidence(kg, 0.80)
    println("   • High confidence filter (>0.8): $(length(high_conf_kg.entities)) entities, $(length(high_conf_kg.relations)) relations")

    # Filter by entity type
    person_kg = filter_by_entity_type(kg, ["PERSON"])
    println("   • Person entities only: $(length(person_kg.entities)) entities, $(length(person_kg.relations)) relations")

    # Simplify graph
    simple_kg = simplify_graph(kg, max_nodes=4, min_confidence=0.70)
    println("   • Simplified (max 4 nodes, min conf 0.7): $(length(simple_kg.entities)) entities, $(length(simple_kg.relations)) relations")

    # Test clustering
    println("\n5. Testing Entity Clustering...")
    clusters = cluster_entities(kg, :entity_type)
    for (cluster_type, entities) in clusters
        println("   • $cluster_type: $(length(entities)) entities")
    end

    # Attempt static visualization (will show error message since packages not installed)
    println("\n6. Testing Static Visualization...")
    try
        p = visualize_graph(kg, layout=:circular, node_color=:entity_type, node_size=:confidence)
        println("   ✅ Static visualization created successfully")

        # Test export (would work if packages were installed)
        # export_graph(p, "sample_graph", format=:png)
        # println("   ✅ Graph exported as PNG")

    catch e
        println("   ℹ️  Static visualization requires GraphRecipes.jl and Plots.jl:")
        println("      $e")
        println("      Install with: using Pkg; Pkg.add([\"GraphRecipes\", \"Plots\"])")
    end

    # Attempt interactive visualization (will show error message since packages not installed)
    println("\n7. Testing Interactive Visualization...")
    try
        p_interactive = visualize_graph_interactive(kg, layout=:spring, node_color=:confidence)
        println("   ✅ Interactive visualization created successfully")

        # Test HTML export (would work if packages were installed)
        # export_graph_interactive(p_interactive, "interactive_graph")
        # println("   ✅ Interactive graph exported as HTML")

    catch e
        println("   ℹ️  Interactive visualization requires PlotlyJS.jl:")
        println("      $e")
        println("      Install with: using Pkg; Pkg.add(\"PlotlyJS\")")
    end

    println("\n" * "="^70)
    println("✅ GraphMERT Visualization Example Completed!")
    println()
    println("To enable full visualization functionality, install:")
    println("  • GraphRecipes.jl and Plots.jl for static visualizations")
    println("  • PlotlyJS.jl for interactive visualizations")
    println()
    println("Example usage:")
    println("  using GraphMERT")
    println("  kg = extract_knowledge_graph(text)")
    println("  p = visualize_graph(kg)")
    println("  export_graph(p, \"my_graph\")")
    println("="^70)
end

# Run the example
main()
