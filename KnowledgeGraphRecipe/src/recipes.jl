"""
    plot_knowledge_graph(kg::KnowledgeGraph;
                        layout=:spring,
                        domain=:general,
                        filter=nothing,
                        node_size=:degree,
                        edge_width=:confidence,
                        kwargs...)

Create a static plot of a knowledge graph.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to plot
- `layout`: Layout algorithm (:spring, :circular, :random)
- `domain`: Domain for styling (:general, :biomedical, :wikipedia)
- `filter`: Optional filter function for nodes/edges
- `node_size`: Node sizing method (:fixed, :degree, :confidence)
- `edge_width`: Edge width method (:fixed, :confidence)
- `kwargs...`: Additional plotting options

# Returns
- Plot object
"""
function plot_knowledge_graph(kg::KnowledgeGraph;
                             layout=:spring,
                             domain=:general,
                             filter=nothing,
                             node_size=:degree,
                             edge_width=:confidence,
                             kwargs...)

    # Apply filtering if provided
    filtered_kg = isnothing(filter) ? kg : apply_filter(kg, filter)

    # Convert to graph format
    g, node_labels, edge_labels = kg_to_graph(filtered_kg)

    # Apply styling
    node_colors = get_node_colors(filtered_kg, domain)
    edge_colors = get_edge_colors(filtered_kg, domain)

    # Node sizes
    node_sizes = get_node_sizes(filtered_kg, node_size)

    # Edge widths
    edge_widths = get_edge_widths(filtered_kg, edge_width)

    # Create plot
    plot(g,
         layout=layout,
         nodecolor=node_colors,
         edgecolor=edge_colors,
         nodesize=node_sizes,
         edgewidth=edge_widths,
         nodelabel=node_labels,
         edgelabel=edge_labels,
         kwargs...)
end

"""
    plot_knowledge_graph_filtered(kg::KnowledgeGraph, filter_func; kwargs...)

Plot a filtered subset of the knowledge graph.

# Arguments
- `kg::KnowledgeGraph`: Full knowledge graph
- `filter_func`: Function that takes a KnowledgeGraph and returns filtered version
- `kwargs...`: Plotting options passed to plot_knowledge_graph
"""
function plot_knowledge_graph_filtered(kg::KnowledgeGraph, filter_func; kwargs...)
    filtered_kg = filter_func(kg)
    plot_knowledge_graph(filtered_kg; kwargs...)
end

"""
    export_graph(p, filename; format=:png)

Export a plot to file.

# Arguments
- `p`: Plot object
- `filename`: Output filename (without extension)
- `format`: Export format (:png, :svg, :pdf)
"""
function export_graph(p, filename; format=:png)
    if format == :png
        savefig(p, "$filename.png")
    elseif format == :svg
        savefig(p, "$filename.svg")
    elseif format == :pdf
        savefig(p, "$filename.pdf")
    end
end

# Helper function to convert KnowledgeGraph to graph format
function kg_to_graph(kg::KnowledgeGraph)
    # Convert KnowledgeGraph to adjacency matrix and labels
    nodes = [e.text for e in kg.entities]
    edges = [(findfirst(==(r.head), nodes), findfirst(==(r.tail), nodes))
             for r in kg.relations if r.head in nodes && r.tail in nodes]

    # Create simple adjacency matrix
    n = length(nodes)
    adj_matrix = falses(n, n)
    for (i, j) in edges
        adj_matrix[i, j] = true
    end

    g = Graph(adj_matrix)
    node_labels = nodes
    edge_labels = [r.relation_type for r in kg.relations]

    return g, node_labels, edge_labels
end
