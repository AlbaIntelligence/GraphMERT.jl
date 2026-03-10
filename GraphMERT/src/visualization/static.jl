"""
Static visualization functions for GraphMERT knowledge graphs.

Provides functions for creating static plots of knowledge graphs using GraphRecipes.jl.
"""

using ..GraphMERT: KnowledgeGraph
using Graphs
using Colors

# For file path handling
using Base: splitext

"""
    visualize_graph(kg::KnowledgeGraph;
                   layout=:spring,
                   node_size=:degree,
                   edge_width=:confidence,
                   node_color=:entity_type,
                   edge_color=:relation_type,
                   show_labels=true,
                   title="Knowledge Graph",
                   kwargs...)

Create a static visualization of a knowledge graph using GraphRecipes.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to visualize
- `layout`: Layout algorithm (:spring, :circular, :random, :stress, :shell)
- `node_size`: Node sizing method (:fixed, :degree, :confidence)
- `edge_width`: Edge width method (:fixed, :confidence)
- `node_color`: Node coloring method (:fixed, :entity_type, :confidence)
- `edge_color`: Edge coloring method (:fixed, :relation_type, :confidence)
- `show_labels`: Whether to show node labels
- `title`: Plot title
- `kwargs...`: Additional arguments passed to graphplot

# Returns
- Plot object that can be displayed or saved

# Examples
```julia
using GraphMERT

# Create and visualize a knowledge graph
kg = extract_knowledge_graph("Alice knows Bob who works at Google.")
p = visualize_graph(kg, layout=:spring, node_color=:entity_type)
display(p)
```
"""
function visualize_graph(kg::KnowledgeGraph;
                        layout=:spring,
                        node_size=:degree,
                        edge_width=:confidence,
                        node_color=:entity_type,
                        edge_color=:relation_type,
                        show_labels=true,
                        title="Knowledge Graph",
                        kwargs...)

    if !GRAPHRECIPES_AVAILABLE || !PLOTS_AVAILABLE
        error("GraphRecipes.jl and Plots.jl are required for static visualization. Install them with: using Pkg; Pkg.add([\"GraphRecipes\", \"Plots\"])")
    end

    # Convert to MetaGraph format
    mg = kg_to_graphs_format(kg, validate=false)

    # Get node labels
    node_labels = show_labels ? [props(mg, v)[:text] for v in vertices(mg)] : nothing

    # Get edge labels (relation types)
    edge_labels = String[]
    for e in edges(mg)
    rel_meta = props(mg, e)
    push!(edge_labels, rel_meta[:relation_type])
    end

    # Node sizes
    node_sizes = get_node_sizes_static(mg, node_size)

    # Edge widths
    edge_widths = get_edge_widths_static(mg, edge_width)

    # Node colors
    node_colors = get_node_colors_static(mg, node_color)

    # Edge colors
    edge_colors = get_edge_colors_static(mg, edge_color)

    # Handle custom layouts if needed
    if layout in [:hierarchical, :bipartite] && NETWORKLAYOUT_AVAILABLE
    # Use custom layout computation for unsupported GraphRecipes layouts
    positions = compute_layout(mg, layout)
    layout_positions = positions
    else
    layout_positions = layout
    end

    # Create the plot
    p = graphplot(mg,
                  layout=layout_positions,
                  nodecolor=node_colors,
                  edgecolor=edge_colors,
                  nodesize=node_sizes,
                  edgewidth=edge_widths,
                  nodelabel=node_labels,
                  edgelabel=edge_labels,
                  title=title,
                  kwargs...)

    return p
end

"""
    plot_knowledge_graph(kg::KnowledgeGraph; kwargs...)

Alias for visualize_graph for backward compatibility.
"""
plot_knowledge_graph(kg::KnowledgeGraph; kwargs...) = visualize_graph(kg; kwargs...)

# Helper functions for static visualization

function get_node_sizes_static(mg::MetaGraph, method::Symbol)
    if method == :fixed
        return fill(0.1, nv(mg))
    elseif method == :degree
        degrees = [degree(mg, v) for v in vertices(mg)]
        max_degree = maximum(degrees)
        return [max(0.05, 0.1 + 0.15 * (d / max_degree)) for d in degrees]
    elseif method == :confidence
        confidences = [props(mg, v)[:confidence] for v in vertices(mg)]
        return [max(0.05, 0.1 + 0.15 * c) for c in confidences]
    else
        return fill(0.1, nv(mg))
    end
end

function get_edge_widths_static(mg::MetaGraph, method::Symbol)
    if method == :fixed
        return fill(1, ne(mg))
    elseif method == :confidence
        confidences = [props(mg, e)[:confidence] for e in edges(mg)]
        return [max(0.5, 3.0 * c) for c in confidences]
    else
        return fill(1, ne(mg))
    end
end

function get_node_colors_static(mg::MetaGraph, method::Symbol)
    if method == :fixed
        return colorant"lightblue"
    elseif method == :entity_type
        colors = Color[]
        entity_types = unique([props(mg, v)[:entity_type] for v in vertices(mg)])
        type_colors = distinguishable_colors(length(entity_types))

        type_map = Dict(entity_types .=> type_colors)

        for v in vertices(mg)
            etype = props(mg, v)[:entity_type]
            push!(colors, type_map[etype])
        end
        return colors
    elseif method == :confidence
        confidences = [props(mg, v)[:confidence] for v in vertices(mg)]
        return [RGB(c, c, 1.0) for c in confidences]  # Blue gradient based on confidence
    else
        return colorant"lightblue"
    end
end

function get_edge_colors_static(mg::MetaGraph, method::Symbol)
    if method == :fixed
        return colorant"gray"
    elseif method == :relation_type
        colors = Color[]
        relation_types = unique([props(mg, e)[:relation_type] for e in edges(mg)])
        type_colors = distinguishable_colors(length(relation_types))

        type_map = Dict(relation_types .=> type_colors)

        for e in edges(mg)
            rtype = props(mg, e)[:relation_type]
            push!(colors, type_map[rtype])
        end
        return colors
    elseif method == :confidence
        confidences = [props(mg, e)[:confidence] for e in edges(mg)]
        return [RGB(c, 0.5, 0.5) for c in confidences]  # Red gradient based on confidence
    else
        return colorant"gray"
    end
end

"""
    export_graph(p::Plots.Plot, filename::String; format=:png, dpi=300)

Export a graph plot to file.

# Arguments
- `p`: Plot object to export (requires Plots.jl)
- `filename::String`: Output filename (without extension)
- `format`: Export format (:png, :svg, :pdf, :html, :eps, :ps)
- `dpi`: Resolution for raster formats (default: 300)
- `width`: Plot width (default: 800)
- `height`: Plot height (default: 600)

# Examples
```julia
p = visualize_graph(kg)
export_graph(p, "my_graph", format=:png)
```
"""
function export_graph(p, filename::String; format=:png, dpi=300, width=800, height=600)
if !PLOTS_AVAILABLE
error("Plots.jl is required for export functionality")
end

# Ensure filename doesn't have extension
base_filename = splitext(filename)[1]

if format == :png
savefig(p, "$(base_filename).png")
elseif format == :svg
savefig(p, "$(base_filename).svg")
elseif format == :pdf
        savefig(p, "$(base_filename).pdf")
    elseif format == :html
        # For plots that support HTML export (like PlotlyJS plots)
        try
            savefig(p, "$(base_filename).html")
        catch
            error("HTML export not supported for this plot type. Use PlotlyJS plots for HTML export.")
        end
    elseif format == :eps
        savefig(p, "$(base_filename).eps")
    elseif format == :ps
        savefig(p, "$(base_filename).ps")
    else
        error("Unsupported export format: $format. Supported formats: :png, :svg, :pdf, :html, :eps, :ps")
    end

    println("Graph exported to $(base_filename).$(format)")
end
