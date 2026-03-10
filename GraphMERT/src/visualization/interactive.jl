"""
Interactive visualization functions for GraphMERT knowledge graphs.

Provides functions for creating interactive plots of knowledge graphs using PlotlyJS.jl.
"""

using ..GraphMERT: KnowledgeGraph
using Graphs

# PlotlyJS availability is checked in parent module

"""
    visualize_graph_interactive(kg::KnowledgeGraph;
                               layout=:spring,
                               node_size=:degree,
                               edge_width=:confidence,
                               node_color=:entity_type,
                               edge_color=:relation_type,
                               title="Interactive Knowledge Graph")

Create an interactive visualization of a knowledge graph using PlotlyJS.

Requires PlotlyJS.jl to be installed.

# Arguments
- `kg::KnowledgeGraph`: Knowledge graph to visualize
- `layout`: Layout algorithm (:spring, :circular, :random)
- `node_size`: Node sizing method (:fixed, :degree, :confidence)
- `edge_width`: Edge width method (:fixed, :confidence)
- `node_color`: Node coloring method (:fixed, :entity_type, :confidence)
- `edge_color`: Edge coloring method (:fixed, :relation_type, :confidence)
- `title`: Plot title

# Returns
- Interactive PlotlyJS plot object

# Examples
```julia
using GraphMERT, PlotlyJS

# Create and visualize a knowledge graph interactively
kg = extract_knowledge_graph("Alice knows Bob who works at Google.")
p = visualize_graph_interactive(kg, layout=:spring)
display(p)
```
"""
function visualize_graph_interactive(kg::KnowledgeGraph;
                                    layout=:spring,
                                    node_size=:degree,
                                    edge_width=:confidence,
                                    node_color=:entity_type,
                                    edge_color=:relation_type,
                                    title="Interactive Knowledge Graph")

    if !PLOTLYJS_AVAILABLE
        error("PlotlyJS.jl is required for interactive visualization. Install it with: using Pkg; Pkg.add(\"PlotlyJS\")")
    end

    # Convert to MetaGraph format
    mg = kg_to_graphs_format(kg, validate=false)

    # Get layout positions
    positions = compute_layout_positions(mg, layout)

    # Create node traces
    node_traces = create_node_traces(mg, positions, node_size, node_color)

    # Create edge traces
    edge_traces = create_edge_traces(mg, positions, edge_width, edge_color)

    # Combine traces
    all_traces = vcat(edge_traces, node_traces)

    # Create layout
    plot_layout = Layout(
        title=title,
        showlegend=true,
        hovermode="closest",
        xaxis=attr(showgrid=false, zeroline=false, showticklabels=false),
        yaxis=attr(showgrid=false, zeroline=false, showticklabels=false),
        margin=attr(l=0, r=0, t=50, b=0)
    )

    return Plot(all_traces, plot_layout)
end

# Helper functions for interactive visualization

function compute_layout_positions(mg::MetaGraph, layout_type::Symbol)
    # Use NetworkLayout.jl or simple circular layout for positioning
    n = nv(mg)

    if layout_type == :circular
        angles = range(0, 2π, length=n+1)[1:end-1]
        x = cos.(angles)
        y = sin.(angles)
    elseif layout_type == :random
        x = rand(n) .* 2 .- 1
        y = rand(n) .* 2 .- 1
    else  # default to spring layout approximation
        # Simple spring layout approximation
        x = randn(n) .* 0.5
        y = randn(n) .* 0.5
    end

    return [(x[i], y[i]) for i in 1:n]
end

function create_node_traces(mg::MetaGraph, positions, node_size, node_color)
    traces = []

    # Group nodes by color for better performance
    if node_color == :entity_type
        entity_types = unique([props(mg, v)["entity_type"] for v in vertices(mg)])
        for etype in entity_types
            type_nodes = [v for v in vertices(mg) if props(mg, v)["entity_type"] == etype]
            if isempty(type_nodes)
                continue
            end

            x_coords = [positions[v][1] for v in type_nodes]
            y_coords = [positions[v][2] for v in type_nodes]

            sizes = get_node_sizes_interactive(mg, type_nodes, node_size)
            colors = get_node_colors_interactive(mg, type_nodes, node_color)
            texts = [get_node_hover_text(mg, v) for v in type_nodes]
            labels = [props(mg, v)["text"] for v in type_nodes]

            trace = scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                marker=attr(
                    size=sizes,
                    color=colors,
                    sizemode="diameter",
                    line=attr(width=2, color="white")
                ),
                text=labels,
                textposition="top center",
                hovertext=texts,
                hoverinfo="text",
                name=etype
            )
            push!(traces, trace)
        end
    else
        # Single trace for all nodes
        x_coords = [positions[v][1] for v in vertices(mg)]
        y_coords = [positions[v][2] for v in vertices(mg)]

        sizes = get_node_sizes_interactive(mg, vertices(mg), node_size)
        colors = get_node_colors_interactive(mg, vertices(mg), node_color)
        texts = [get_node_hover_text(mg, v) for v in vertices(mg)]
        labels = [props(mg, v)["text"] for v in vertices(mg)]

        trace = scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            marker=attr(
                size=sizes,
                color=colors,
                sizemode="diameter",
                line=attr(width=2, color="white")
            ),
            text=labels,
            textposition="top center",
            hovertext=texts,
            hoverinfo="text",
            name="Entities"
        )
        push!(traces, trace)
    end

    return traces
end

function create_edge_traces(mg::MetaGraph, positions, edge_width, edge_color)
    traces = []

    for e in edges(mg)
        src, dst = src(e), dst(e)
        x0, y0 = positions[src]
        x1, y1 = positions[dst]

        # Create line trace for edge
        width = get_edge_width_interactive(mg, e, edge_width)
        color = get_edge_color_interactive(mg, e, edge_color)
        hover_text = get_edge_hover_text(mg, e)

        trace = scatter(
            x=[x0, x1, nothing],  # nothing creates break for multiple edges
            y=[y0, y1, nothing],
            mode="lines",
            line=attr(width=width, color=color),
            hoverinfo="text",
            hovertext=hover_text,
            showlegend=false
        )
        push!(traces, trace)
    end

    return traces
end

# Helper functions for interactive visualization styling

function get_node_sizes_interactive(mg::MetaGraph, nodes, method::Symbol)
    if method == :fixed
        return fill(20, length(nodes))
    elseif method == :degree
        degrees = [degree(mg, v) for v in nodes]
        max_degree = maximum(degrees)
        return [max(10, 20 + 30 * (d / max_degree)) for d in degrees]
    elseif method == :confidence
        confidences = [props(mg, v)["confidence"] for v in nodes]
        return [max(10, 20 + 30 * c) for c in confidences]
    else
        return fill(20, length(nodes))
    end
end

function get_node_colors_interactive(mg::MetaGraph, nodes, method::Symbol)
    if method == :fixed
        return fill("lightblue", length(nodes))
    elseif method == :entity_type
        return [get_entity_type_color(props(mg, v)["entity_type"]) for v in nodes]
    elseif method == :confidence
        confidences = [props(mg, v)["confidence"] for v in nodes]
        return ["rgba(0, 0, 255, $(c))" for c in confidences]
    else
        return fill("lightblue", length(nodes))
    end
end

function get_edge_width_interactive(mg::MetaGraph, edge, method::Symbol)
    if method == :fixed
        return 2
    elseif method == :confidence
        confidence = props(mg, edge)["confidence"]
        return max(1, 5 * confidence)
    else
        return 2
    end
end

function get_edge_color_interactive(mg::MetaGraph, edge, method::Symbol)
    if method == :fixed
        return "gray"
    elseif method == :relation_type
        rtype = props(mg, edge)["relation_type"]
        return get_relation_type_color(rtype)
    elseif method == :confidence
        confidence = props(mg, edge)["confidence"]
        return "rgba(255, 0, 0, $(confidence))"
    else
        return "gray"
    end
end

function get_node_hover_text(mg::MetaGraph, v)
    props_dict = props(mg, v)
    return """
    <b>$(props_dict["text"])</b><br>
    Type: $(props_dict["entity_type"])<br>
    Confidence: $(round(props_dict["confidence"], digits=3))<br>
    ID: $(props_dict["id"])
    """
end

function get_edge_hover_text(mg::MetaGraph, e)
    props_dict = props(mg, e)
    return """
    <b>$(props_dict["relation_type"])</b><br>
    From: $(props_dict["head_id"])<br>
    To: $(props_dict["tail_id"])<br>
    Confidence: $(round(props_dict["confidence"], digits=3))
    """
end

# Color mapping functions
function get_entity_type_color(entity_type::String)
    colors = Dict(
        "PERSON" => "lightblue",
        "ORGANIZATION" => "lightgreen",
        "LOCATION" => "lightcoral",
        "DISEASE" => "red",
        "DRUG" => "blue",
        "GENE" => "purple",
        "PROTEIN" => "orange"
    )
    return get(colors, entity_type, "gray")
end

function get_relation_type_color(relation_type::String)
    colors = Dict(
        "knows" => "blue",
        "works_at" => "green",
        "treats" => "red",
        "causes" => "darkred",
        "interacts_with" => "purple",
        "related_to" => "orange"
    )
    return get(colors, relation_type, "gray")
end

"""
    export_graph_interactive(p::PlotlyJS.Plot, filename::String)

Export an interactive PlotlyJS plot to HTML file.

# Arguments
- `p::PlotlyJS.Plot`: Interactive plot object
- `filename::String`: Output filename (without extension)

# Examples
```julia
p = visualize_graph_interactive(kg)
export_graph_interactive(p, "interactive_graph")
```
"""
function export_graph_interactive(p, filename::String)
    if !PLOTLYJS_AVAILABLE
        error("PlotlyJS.jl is required for interactive export")
    end

    savefig(p, "$filename.html")
end
