"""
GraphMERT Visualization Module

Provides comprehensive visualization capabilities for knowledge graphs extracted by GraphMERT,
including static and interactive visualizations using various Julia plotting libraries.
"""

module Visualization

using ..GraphMERT: KnowledgeGraph, KnowledgeEntity, KnowledgeRelation
using ..GraphMERT: Entity, Relation

# Import core dependencies
using Graphs
using MetaGraphs
using Colors
using Statistics

# Check availability of visualization packages
let
    global GRAPHRECIPES_AVAILABLE = false
    global PLOTS_AVAILABLE = false
    global PLOTLYJS_AVAILABLE = false

    try
        using GraphRecipes
        using Plots
        GRAPHRECIPES_AVAILABLE = true
        PLOTS_AVAILABLE = true
    catch
        # Packages not available
    end

    try
        using PlotlyJS
        PLOTLYJS_AVAILABLE = true
    catch
        # Package not available
    end
end

# Include submodules
include("graphs.jl")
include("layouts.jl")
include("static.jl")
include("interactive.jl")
include("styles.jl")
include("utils.jl")

# Export main API functions
export kg_to_graphs_format, visualize_graph, plot_knowledge_graph
export filter_by_confidence, filter_by_entity_type
export export_graph

end # module Visualization
