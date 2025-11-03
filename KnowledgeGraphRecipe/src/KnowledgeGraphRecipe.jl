module KnowledgeGraphRecipe

using GraphRecipes
using Plots
using Colors

# Export main functions
export plot_knowledge_graph, plot_knowledge_graph_filtered, export_graph
export filter_by_confidence, filter_by_entity_type

# Include implementations
include("recipes.jl")
include("filters.jl")
include("styles.jl")
include("layouts.jl")

end
