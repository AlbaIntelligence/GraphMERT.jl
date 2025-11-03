# KnowledgeGraphRecipe.jl

Plotting recipes for knowledge graphs built with GraphMERT.jl.

## Features

- **Basic Plotting**: Static visualization of knowledge graphs
- **Filtering**: Select subsets by entity types, confidence, etc.
- **Domain Styling**: Biomedical and Wikipedia-specific color schemes
- **2D Layouts**: Spring and circular layouts
- **Export**: PNG, SVG, PDF export capabilities

## Installation

This package should be installed alongside GraphMERT.jl:

```julia
] add https://github.com/your-repo/KnowledgeGraphRecipe.jl
```

## Usage

```julia
using KnowledgeGraphRecipe
using GraphMERT

# Load or create knowledge graph
kg = load_knowledge_graph("path/to/kg.json")

# Basic plot
plot_knowledge_graph(kg)

# Filtered plot with biomedical styling
plot_knowledge_graph(kg, domain=:biomedical,
                    filter=kg -> filter_by_confidence(kg, 0.7))

# Export
p = plot_knowledge_graph(kg)
export_graph(p, "my_graph", format=:png)
```

## Dependencies

- GraphRecipes.jl: Graph plotting
- Plots.jl: Plotting framework
- Colors.jl: Color utilities
- GraphMERT.jl: Knowledge graph types (assumed available)
