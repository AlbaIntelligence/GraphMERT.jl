# Graph Visualization Research

## Research Summary

This document summarizes research on graph visualization libraries for knowledge graphs, focusing on Julia options and reference implementations from other languages.

## Julia Graph Visualization Libraries

### 1. **GraphRecipes.jl** (Plots.jl ecosystem)
- **Status**: Active, mature
- **Capabilities**:
  - Works with Plots.jl backend (GR, PlotlyJS, PyPlot, etc.)
  - Supports various graph types (directed, undirected, weighted)
  - Multiple layout algorithms (spring, circular, force-directed)
  - Interactive visualization with PlotlyJS backend
  - Customizable node/edge styling
- **UI Interface**: Static plots or interactive PlotlyJS widgets
- **Pros**: 
  - Easy integration with existing Plots.jl code
  - Multiple backend options
  - Good for static/exported visualizations
- **Cons**: 
  - Limited interactivity compared to web-based solutions
  - Performance may degrade with very large graphs (>1000 nodes)

### 2. **Graphviz.jl** (Graphviz wrapper)
- **Status**: Active
- **Capabilities**:
  - Wrapper for Graphviz (DOT language)
  - Professional graph layout algorithms (dot, neato, fdp, sfdp, twopi, circo)
  - High-quality static renderings
  - Export to SVG, PNG, PDF
- **UI Interface**: Static images (SVG/PNG/PDF)
- **Pros**:
  - Excellent layout algorithms
  - Industry-standard tool
  - Great for documentation/publications
- **Cons**:
  - No interactivity
  - Requires Graphviz system dependency
  - Less flexible styling

### 3. **Makie.jl / CairoMakie.jl** (Custom rendering)
- **Status**: Active, modern
- **Capabilities**:
  - Custom GPU-accelerated rendering
  - Interactive plots with GLMakie
  - Can build custom graph visualization widgets
  - Real-time updates
- **UI Interface**: Interactive GLMakie windows or static CairoMakie images
- **Pros**:
  - Highly customizable
  - GPU acceleration for large graphs
  - Modern, active development
- **Cons**:
  - Requires custom implementation
  - Steeper learning curve
  - Web export requires additional work

### 4. **NetworkLayout.jl** (Layout algorithms)
- **Status**: Active
- **Capabilities**:
  - Pure Julia layout algorithms
  - Multiple algorithms (spring, stress, spectral, etc.)
  - Provides coordinates for nodes
- **UI Interface**: Layout coordinates only (must use with other visualization)
- **Pros**:
  - Pure Julia (no external dependencies)
  - Good layout quality
- **Cons**:
  - No visualization itself, only layout
  - Must be combined with other tools

### 5. **PlotlyJS.jl** (Interactive web visualizations)
- **Status**: Active
- **Capabilities**:
  - Interactive web-based visualizations
  - Zoom, pan, hover tooltips
  - Export to HTML
  - Works well with GraphRecipes
- **UI Interface**: Interactive HTML widgets/browser
- **Pros**:
  - Interactive exploration
  - Exportable HTML files
  - Professional appearance
- **Cons**:
  - Requires browser for display
  - Large graphs may be slow

## Other Language Reference Implementations

### Python Libraries

#### 1. **NetworkX + Matplotlib**
- **Capabilities**: Static graphs, multiple layouts
- **UI**: Static plots
- **Use Case**: Analysis, publications

#### 2. **pyvis** (vis.js wrapper)
- **Capabilities**: Interactive HTML visualizations
- **UI**: Interactive web interface (HTML file)
- **Features**: 
  - Zoom, pan, drag nodes
  - Hover tooltips
  - Physics simulation
  - Node clustering
- **Use Case**: Interactive exploration, presentations

#### 3. **Plotly Network Graphs**
- **Capabilities**: Interactive web visualizations
- **UI**: Interactive Plotly widgets
- **Features**: 
  - Zoom, pan, hover
  - 3D layouts
  - Custom styling
- **Use Case**: Dashboards, interactive analysis

#### 4. **Cytoscape.js** (via python-cyto)
- **Capabilities**: Advanced graph visualization
- **UI**: Interactive web interface
- **Features**:
  - Rich styling options
  - Animations
  - Layout algorithms
  - Extensible
- **Use Case**: Complex graph visualizations

### JavaScript Libraries

#### 1. **vis.js Network**
- **Capabilities**: Interactive network visualization
- **UI**: HTML5 canvas-based interactive visualization
- **Features**:
  - Physics simulation
  - Drag nodes
  - Zoom/pan
  - Clustering
  - Custom styling
- **Use Case**: Web applications, dashboards

#### 2. **D3.js**
- **Capabilities**: Highly customizable visualizations
- **UI**: SVG-based interactive visualizations
- **Features**:
  - Complete control over rendering
  - Custom animations
  - Rich interactions
- **Use Case**: Custom web visualizations

#### 3. **Cytoscape.js**
- **Capabilities**: Professional graph visualization
- **UI**: Web-based interactive visualization
- **Features**:
  - Advanced layout algorithms
  - Rich styling
  - Extensions ecosystem
  - Performant for large graphs
- **Use Case**: Complex graph applications

#### 4. **sigma.js**
- **Capabilities**: Large graph visualization
- **UI**: WebGL-based interactive visualization
- **Features**:
  - Optimized for large graphs (10k+ nodes)
  - GPU acceleration
  - Interactive exploration
- **Use Case**: Large-scale graph visualization

## Key Visualization Features to Consider

### Essential Features
1. **Node Display**:
   - Node labels (entity names)
   - Node size (based on importance/confidence)
   - Node color (based on entity type/domain)
   - Node shape (optional, for entity types)

2. **Edge Display**:
   - Edge labels (relation types)
   - Edge thickness (based on confidence)
   - Edge color (based on relation type)
   - Directed arrows (for directed relations)

3. **Layout Algorithms**:
   - Force-directed (spring-like)
   - Hierarchical (tree-like)
   - Circular
   - Spectral clustering
   - Custom domain-specific layouts

4. **Interactivity**:
   - Zoom/pan
   - Hover tooltips (show entity/relation details)
   - Click to select/expand
   - Search/filter nodes
   - Legend for entity/relation types

### Advanced Features
1. **Filtering**:
   - Filter by entity type
   - Filter by relation type
   - Filter by confidence threshold
   - Filter by domain

2. **Clustering**:
   - Group related entities
   - Community detection visualization
   - Semantic clustering

3. **Metadata Display**:
   - Show UMLS CUI (biomedical)
   - Show Wikidata IDs (Wikipedia)
   - Show confidence scores
   - Show provenance information

4. **Export**:
   - Export to PNG/SVG (static)
   - Export to HTML (interactive)
   - Export to GraphML/Graphviz DOT

## Recommended Approach for GraphMERT.jl

### Phase 1: Static Visualization (Quick Implementation)
- **Tool**: GraphRecipes.jl + Plots.jl (GR backend)
- **Features**: 
  - Basic node/edge visualization
  - Layout algorithms
  - Static export (PNG/SVG)
- **Use Case**: Quick visualization, documentation

### Phase 2: Interactive Visualization (Enhanced UX)
- **Tool**: GraphRecipes.jl + PlotlyJS.jl backend
- **Features**:
  - Interactive zoom/pan
  - Hover tooltips
  - HTML export
- **Use Case**: Interactive exploration, presentations

### Phase 3: Advanced Visualization (Optional)
- **Tool**: Custom Makie.jl implementation OR web-based solution
- **Features**:
  - GPU acceleration for large graphs
  - Advanced filtering/clustering
  - Custom domain-specific layouts
- **Use Case**: Large-scale graphs, specialized needs

## Implementation Considerations

### Graph Data Structure
- GraphMERT uses `KnowledgeGraph` with `KnowledgeEntity` and `KnowledgeRelation`
- Need conversion to graph library format (e.g., Graphs.jl, MetaGraphs.jl)
- MetaGraphs.jl already used in biomedical domain

### Domain-Specific Customization
- Different domains may need different visualization styles
- Biomedical: Color by semantic type, show UMLS CUIs
- Wikipedia: Color by entity type, show Wikidata IDs
- Should be extensible via domain provider interface

### Performance
- Small graphs (<100 nodes): Any library works
- Medium graphs (100-1000 nodes): GraphRecipes/PlotlyJS sufficient
- Large graphs (1000+ nodes): May need GPU acceleration or web-based solution

### Integration Points
- Export function: `visualize_graph(kg::KnowledgeGraph; options...)`
- Should support domain-specific styling
- Should integrate with domain system
- Should support both static and interactive modes
