"""
Biomedical Domain Module Loader

This module provides a convenient way to load and register the biomedical domain.
This file should be included from within the GraphMERT module context.
"""

# Load the biomedical domain module
# Path is relative to this file: domains/biomedical.jl -> biomedical/domain.jl
include("biomedical/domain.jl")

# Load optional biomedical domain modules
# These are loaded on-demand when needed, but can be included here for convenience
# include("biomedical/graph.jl")      # Biomedical knowledge graph structures
# include("biomedical/evaluation.jl") # Diabetes evaluation benchmarks
# include("biomedical/pubmed.jl")     # PubMed text processing

# Note: These modules can be included explicitly when needed:
# include("GraphMERT/src/domains/biomedical/graph.jl")
# include("GraphMERT/src/domains/biomedical/evaluation.jl")
# include("GraphMERT/src/domains/biomedical/pubmed.jl")

"""
    load_biomedical_domain(umls_client::Union{Any, Nothing} = nothing)

Load and register the biomedical domain.

# Arguments
- `umls_client::Union{Any, Nothing}`: Optional UMLS client for entity linking

# Returns
- `BiomedicalDomain`: The created domain instance

# Example
```julia
using GraphMERT
using GraphMERT.Domains.Biomedical

# Load domain
domain = load_biomedical_domain()
register_domain!("biomedical", domain)
```
"""
function load_biomedical_domain(umls_client::Union{Any, Nothing} = nothing)
    return BiomedicalDomain(umls_client)
end

# Export
export BiomedicalDomain, load_biomedical_domain
