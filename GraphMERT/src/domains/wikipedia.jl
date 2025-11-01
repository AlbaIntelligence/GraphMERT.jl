"""
Wikipedia Domain Module Loader

This module provides a convenient way to load and register the Wikipedia domain.
"""

# Load the Wikipedia domain module
include("domains/wikipedia/domain.jl")

"""
    load_wikipedia_domain(wikidata_client::Union{Any, Nothing} = nothing)

Load and register the Wikipedia domain.

# Arguments
- `wikidata_client::Union{Any, Nothing}`: Optional Wikidata client for entity linking

# Returns
- `WikipediaDomain`: The created domain instance

# Example
```julia
using GraphMERT
using GraphMERT.Domains.Wikipedia

# Load domain
domain = load_wikipedia_domain()
register_domain!("wikipedia", domain)
```
"""
function load_wikipedia_domain(wikidata_client::Union{Any, Nothing} = nothing)
    return WikipediaDomain(wikidata_client)
end

# Export
export WikipediaDomain, load_wikipedia_domain
