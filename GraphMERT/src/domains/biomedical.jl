"""
Biomedical Domain Module Loader

This module provides a convenient way to load and register the biomedical domain.
"""

# Load the biomedical domain module
include("domains/biomedical/domain.jl")

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
