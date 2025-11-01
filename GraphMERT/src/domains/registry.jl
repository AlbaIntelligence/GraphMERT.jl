"""
Domain Registry for GraphMERT.jl

This module provides a central registry for domain implementations, allowing
domains to be registered and retrieved at runtime.
"""

using Logging

# ============================================================================
# Domain Registry
# ============================================================================

"""
    DomainRegistry

Registry for domain providers.
"""
mutable struct DomainRegistry
    domains::Dict{String, Any}  # DomainProvider
    default_domain::Union{String, Nothing}
    
    function DomainRegistry()
        new(Dict{String, Any}(), nothing)
    end
end

# Global registry instance
const DOMAIN_REGISTRY = DomainRegistry()

"""
    register_domain!(domain_name::String, provider::DomainProvider)

Register a domain provider in the global registry.

# Arguments
- `domain_name::String`: Name/identifier for the domain (e.g., "biomedical", "wikipedia")
- `provider::DomainProvider`: Domain provider instance

# Example
```julia
biomedical_domain = BiomedicalDomain()
register_domain!("biomedical", biomedical_domain)
```
"""
function register_domain!(domain_name::String, provider::Any)
    if haskey(DOMAIN_REGISTRY.domains, domain_name)
        @warn "Domain '$domain_name' already registered, overwriting"
    end
    
    DOMAIN_REGISTRY.domains[domain_name] = provider
    
    # Set as default if no default is set
    if DOMAIN_REGISTRY.default_domain === nothing
        DOMAIN_REGISTRY.default_domain = domain_name
        @info "Set '$domain_name' as default domain"
    end
    
    @info "Registered domain: $domain_name"
end

"""
    get_domain(domain_name::String)

Get a domain provider from the registry.

# Arguments
- `domain_name::String`: Name of the domain to retrieve

# Returns
- Domain provider instance or nothing if not found

# Example
```julia
biomedical_domain = get_domain("biomedical")
```
"""
function get_domain(domain_name::String)
    if !haskey(DOMAIN_REGISTRY.domains, domain_name)
        @error "Domain '$domain_name' not found. Available domains: $(keys(DOMAIN_REGISTRY.domains))"
        return nothing
    end
    
    return DOMAIN_REGISTRY.domains[domain_name]
end

"""
    list_domains()

List all registered domain names.

# Returns
- Vector of domain names
"""
function list_domains()
    return collect(keys(DOMAIN_REGISTRY.domains))
end

"""
    set_default_domain(domain_name::String)

Set the default domain for operations.

# Arguments
- `domain_name::String`: Name of the domain to set as default
"""
function set_default_domain(domain_name::String)
    if !haskey(DOMAIN_REGISTRY.domains, domain_name)
        error("Domain '$domain_name' not found. Available domains: $(keys(DOMAIN_REGISTRY.domains))")
    end
    
    DOMAIN_REGISTRY.default_domain = domain_name
    @info "Set '$domain_name' as default domain"
end

"""
    get_default_domain()

Get the default domain provider.

# Returns
- Domain provider instance or nothing if no default is set
"""
function get_default_domain()
    if DOMAIN_REGISTRY.default_domain === nothing
        available_domains = collect(keys(DOMAIN_REGISTRY.domains))
        error_msg = "No default domain set."
        if !isempty(available_domains)
            error_msg *= " Available domains: $(join(available_domains, ", "))"
            error_msg *= " Use set_default_domain() to set a default domain."
        else
            error_msg *= " No domains are registered. Register a domain first using register_domain!()."
        end
        @error error_msg
        return nothing
    end
    
    return get_domain(DOMAIN_REGISTRY.default_domain)
end

"""
    has_domain(domain_name::String)

Check if a domain is registered.

# Returns
- Bool indicating if domain is registered
"""
function has_domain(domain_name::String)
    return haskey(DOMAIN_REGISTRY.domains, domain_name)
end

# Export functions
export DomainRegistry, DOMAIN_REGISTRY
export register_domain!, get_domain, list_domains
export set_default_domain, get_default_domain, has_domain
