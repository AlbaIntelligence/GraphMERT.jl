"""
Test fixtures for Wikipedia domain testing.

Provides shared test fixtures and configuration for French monarchy testing.
"""

using GraphMERT

# Test configuration
const TEST_CONF = Dict(
    :confidence_threshold => 0.5,
    :max_entities => 100,
    :max_relations => 100,
    :domain => "wikipedia"
)

"""
Create ProcessingOptions for Wikipedia testing
"""
function create_test_options(; confidence_threshold=0.5)
    return ProcessingOptions(
        domain="wikipedia",
        confidence_threshold=confidence_threshold,
        max_entities=100,
        max_relations=100
    )
end

"""
Setup function to run before each test
"""
function setup_wikipedia_tests()
    # Load Wikipedia domain
    try
        include(joinpath(@__DIR__, "..", "..", "src", "domains", "wikipedia.jl"))
        @info "Wikipedia domain loaded"
    catch e
        @warn "Could not load Wikipedia domain: $e"
    end
    
    # Register domain if not already registered
    if !GraphMERT.has_domain("wikipedia")
        try
            domain = GraphMERT.load_wikipedia_domain()
            GraphMERT.register_domain!("wikipedia", domain)
            @info "Wikipedia domain registered"
        catch e
            @warn "Could not register Wikipedia domain: $e"
        end
    end
end

"""
Get test articles
"""
function get_test_articles()
    return [
        ("louis_xiv", "Louis XIV of France", LOUIS_XIV_ARTICLE),
        ("henry_iv", "Henry IV of France", HENRY_IV_ARTICLE),
        ("marie_antoinette", "Marie Antoinette", MARIE_ANTOINETTE_ARTICLE)
    ]
end

# Include test utilities
include("test_utils.jl")
include("reference_facts.jl")
include("metrics.jl")
