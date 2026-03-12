"""
Test fixtures for Wikipedia domain testing.

Provides shared test fixtures and configuration for French monarchy testing.
"""

using GraphMERT

# Test configuration
const TEST_CONF = Dict(
    :confidence_threshold => 0.5,
    :max_length => 2048,
    :domain => "wikipedia"
)

"""
Create ProcessingOptions for Wikipedia testing
"""
function create_test_options(; confidence_threshold=0.5)
    return ProcessingOptions(
        domain="wikipedia",
        confidence_threshold=confidence_threshold,
        max_length=2048,
        batch_size=32,
        verbose=true
    )
end

"""
Setup function to run before each test
"""
function setup_wikipedia_tests()
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
