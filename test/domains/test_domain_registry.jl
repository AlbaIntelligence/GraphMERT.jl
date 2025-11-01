"""
Test suite for domain registry functionality
============================================

This test suite verifies that the domain registry system works correctly,
including domain registration, retrieval, listing, and default domain management.
"""

using Test
using Pkg
Pkg.activate("../../")

using GraphMERT
using Logging

# Configure logging for tests
# Use global logger to suppress verbose output
global_logger(Logging.ConsoleLogger(stderr, Logging.Warn))

@testset "Domain Registry Tests" begin
    @testset "Registry Initialization" begin
        # Test that global registry exists
        @test DOMAIN_REGISTRY isa DomainRegistry
        @test isempty(DOMAIN_REGISTRY.domains)
        
        # Test registry structure
        @test hasfield(DomainRegistry, :domains)
        @test hasfield(DomainRegistry, :default_domain)
    end
    
    @testset "Domain Registration" begin
        # Load domains
        include("../../GraphMERT/src/domains/biomedical.jl")
        include("../../GraphMERT/src/domains/wikipedia.jl")
        
        # Create domain instances
        bio_domain = load_biomedical_domain()
        wiki_domain = load_wikipedia_domain()
        
        # Test registration
        register_domain!("biomedical", bio_domain)
        register_domain!("wikipedia", wiki_domain)
        
        # Verify domains are registered
        @test has_domain("biomedical")
        @test has_domain("wikipedia")
        @test !has_domain("nonexistent")
        
        # Test domain listing
        domains = list_domains()
        @test "biomedical" in domains
        @test "wikipedia" in domains
        @test length(domains) >= 2
        
        # Test domain retrieval
        retrieved_bio = get_domain("biomedical")
        @test retrieved_bio !== nothing
        @test retrieved_bio isa DomainProvider
        
        retrieved_wiki = get_domain("wikipedia")
        @test retrieved_wiki !== nothing
        @test retrieved_wiki isa DomainProvider
        
        # Test retrieval of non-existent domain
        nonexistent = get_domain("nonexistent")
        @test nonexistent === nothing
    end
    
    @testset "Default Domain Management" begin
        # Test that default domain is set (first registered domain)
        default_domain = get_default_domain()
        @test default_domain !== nothing
        @test default_domain isa DomainProvider
        
        # Test setting default domain
        set_default_domain("wikipedia")
        new_default = get_default_domain()
        @test new_default !== nothing
        @test get_domain_name(new_default) == "wikipedia"
        
        # Test setting invalid default domain
        @test_throws ErrorException set_default_domain("nonexistent")
        
        # Reset to biomedical
        set_default_domain("biomedical")
    end
    
    @testset "Domain Metadata" begin
        bio_domain = get_domain("biomedical")
        wiki_domain = get_domain("wikipedia")
        
        # Test domain names
        @test get_domain_name(bio_domain) == "biomedical"
        @test get_domain_name(wiki_domain) == "wikipedia"
        
        # Test domain configs
        bio_config = get_domain_config(bio_domain)
        wiki_config = get_domain_config(wiki_domain)
        
        @test bio_config isa DomainConfig
        @test wiki_config isa DomainConfig
        @test bio_config.name == "biomedical"
        @test wiki_config.name == "wikipedia"
        
        # Test entity types registration
        bio_entity_types = register_entity_types(bio_domain)
        wiki_entity_types = register_entity_types(wiki_domain)
        
        @test isa(bio_entity_types, Dict)
        @test isa(wiki_entity_types, Dict)
        @test length(bio_entity_types) > 0
        @test length(wiki_entity_types) > 0
        
        # Test relation types registration
        bio_relation_types = register_relation_types(bio_domain)
        wiki_relation_types = register_relation_types(wiki_domain)
        
        @test isa(bio_relation_types, Dict)
        @test isa(wiki_relation_types, Dict)
        @test length(bio_relation_types) > 0
        @test length(wiki_relation_types) > 0
    end
    
    @testset "Domain Overwriting" begin
        # Test that overwriting a domain warns but succeeds
        bio_domain = get_domain("biomedical")
        register_domain!("biomedical", bio_domain)  # Should warn but succeed
        
        @test has_domain("biomedical")
    end
end

println("✅ All domain registry tests passed!")
