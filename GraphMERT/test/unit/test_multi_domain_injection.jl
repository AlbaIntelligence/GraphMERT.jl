using Test
using GraphMERT
using GraphMERT: OntologySource, SemanticTriple, SeedInjectionConfig, inject_seed_kg, retrieve_triples

# Mock Ontology Sources
struct MockUMLSSource <: OntologySource end
struct MockWikidataSource <: OntologySource end

# Implement retrieve_triples for mock sources
function GraphMERT.retrieve_triples(source::MockUMLSSource, domain::Any, entity_id::String)
    if startswith(entity_id, "C")
        return [
            SemanticTriple("aspirin", "C0004057", "treats", "pain", [101, 200, 102], 0.9, "UMLS"),
            SemanticTriple("aspirin", "C0004057", "is_a", "drug", [101, 201, 102], 0.8, "UMLS")
        ]
    end
    return SemanticTriple[]
end

function GraphMERT.retrieve_triples(source::MockWikidataSource, domain::Any, entity_id::String)
    if startswith(entity_id, "Q")
        return [
            SemanticTriple("aspirin", "Q18216", "instance of", "medication", [101, 300, 102], 0.95, "Wikidata"),
            SemanticTriple("aspirin", "Q18216", "discovered by", "Felix Hoffmann", [101, 301, 102], 0.7, "Wikidata")
        ]
    end
    return SemanticTriple[]
end

# Mock Domain Provider
struct MockDomain <: DomainProvider end
GraphMERT.get_domain_name(::MockDomain) = "mock_domain"
GraphMERT.extract_entities(::MockDomain, text::String, options::ProcessingOptions) = [
    KnowledgeEntity("e1", "aspirin", "Chemical", 1.0, TextPosition(0, 7, 1, 1))
]
GraphMERT.link_entity(::MockDomain, text::String, config::SeedInjectionConfig) = Dict(
    :candidates => [
        Dict(:kb_id => "C0004057", :name => "aspirin", :score => 1.0, :source => "UMLS"),
        Dict(:kb_id => "Q18216", :name => "aspirin", :score => 1.0, :source => "Wikidata")
    ]
)

@testset "Multi-domain Seed Injection" begin
    # Test with UMLS source
    config_umls = SeedInjectionConfig(
        ontology_source = MockUMLSSource(),
        injection_ratio = 1.0,
        use_contextual_filtering = false
    )
    
    sequences = ["Aspirin is used to treat pain."]
    injected_umls = inject_seed_kg(sequences, SemanticTriple[], config_umls, MockDomain())
    
    @test length(injected_umls) == 1
    @test length(injected_umls[1][2]) == 2
    @test injected_umls[1][2][1].source == "UMLS"
    @test injected_umls[1][2][1].relation == "treats"

    # Test with Wikidata source
    config_wiki = SeedInjectionConfig(
        ontology_source = MockWikidataSource(),
        injection_ratio = 1.0,
        use_contextual_filtering = false
    )
    
    injected_wiki = inject_seed_kg(sequences, SemanticTriple[], config_wiki, MockDomain())
    
    @test length(injected_wiki) == 1
    @test length(injected_wiki[1][2]) == 2
    @test injected_wiki[1][2][1].source == "Wikidata"
    @test injected_wiki[1][2][1].relation == "instance of"
end
