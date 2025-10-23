using Test
using GraphMERT

@testset "Knowledge Graph Integration Test" begin
    TEST_TEXT = "Diabetes is a chronic condition that affects blood glucose levels. Metformin is a common treatment."
    
    @testset "Knowledge Graph Construction" begin
        println("📊 Testing knowledge graph construction...")
        
        # Extract entities and relations
        entities = discover_head_entities(TEST_TEXT)
        relations = match_relations_for_entities(entities, TEST_TEXT)
        
        # Create knowledge graph with empty relations for now
        metadata = Dict{String,Any}(
            "source" => "integration_test",
            "extraction_time" => now(),
            "text_length" => length(TEST_TEXT)
        )
        
        kg = KnowledgeGraph(entities, BiomedicalRelation[], metadata)
        
        @test kg !== nothing
        @test length(kg.entities) == length(entities)
        @test length(kg.relations) == 0  # Empty relations for now
        @test haskey(kg.metadata, "source")
        @test kg.metadata["source"] == "integration_test"
        
        println("   ✅ Knowledge graph constructed successfully")
        println("   • Entities: $(length(kg.entities))")
        println("   • Relations: $(length(kg.relations))")
        println("   • Metadata keys: $(length(kg.metadata))")
    end
end

println("✅ Knowledge Graph Integration Test Complete!")
