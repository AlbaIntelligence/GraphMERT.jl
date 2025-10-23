using Test
using GraphMERT
using Statistics: mean

@testset "Entity Extraction Integration Test" begin
    TEST_TEXT = "Diabetes is a chronic condition that affects blood glucose levels. Metformin is a common treatment."
    
    @testset "Entity Extraction" begin
        println("🔍 Testing entity extraction...")
        
        entities = discover_head_entities(TEST_TEXT)
        
        @test length(entities) > 0
        @test all(e isa BiomedicalEntity for e in entities)
        @test all(0.0 <= e.confidence <= 1.0 for e in entities)
        
        # Verify entity properties
        for entity in entities
            @test !isempty(entity.text)
            @test !isempty(entity.label)
            @test entity.position.start >= 1
            @test entity.position.stop >= entity.position.start
        end
        
        println("   ✅ Entity extraction successful")
        println("   • Entities found: $(length(entities))")
        println("   • Average confidence: $(round(mean([e.confidence for e in entities]), digits=3))")
    end
end

println("✅ Entity Extraction Integration Test Complete!")
