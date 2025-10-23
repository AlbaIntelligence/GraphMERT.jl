using Test
using GraphMERT

@testset "Evaluation Integration Test" begin
    TEST_TEXT = "Diabetes is a chronic condition that affects blood glucose levels. Metformin is a common treatment."
    
    @testset "Evaluation Metrics" begin
        println("📈 Testing evaluation metrics...")
        
        # Create test knowledge graph
        entities = discover_head_entities(TEST_TEXT)
        kg = KnowledgeGraph(entities, BiomedicalRelation[], Dict{String,Any}("source" => "test"))
        
        # Test FActScore evaluation
        factscore_result = evaluate_factscore(kg, TEST_TEXT)
        
        @test factscore_result !== nothing
        @test 0.0 <= factscore_result.factscore <= 1.0
        @test factscore_result.total_triples >= 0
        @test factscore_result.supported_triples >= 0
        @test factscore_result.supported_triples <= factscore_result.total_triples
        
        println("   ✅ FActScore evaluation successful")
        println("   • FActScore: $(round(factscore_result.factscore * 100, digits=1))%")
        println("   • Total triples: $(factscore_result.total_triples)")
        println("   • Supported triples: $(factscore_result.supported_triples)")
        
        # Test ValidityScore evaluation
        validity_result = evaluate_validity(kg)
        
        @test validity_result !== nothing
        @test 0.0 <= validity_result.validity_score <= 1.0
        @test validity_result.total_triples >= 0
        @test validity_result.valid_triples >= 0
        @test validity_result.valid_triples <= validity_result.total_triples
        
        println("   ✅ ValidityScore evaluation successful")
        println("   • ValidityScore: $(round(validity_result.validity_score * 100, digits=1))%")
        println("   • Total triples: $(validity_result.total_triples)")
        println("   • Valid triples: $(validity_result.valid_triples)")
    end
end

println("✅ Evaluation Integration Test Complete!")
