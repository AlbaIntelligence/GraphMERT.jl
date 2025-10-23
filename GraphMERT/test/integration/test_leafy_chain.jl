using Test
using GraphMERT

@testset "Leafy Chain Integration Test" begin
    TEST_TEXT = "Diabetes is a chronic condition that affects blood glucose levels."
    
    @testset "Leafy Chain Creation" begin
        println("🧬 Testing leafy chain graph creation...")
        
        graph = create_leafy_chain_from_text(TEST_TEXT)
        
        @test graph !== nothing
        @test length(graph.root_nodes) > 0
        @test length(graph.leaf_nodes) > 0
        
        println("   ✅ Leafy chain graph created successfully")
        println("   • Roots: $(length(graph.root_nodes))")
        println("   • Leaves: $(length(graph.leaf_nodes))")
        println("   • Total nodes: $(length(graph.root_nodes) + length(graph.leaf_nodes))")
    end
end

println("✅ Leafy Chain Integration Test Complete!")
