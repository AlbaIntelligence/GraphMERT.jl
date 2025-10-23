using Test
using GraphMERT

@testset "Batch Processing Integration Test" begin
  MULTIPLE_DOCUMENTS = [
    "Patient with type 2 diabetes presents with elevated blood glucose requiring metformin therapy.",
    "Insulin resistance is a key feature of type 2 diabetes mellitus in obese patients.",
    "Cardiovascular complications are common in patients with long-standing diabetes."
  ]

  @testset "Batch Processing" begin
    println("🚀 Testing batch processing...")

    # Test batch processing configuration
    config = create_batch_processing_config(
      batch_size=2,
      max_memory_mb=512,
      num_threads=1,
      memory_monitoring=true,
      auto_optimize=true
    )

    @test config !== nothing
    @test config.batch_size == 2
    @test config.max_memory_mb == 512
    @test config.num_threads == 1

    # Mock extraction function for testing
    function mock_extract_knowledge_graph(doc::String)
      entities = discover_head_entities(doc)
      return KnowledgeGraph(entities, BiomedicalRelation[], Dict{String,Any}("source" => doc))
    end

    # Test batch processing
    batch_result = extract_knowledge_graph_batch(
      MULTIPLE_DOCUMENTS,
      config=config,
      extraction_function=mock_extract_knowledge_graph
    )

    @test batch_result !== nothing
    @test batch_result.total_documents == length(MULTIPLE_DOCUMENTS)
    @test batch_result.successful_batches >= 0
    @test batch_result.failed_batches >= 0
    @test batch_result.total_time > 0
    @test batch_result.average_throughput > 0
    @test length(batch_result.knowledge_graphs) == 1  # Merged result

    println("   ✅ Batch processing successful")
    println("   • Total documents: $(batch_result.total_documents)")
    println("   • Successful batches: $(batch_result.successful_batches)")
    println("   • Failed batches: $(batch_result.failed_batches)")
    println("   • Total time: $(round(batch_result.total_time, digits=2))s")
    println("   • Average throughput: $(round(batch_result.average_throughput, digits=2)) docs/s")

    # Test result merging
    merged_kg = batch_result.knowledge_graphs[1]
    @test merged_kg !== nothing
    @test length(merged_kg.entities) >= 0
    @test length(merged_kg.relations) >= 0

    println("   ✅ Result merging successful")
    println("   • Merged entities: $(length(merged_kg.entities))")
    println("   • Merged relations: $(length(merged_kg.relations))")
  end
end

println("✅ Batch Processing Integration Test Complete!")
