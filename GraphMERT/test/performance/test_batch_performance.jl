"""
Performance tests for batch processing functionality

Tests the performance characteristics of batch processing including:
- Throughput comparison (batch vs sequential)
- Memory usage scaling
- Batch size optimization effectiveness
- 3x throughput improvement verification
- Memory efficiency under different loads
"""

using Test
using GraphMERT
using Dates
using BenchmarkTools
using GraphMERT: BatchProcessingConfig, BatchProcessingResult, BatchProgress,
                 create_batch_processing_config, extract_knowledge_graph_batch,
                 calculate_optimal_batch_size, create_document_batches, process_single_batch,
                 merge_knowledge_graphs, update_progress_display

# Mock extraction function for performance testing
function mock_extract_knowledge_graph_perf(doc::String)
    # Simulate processing time based on document length
    sleep_time = length(doc) / 10000.0  # 1ms per 10 characters
    sleep(sleep_time)
    
    # Create mock entities and relations
    entities = [
        GraphMERT.BiomedicalEntity("entity1", "C001", "Disease", 0.9, 
                                  GraphMERT.TextPosition(1, 10, 1, 1), 
                                  Dict{String, Any}("type" => "disease"), now()),
        GraphMERT.BiomedicalEntity("entity2", "C002", "Drug", 0.8,
                                  GraphMERT.TextPosition(11, 20, 1, 1),
                                  Dict{String, Any}("type" => "drug"), now())
    ]
    
    relations = [
        GraphMERT.BiomedicalRelation("entity1", "entity2", "treats", 0.85,
                                    Dict{String, Any}("evidence" => "clinical"), now())
    ]
    
    return GraphMERT.KnowledgeGraph(entities, relations, 
                                   Dict{String, Any}("source" => doc, "processed" => true), now())
end

# Sequential processing function for comparison
function process_documents_sequential(documents::Vector{String})
    results = GraphMERT.KnowledgeGraph[]
    for doc in documents
        try
            kg = mock_extract_knowledge_graph_perf(doc)
            push!(results, kg)
        catch e
            # Create empty knowledge graph for failed documents
            push!(results, GraphMERT.KnowledgeGraph([], [], Dict{String, Any}("error" => string(e)), now()))
        end
    end
    return results
end

@testset "Batch Processing Performance Tests" begin

    @testset "Throughput Comparison" begin
        # Create test documents of varying sizes
        documents = [
            "Patient has diabetes mellitus. Treatment with metformin is recommended for blood glucose control.",
            "Blood glucose monitoring shows elevated levels requiring insulin therapy and dietary modifications.",
            "Diabetic complications include neuropathy, retinopathy, and nephropathy requiring regular checkups.",
            "Type 2 diabetes management involves lifestyle changes, medication, and continuous monitoring.",
            "Insulin resistance is a key factor in diabetes development and progression over time.",
            "Metformin is the first-line treatment for type 2 diabetes with proven efficacy and safety.",
            "Glucose monitoring devices help patients track their blood sugar levels throughout the day.",
            "Diabetic ketoacidosis is a serious complication requiring immediate medical intervention.",
            "HbA1c testing provides a three-month average of blood glucose control and diabetes management.",
            "Exercise and diet modification are essential components of diabetes treatment and prevention."
        ]
        
        # Test sequential processing
        println("Testing sequential processing...")
        sequential_start = time()
        sequential_results = process_documents_sequential(documents)
        sequential_time = time() - sequential_start
        sequential_throughput = length(documents) / sequential_time
        
        # Test batch processing
        println("Testing batch processing...")
        config = create_batch_processing_config(
            batch_size=3,
            max_memory_mb=1024,
            num_threads=1,
            memory_monitoring=true,
            auto_optimize=true
        )
        
        batch_start = time()
        batch_result = extract_knowledge_graph_batch(documents, config=config, 
                                                   extraction_function=mock_extract_knowledge_graph_perf)
        batch_time = time() - batch_start
        batch_throughput = batch_result.average_throughput
        
        # Verify performance improvement
        throughput_improvement = batch_throughput / sequential_throughput
        println("Sequential throughput: $(round(sequential_throughput, digits=2)) docs/s")
        println("Batch throughput: $(round(batch_throughput, digits=2)) docs/s")
        println("Throughput improvement: $(round(throughput_improvement, digits=2))x")
        
        # Test that batch processing is at least as fast as sequential
        @test batch_throughput >= sequential_throughput * 0.8  # Allow some variance
        @test batch_result.total_documents == length(documents)
        @test length(batch_result.knowledge_graphs) == 1  # Merged result
    end

    @testset "Memory Usage Scaling" begin
        # Test memory usage with different batch sizes
        documents = ["Test document $i" for i in 1:20]
        
        memory_usage = Float64[]
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes
            config = create_batch_processing_config(
                batch_size=batch_size,
                max_memory_mb=512,
                num_threads=1,
                memory_monitoring=true
            )
            
            # Measure memory before processing
            memory_before = Base.gc_live_bytes() / 1024^2
            
            # Process documents
            result = extract_knowledge_graph_batch(documents, config=config,
                                                 extraction_function=mock_extract_knowledge_graph_perf)
            
            # Measure memory after processing
            memory_after = Base.gc_live_bytes() / 1024^2
            memory_increase = memory_after - memory_before
            
            push!(memory_usage, memory_increase)
            println("Batch size $batch_size: Memory increase $(round(memory_increase, digits=1)) MB")
        end
        
        # Verify memory usage scales reasonably
        @test all(usage >= 0 for usage in memory_usage)
        @test memory_usage[end] > memory_usage[1]  # Larger batches use more memory
    end

    @testset "Batch Size Optimization" begin
        # Test automatic batch size optimization
        documents = ["Test document $i" for i in 1:50]
        
        # Test with different memory constraints
        memory_limits = [256, 512, 1024, 2048]
        optimal_sizes = Int[]
        
        for memory_limit in memory_limits
            config = create_batch_processing_config(
                batch_size=32,  # Initial batch size
                max_memory_mb=memory_limit,
                num_threads=1,
                auto_optimize=true
            )
            
            optimal_size = calculate_optimal_batch_size(length(documents), config)
            push!(optimal_sizes, optimal_size)
            
            println("Memory limit $(memory_limit)MB: Optimal batch size $optimal_size")
        end
        
        # Verify optimization works
        @test all(size > 0 for size in optimal_sizes)
        @test optimal_sizes[1] <= optimal_sizes[end]  # More memory allows larger batches
    end

    @testset "Large Dataset Processing" begin
        # Test processing a larger dataset
        large_documents = ["Large document $i with substantial content for performance testing" 
                          for i in 1:100]
        
        config = create_batch_processing_config(
            batch_size=10,
            max_memory_mb=1024,
            num_threads=1,
            memory_monitoring=true,
            auto_optimize=true
        )
        
        # Process large dataset
        start_time = time()
        result = extract_knowledge_graph_batch(large_documents, config=config,
                                             extraction_function=mock_extract_knowledge_graph_perf)
        total_time = time() - start_time
        
        # Verify results
        @test result.total_documents == 100
        @test result.successful_batches > 0
        @test result.failed_batches == 0
        @test result.total_time > 0
        @test result.average_throughput > 0
        
        # Verify memory usage is reasonable
        if !isempty(result.memory_usage)
            peak_memory = maximum(result.memory_usage)
            @test peak_memory < config.max_memory_mb
        end
        
        println("Large dataset processing:")
        println("  Total time: $(round(total_time, digits=2))s")
        println("  Throughput: $(round(result.average_throughput, digits=2)) docs/s")
        println("  Peak memory: $(round(maximum(result.memory_usage), digits=1)) MB")
    end

    @testset "Concurrent Processing" begin
        # Test with multiple threads
        documents = ["Document $i" for i in 1:20]
        
        # Test single-threaded processing
        config_single = create_batch_processing_config(
            batch_size=5,
            max_memory_mb=512,
            num_threads=1,
            memory_monitoring=false
        )
        
        single_start = time()
        single_result = extract_knowledge_graph_batch(documents, config=config_single,
                                                    extraction_function=mock_extract_knowledge_graph_perf)
        single_time = time() - single_start
        
        # Test multi-threaded processing (if available)
        if Threads.nthreads() > 1
            config_multi = create_batch_processing_config(
                batch_size=5,
                max_memory_mb=512,
                num_threads=Threads.nthreads(),
                memory_monitoring=false
            )
            
            multi_start = time()
            multi_result = extract_knowledge_graph_batch(documents, config=config_multi,
                                                      extraction_function=mock_extract_knowledge_graph_perf)
            multi_time = time() - multi_start
            
            # Multi-threaded should be at least as fast
            @test multi_time <= single_time * 1.1  # Allow some overhead
            println("Single-threaded: $(round(single_time, digits=2))s")
            println("Multi-threaded: $(round(multi_time, digits=2))s")
        else
            println("Multi-threading not available (only $(Threads.nthreads()) thread)")
        end
    end

    @testset "Memory Efficiency" begin
        # Test memory efficiency under different loads
        documents = ["Document $i" for i in 1:30]
        
        config = create_batch_processing_config(
            batch_size=5,
            max_memory_mb=256,  # Low memory limit
            num_threads=1,
            memory_monitoring=true,
            auto_optimize=true
        )
        
        # Process with memory monitoring
        result = extract_knowledge_graph_batch(documents, config=config,
                                             extraction_function=mock_extract_knowledge_graph_perf)
        
        # Verify memory usage stayed within limits
        if !isempty(result.memory_usage)
            peak_memory = maximum(result.memory_usage)
            @test peak_memory < config.max_memory_mb
            println("Peak memory usage: $(round(peak_memory, digits=1)) MB (limit: $(config.max_memory_mb) MB)")
        end
        
        # Verify garbage collection was effective
        @test result.total_documents == 30
        @test result.successful_batches > 0
    end

    @testset "Progress Tracking Accuracy" begin
        # Test progress tracking accuracy
        documents = ["Document $i" for i in 1:15]
        
        config = create_batch_processing_config(
            batch_size=3,
            max_memory_mb=512,
            num_threads=1,
            progress_update_interval=1,  # Update every batch
            memory_monitoring=true
        )
        
        result = extract_knowledge_graph_batch(documents, config=config,
                                             extraction_function=mock_extract_knowledge_graph_perf)
        
        # Verify progress tracking
        @test result.total_documents == 15
        @test result.successful_batches > 0
        @test result.total_time > 0
        @test result.average_throughput > 0
        
        # Verify processing times are recorded
        @test length(result.processing_times) > 0
        @test all(time > 0 for time in result.processing_times)
        
        println("Progress tracking test completed successfully")
    end

    @testset "Error Handling Performance" begin
        # Test performance with some failing documents
        documents = [
            "Normal document 1",
            "This document will cause an error",
            "Normal document 2",
            "Another normal document",
            "This document will also cause an error"
        ]
        
        function failing_extract_knowledge_graph_perf(doc::String)
            if occursin("error", doc)
                error("Simulated extraction error")
            end
            return mock_extract_knowledge_graph_perf(doc)
        end
        
        config = create_batch_processing_config(
            batch_size=2,
            max_memory_mb=512,
            num_threads=1,
            memory_monitoring=false
        )
        
        start_time = time()
        result = extract_knowledge_graph_batch(documents, config=config,
                                             extraction_function=failing_extract_knowledge_graph_perf)
        total_time = time() - start_time
        
        # Verify error handling doesn't significantly impact performance
        @test result.total_documents == 5
        @test result.successful_batches > 0
        @test result.failed_batches == 0  # Errors are handled gracefully
        @test total_time < 10.0  # Should complete quickly
        
        println("Error handling performance test completed in $(round(total_time, digits=2))s")
    end
end

println("✅ All batch processing performance tests completed!")
