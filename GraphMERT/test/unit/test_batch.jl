"""
Unit tests for batch processing functionality

Tests the batch processing API including:
- Batch configuration creation
- Document batching
- Batch processing
- Result merging
- Progress tracking
- Memory monitoring
"""

using Test
using GraphMERT
using Dates
using GraphMERT: BatchProcessingConfig, BatchProcessingResult, BatchProgress,
                 create_batch_processing_config, extract_knowledge_graph_batch,
                 calculate_optimal_batch_size, create_document_batches, process_single_batch,
                 merge_knowledge_graphs, update_progress_display

# Mock extraction function for testing
function mock_extract_knowledge_graph(doc::String)
    # Create mock entities and relations based on document content
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

@testset "Batch Processing Unit Tests" begin

    @testset "Batch Configuration" begin
        # Test default configuration
        config = create_batch_processing_config()
        @test config.batch_size == 32
        @test config.max_memory_mb == 2048
        @test config.num_threads >= 1
        @test config.progress_update_interval == 10
        @test config.memory_monitoring == true
        @test config.auto_optimize == true
        @test config.merge_strategy == "union"
        
        # Test custom configuration
        custom_config = create_batch_processing_config(
            batch_size=16,
            max_memory_mb=1024,
            num_threads=2,
            progress_update_interval=5,
            memory_monitoring=false,
            auto_optimize=false,
            merge_strategy="intersection"
        )
        @test custom_config.batch_size == 16
        @test custom_config.max_memory_mb == 1024
        @test custom_config.num_threads == 2
        @test custom_config.progress_update_interval == 5
        @test custom_config.memory_monitoring == false
        @test custom_config.auto_optimize == false
        @test custom_config.merge_strategy == "intersection"
    end

    @testset "Optimal Batch Size Calculation" begin
        config = create_batch_processing_config(max_memory_mb=1000, num_threads=4)
        
        # Test with small document count
        optimal_size = calculate_optimal_batch_size(10, config)
        @test optimal_size >= 1
        @test optimal_size <= config.batch_size
        
        # Test with large document count
        optimal_size_large = calculate_optimal_batch_size(1000, config)
        @test optimal_size_large >= 1
        @test optimal_size_large <= config.batch_size
        
        # Test with very large document count
        optimal_size_very_large = calculate_optimal_batch_size(10000, config)
        @test optimal_size_very_large >= 1
        @test optimal_size_very_large <= config.batch_size
    end

    @testset "Document Batching" begin
        # Test with small document set
        docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        batches = create_document_batches(docs, 2)
        @test length(batches) == 3
        @test length(batches[1]) == 2
        @test length(batches[2]) == 2
        @test length(batches[3]) == 1
        
        # Test with exact batch size
        docs_exact = ["doc1", "doc2", "doc3", "doc4"]
        batches_exact = create_document_batches(docs_exact, 2)
        @test length(batches_exact) == 2
        @test length(batches_exact[1]) == 2
        @test length(batches_exact[2]) == 2
        
        # Test with single document
        docs_single = ["doc1"]
        batches_single = create_document_batches(docs_single, 2)
        @test length(batches_single) == 1
        @test length(batches_single[1]) == 1
        
        # Test with empty document set
        docs_empty = String[]
        batches_empty = create_document_batches(docs_empty, 2)
        @test length(batches_empty) == 0
    end

    @testset "Single Batch Processing" begin
        config = create_batch_processing_config(num_threads=1)  # Sequential processing
        docs = ["Patient has diabetes. Treatment with metformin.", 
                "Blood glucose levels are elevated. Insulin therapy needed."]
        
        results = process_single_batch(docs, mock_extract_knowledge_graph, config)
        
        @test length(results) == 2
        @test all(kg isa GraphMERT.KnowledgeGraph for kg in results)
        @test all(length(kg.entities) > 0 for kg in results)
        @test all(length(kg.relations) > 0 for kg in results)
    end

    @testset "Knowledge Graph Merging" begin
        # Create test knowledge graphs
        kg1 = mock_extract_knowledge_graph("doc1")
        kg2 = mock_extract_knowledge_graph("doc2")
        kg3 = mock_extract_knowledge_graph("doc3")
        
        # Test union merging
        merged_union = merge_knowledge_graphs([kg1, kg2, kg3], "union")
        @test merged_union isa GraphMERT.KnowledgeGraph
        @test merged_union.metadata["merged"] == true
        @test merged_union.metadata["source_graphs"] == 3
        @test merged_union.metadata["strategy"] == "union"
        
        # Test intersection merging
        merged_intersection = merge_knowledge_graphs([kg1, kg2], "intersection")
        @test merged_intersection isa GraphMERT.KnowledgeGraph
        @test merged_intersection.metadata["merged"] == true
        @test merged_intersection.metadata["source_graphs"] == 2
        @test merged_intersection.metadata["strategy"] == "intersection"
        
        # Test weighted merging
        merged_weighted = merge_knowledge_graphs([kg1, kg2], "weighted")
        @test merged_weighted isa GraphMERT.KnowledgeGraph
        @test merged_weighted.metadata["merged"] == true
        @test merged_weighted.metadata["source_graphs"] == 2
        @test merged_weighted.metadata["strategy"] == "weighted"
        
        # Test empty list
        merged_empty = merge_knowledge_graphs(GraphMERT.KnowledgeGraph[], "union")
        @test merged_empty isa GraphMERT.KnowledgeGraph
        @test merged_empty.metadata["merged"] == true
        
        # Test single graph
        merged_single = merge_knowledge_graphs([kg1], "union")
        @test merged_single === kg1
    end

    @testset "Progress Tracking" begin
        # Test progress initialization
        progress = BatchProgress(10, 0, 0, time(), time(), 0.0, 0.0, 0.0, "initializing")
        @test progress.total_batches == 10
        @test progress.completed_batches == 0
        @test progress.status == "initializing"
        
        # Test progress update
        progress.completed_batches = 5
        progress.current_throughput = 2.5
        progress.memory_usage = 512.0
        progress.status = "processing"
        
        @test progress.completed_batches == 5
        @test progress.current_throughput == 2.5
        @test progress.memory_usage == 512.0
        @test progress.status == "processing"
    end

    @testset "Batch Processing Integration" begin
        # Test with small document set
        docs = [
            "Patient has diabetes mellitus. Treatment with metformin is recommended.",
            "Blood glucose monitoring shows elevated levels. Insulin therapy may be required.",
            "Diabetic complications include neuropathy and retinopathy. Regular checkups needed.",
            "Type 2 diabetes management involves lifestyle changes and medication.",
            "Insulin resistance is a key factor in diabetes development."
        ]
        
        config = create_batch_processing_config(
            batch_size=2,
            max_memory_mb=512,
            num_threads=1,
            progress_update_interval=1,
            memory_monitoring=true,
            auto_optimize=false
        )
        
        # Test batch processing
        result = extract_knowledge_graph_batch(docs, config=config, extraction_function=mock_extract_knowledge_graph)
        
        @test result isa BatchProcessingResult
        @test result.total_documents == 5
        @test result.successful_batches > 0
        @test result.failed_batches == 0
        @test result.total_time > 0
        @test result.average_throughput > 0
        @test length(result.knowledge_graphs) == 1  # Merged result
        @test length(result.processing_times) > 0
        @test length(result.memory_usage) > 0
        
        # Test merged knowledge graph
        merged_kg = result.knowledge_graphs[1]
        @test merged_kg isa GraphMERT.KnowledgeGraph
        @test merged_kg.metadata["merged"] == true
        @test merged_kg.metadata["source_graphs"] > 0
    end

    @testset "Error Handling" begin
        # Test with documents that cause extraction errors
        function failing_extract_knowledge_graph(doc::String)
            if occursin("error", doc)
                error("Simulated extraction error")
            end
            return mock_extract_knowledge_graph(doc)
        end
        
        docs_with_errors = [
            "Normal document for processing.",
            "This document will cause an error.",
            "Another normal document."
        ]
        
        config = create_batch_processing_config(batch_size=1, num_threads=1)
        result = extract_knowledge_graph_batch(docs_with_errors, 
                                             config=config, 
                                             extraction_function=failing_extract_knowledge_graph)
        
        @test result isa BatchProcessingResult
        @test result.total_documents == 3
        @test result.successful_batches > 0
        # Should handle errors gracefully
    end

    @testset "Memory Monitoring" begin
        config = create_batch_processing_config(
            batch_size=1,
            max_memory_mb=100,
            memory_monitoring=true
        )
        
        docs = ["Test document for memory monitoring."]
        result = extract_knowledge_graph_batch(docs, config=config, extraction_function=mock_extract_knowledge_graph)
        
        @test result isa BatchProcessingResult
        @test length(result.memory_usage) > 0
        @test all(usage >= 0 for usage in result.memory_usage)
    end

    @testset "Performance Metrics" begin
        config = create_batch_processing_config(batch_size=2, num_threads=1)
        docs = ["doc1", "doc2", "doc3", "doc4"]
        
        result = extract_knowledge_graph_batch(docs, config=config, extraction_function=mock_extract_knowledge_graph)
        
        # Test performance metrics
        @test result.total_time > 0
        @test result.average_throughput > 0
        @test result.average_throughput == result.total_documents / result.total_time
        @test length(result.processing_times) > 0
        @test all(time > 0 for time in result.processing_times)
    end
end

println("✅ All batch processing unit tests passed!")
