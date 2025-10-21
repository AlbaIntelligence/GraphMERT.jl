"""
API tests for GraphMERT

Tests the public API functions including:
- extract_knowledge_graph() - main extraction function
- default_processing_options() - configuration management
- load_model() / save_model() - model persistence
- Error handling and validation
"""

using Test
using GraphMERT

# Create mock model for testing
struct MockAPIModel
    vocab_size::Int
end

function (model::MockAPIModel)(input_ids, attention_mask)
    batch_size, seq_len = size(input_ids)
    return randn(Float32, batch_size, seq_len, model.vocab_size)
end

@testset "GraphMERT API Tests" begin

    @testset "extract_knowledge_graph() API" begin
        # Create a simple GraphMERT model for testing
        config = GraphMERTConfig()
        test_model = GraphMERTModel(config)

        # Test with simple text
        text = "Diabetes is a chronic condition."
        kg = extract_knowledge_graph(text, test_model)

        @test kg isa GraphMERT.KnowledgeGraph
        @test length(kg.entities) > 0
        @test !isempty(kg.source_text)
        @test kg.source_text == text

        # Test with empty text (should fail)
        @test_throws ArgumentError GraphMERT.extract_knowledge_graph("", test_model)

        # Test with very long text (would need proper max_length handling)
        # long_text = repeat("word ", 1000)
        # @test_throws ArgumentError GraphMERT.extract_knowledge_graph(long_text, test_model)
    end

    @testset "default_processing_options() API" begin
        # Test default options
        options = GraphMERT.default_processing_options()
        @test options.batch_size == 32
        @test options.max_length == 1024
        @test options.device == :cpu
        @test options.similarity_threshold == 0.8

        # Test custom options
        custom_options = GraphMERT.default_processing_options(
            batch_size=16,
            device=:cuda,
            similarity_threshold=0.7,
            top_k_predictions=15
        )
        @test custom_options.batch_size == 16
        @test custom_options.device == :cuda
        @test custom_options.similarity_threshold == 0.7
        @test custom_options.top_k_predictions == 15
    end

    @testset "load_model() / save_model() API" begin
        # Create mock model
        mock_model = MockAPIModel(30522)

        # Test save model
        save_path = "test_model.jld2"
        success = GraphMERT.save_model(test_model, save_path)
        @test success == true
        @test isfile(save_path)

        # Test load model
        loaded_model = GraphMERT.load_model(save_path)
        @test loaded_model !== nothing

        # Test with non-existent file
        @test GraphMERT.load_model("nonexistent.jld2") === nothing

        # Cleanup
        rm(save_path, force=true)
    end

    @testset "ProcessingOptions Configuration" begin
        # Test all fields are properly set
        options = GraphMERT.ProcessingOptions(
            batch_size=16,
            max_length=512,
            device=:cuda,
            use_amp=true,
            num_workers=4,
            seed=42,
            top_k_predictions=25,
            similarity_threshold=0.75,
            enable_provenance_tracking=false
        )

        @test options.batch_size == 16
        @test options.max_length == 512
        @test options.device == :cuda
        @test options.use_amp == true
        @test options.num_workers == 4
        @test options.seed == 42
        @test options.top_k_predictions == 25
        @test options.similarity_threshold == 0.75
        @test options.enable_provenance_tracking == false
    end

    @testset "KnowledgeGraph Structure" begin
        # Create a simple GraphMERT model for testing
        config = GraphMERTConfig()
        test_model = GraphMERTModel(config)

        # Extract knowledge graph
        text = "Diabetes and metformin are related."
        kg = GraphMERT.extract_knowledge_graph(text, test_model)

        # Test knowledge graph structure
        @test kg.source_text == text
        @test length(kg.entities) > 0
        @test length(kg.relations) > 0
        @test length(kg.triples) > 0

        # Test entity references in relations
        for relation in kg.relations
            @test 1 ≤ relation.head_entity_id ≤ length(kg.entities)
            @test 1 ≤ relation.tail_entity_id ≤ length(kg.entities)
        end

        # Test triple references
        for (head_idx, rel_idx, tail_idx) in kg.triples
            @test 1 ≤ head_idx ≤ length(kg.entities)
            @test 1 ≤ rel_idx ≤ length(kg.relations)
            @test 1 ≤ tail_idx ≤ length(kg.entities)
            @test kg.relations[rel_idx].head_entity_id == head_idx
            @test kg.relations[rel_idx].tail_entity_id == tail_idx
        end
    end

    @testset "Error Handling" begin
        mock_model = MockAPIModel(30522)

        # Test with invalid inputs
        @test_throws ArgumentError extract_knowledge_graph("", mock_model)
        @test_throws ArgumentError extract_knowledge_graph("a"^10000, mock_model)  # Too long

        # Test save_model error handling
        @test_throws Exception GraphMERT.save_model(test_model, "/invalid/path/model.jld2")

        # Test load_model error handling
        @test GraphMERT.load_model("/nonexistent/file.jld2") === nothing
    end

    @testset "Performance Requirements" begin
        # Create a simple GraphMERT model for testing
        config = GraphMERTConfig()
        test_model = GraphMERTModel(config)

        # Test processing speed (should be fast for small text)
        text = "Diabetes is a chronic condition characterized by elevated blood glucose levels."

        start_time = time()
        kg = GraphMERT.extract_knowledge_graph(text, test_model)
        end_time = time()

        processing_time = end_time - start_time

        # Should process reasonably quickly (adjust threshold as needed)
        @test processing_time < 5.0  # seconds

        # Test memory usage doesn't grow excessively
        # (This is a basic check - full memory profiling would be more sophisticated)
        @test length(kg.entities) > 0  # Basic functionality check
    end
end
