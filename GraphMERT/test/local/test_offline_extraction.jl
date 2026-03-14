"""
Test script for offline extraction using local LLM backend.

This test verifies the code structure is correct for offline extraction
without requiring an actual GGUF model file.

To run this test:
    julia --project=. test/local/test_offline_extraction.jl

Note: Actual offline extraction requires:
1. A GGUF model file (e.g., TinyLlama-1.1B-Q4_0.gguf)
2. The LlamaCpp.jl package to be installed
"""
module TestOfflineExtraction

using Test
using GraphMERT

# Test 1: Verify LocalLLM module can be loaded (syntax check)
function test_local_llm_module_syntax()
    @testset "LocalLLM module syntax" begin
        # Import the module - this verifies syntax is correct
        @test_throws ArgumentError LocalLLMConfig(model_path = "nonexistent.gguf")
    end
end

# Test 2: Verify ProcessingOptions with local config
function test_processing_options_with_local()
    @testset "ProcessingOptions with local config" begin
        # Create a temp file to satisfy validation
        tmppath = mktemp()[1]

        # Test that ProcessingOptions accepts local config
        config = GraphMERT.LocalLLMConfig(;
            model_path = tmppath,
            context_length = 2048,
            threads = 4,
            temperature = 0.7,
            max_tokens = 512,
            n_gpu_layers = 0,
        )

        # Verify config fields
        @test config.model_path == tmppath
        @test config.context_length == 2048
        @test config.threads == 4
        @test config.temperature == 0.7
        @test config.max_tokens == 512
        @test config.n_gpu_layers == 0
    end
end

# Test 3: Verify LocalLLMClient can be created
function test_local_llm_client_creation()
    @testset "LocalLLMClient creation" begin
        tmppath = mktemp()[1]
        config = GraphMERT.LocalLLMConfig(; model_path = tmppath)

        # Create client (unloaded state)
        client = GraphMERT.LocalLLMClient(config)

        @test client isa GraphMERT.LocalLLMClient
        @test client.config === config
        @test client.is_loaded == false
        @test client.model === nothing
    end
end

# Test 4: Verify extraction API with local config
function test_extraction_with_local_config()
    @testset "Extraction with local config" begin
        tmppath = mktemp()[1]

        # Create local config
        local_config = GraphMERT.LocalLLMConfig(;
            model_path = tmppath,
            context_length = 4096,
            threads = 8,
            temperature = 0.5,
        )

        # Create options with local config enabled (using keyword args)
        options = GraphMERT.ProcessingOptions(
            domain = "wikipedia",
            use_local = true,
            local_config = local_config,
        )

        @test options.use_local == true
        @test options.local_config !== nothing
        @test options.local_config.model_path == tmppath
    end
end

# Test 5: Verify entity/relation extraction signatures
function test_extraction_function_signatures()
    @testset "Extraction function signatures" begin
        # Test that functions have correct signatures
        text = "Python is a programming language."

        # Test discover_head_entities signature
        entities = GraphMERT.discover_head_entities(text)
        @test entities isa Vector{GraphMERT.Entity}
    end
end

# Test 6: Show usage pattern for offline extraction
function test_offline_extraction_usage_pattern()
    @testset "Offline extraction usage pattern" begin
        println("\n" ^ 2)
        println("=" ^ 60)
        println("Offline Extraction Usage Pattern")
        println("=" ^ 60)

        tmppath = mktemp()[1]

        # Create config
        config = LocalLLMConfig(
            model_path = tmppath,
            context_length = 2048,
            threads = 4,
            temperature = 0.7,
            max_tokens = 512
        )

        # Note: In production, you would load the model first:
        # client = load_local_model(config)

        println("Config created with model_path: ", config.model_path)
        println("context_length: ", config.context_length)
        println("threads: ", config.threads)
        println("temperature: ", config.temperature)
        println("max_tokens: ", config.max_tokens)

        println("\nTo run actual extraction:")
        println("1. Download TinyLlama model to ~/.ollama/models/")
        println("2. client = load_local_model(config)")
        println("3. entities = discover_entities(client, text, domain)")
        println("=" ^ 60)

        @test true  # Always pass - this is just demonstration
    end
end

# Test 7: Config validation
function test_config_validation()
    @testset "Config validation" begin
        # Test invalid model_path
        @test_throws ArgumentError LocalLLMConfig(model_path = "nonexistent.gguf")

        # Test invalid context_length
        tmppath = mktemp()[1]
        @test_throws ArgumentError LocalLLMConfig(model_path = tmppath, context_length = 0)
        @test_throws ArgumentError LocalLLMConfig(model_path = tmppath, context_length = 10000)

        # Test invalid threads
        @test_throws ArgumentError LocalLLMConfig(model_path = tmppath, threads = 0)

        # Test invalid temperature
        @test_throws ArgumentError LocalLLMConfig(model_path = tmppath, temperature = -0.1)
        @test_throws ArgumentError LocalLLMConfig(model_path = tmppath, temperature = 2.5)

        # Test invalid max_tokens
        @test_throws ArgumentError LocalLLMConfig(model_path = tmppath, max_tokens = 0)
    end
end

# Test 8: ModelMetadata
function test_model_metadata()
    @testset "LocalModelMetadata" begin
        metadata = LocalModelMetadata(
            name = "TinyLlama",
            filename = "~/.ollama/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            params = 1_100_000_000,
            quantization = "Q4_0",
            ram_estimate = 700,
            context_length = 2048
        )

        @test metadata.name == "TinyLlama"
        @test metadata.params == 1_100_000_000
        @test metadata.quantization == "Q4_0"
        @test metadata.ram_estimate == 700
    end
end

function run_tests()
    println("Running offline extraction tests...")
    println("=" ^ 60)

    test_local_llm_module_syntax()
    test_processing_options_with_local()
    test_local_llm_client_creation()
    test_extraction_with_local_config()
    test_extraction_function_signatures()
    test_offline_extraction_usage_pattern()
    test_config_validation()
    test_model_metadata()

    println("=" ^ 60)
    println("All tests passed!")
end

# Run tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end

end  # module
