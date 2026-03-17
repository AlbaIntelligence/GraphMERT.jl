"""
Unit tests for Model Persistence in GraphMERT.jl

Tests the checkpoint saving and loading functionality including:
- Model checkpoint saving
- Checkpoint loading and restoration
- Metadata persistence
- Error handling for corrupted checkpoints
- Cross-platform compatibility
"""

using Test
using Random
using GraphMERT
using Flux
using Flux: state, loadmodel!
# using GraphMERT: save_training_checkpoint, log_training_step, create_training_configurations
# using GraphMERT: MLMConfig, MNMConfig, SeedInjectionConfig

@testset "Model Persistence Tests" begin
    
    # Helper to create a small test model config to keep tests fast
    function create_test_config()
        r_conf = GraphMERT.RoBERTaConfig(
            vocab_size = 100,
            hidden_size = 32,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            max_position_embeddings = 64
        )
        # HGAT config must be compatible
        h_conf = GraphMERT.HGATConfig(
            input_dim = 32,
            hidden_dim = 32,
            num_heads = 2,
            num_layers = 1
        )
        return GraphMERT.GraphMERTConfig(
            roberta_config = r_conf,
            hgat_config = h_conf,
            max_sequence_length = 64,
            hidden_dim = 32
        )
    end

    @testset "JLD2 Round-Trip" begin
        # Use mktempdir via Base
        test_dir = mktempdir()
        test_path = joinpath(test_dir, "model_roundtrip.jld2")
        
        try
            # 1. Create model
            config = create_test_config()
            model = GraphMERT.GraphMERTModel(config)
            
            # Mutate a weight to verify restoration (ensure we don't just get fresh init)
            # Access deep into RoBERTa word embeddings
            w_matrix = model.roberta.embeddings.word_embeddings.weight
            original_val = w_matrix[1]
            w_matrix[1] += 1.5f0
            expected_val = w_matrix[1]
            
            # 2. Save
            success = GraphMERT.save_model(model, test_path)
            @test success
            @test isfile(test_path)
            
            # 3. Load back
            loaded_model = GraphMERT.load_model(test_path)
            @test loaded_model isa GraphMERT.GraphMERTModel
            
            # 4. Verify config restoration
            @test loaded_model.config.hidden_dim == 32
            @test loaded_model.config.roberta_config.vocab_size == 100
            
            # 5. Verify weight restoration
            loaded_w = loaded_model.roberta.embeddings.word_embeddings.weight
            loaded_val = loaded_w[1]
            
            @test loaded_val ≈ expected_val
            @test loaded_val ≉ original_val 
            
            # 6. Verify entire state structure matches
            ps_orig = Flux.params(model)
            ps_load = Flux.params(loaded_model)
            @test length(ps_orig) == length(ps_load)

        finally
            rm(test_dir, recursive=true, force=true)
        end
    end

    @testset "Optimizer State" begin
        test_dir = mktempdir()
        test_path = joinpath(test_dir, "model_opt.jld2")
        
        try
            config = create_test_config()
            model = GraphMERT.GraphMERTModel(config)
            
            # Create optimizer with specific params
            # Note: Flux.Adam(eta, (beta1, beta2))
            lr = 0.00123
            betas = (0.85, 0.995)
            opt = Flux.Adam(lr, betas)
            
            # Save with optimizer state
            success = GraphMERT.save_model(
                model, 
                test_path; 
                optimizer=opt, 
                include_optimizer_state=true
            )
            @test success
            
            # Create a fresh optimizer (default params)
            new_opt = Flux.Adam(0.1, (0.9, 0.999))
            @test new_opt.eta != lr
            
            # Load state into new optimizer
            # Note: load_optimizer_state! is not exported, access via module
            success_load = GraphMERT.load_optimizer_state!(new_opt, test_path)
            @test success_load
            
            # Verify restoration
            @test new_opt.eta ≈ lr
            @test new_opt.beta == betas
            
        finally
            rm(test_dir, recursive=true, force=true)
        end
    end

    @testset "Error Handling" begin
        test_dir = mktempdir()
        try
            # Test loading non-existent file
            @test GraphMERT.load_model(joinpath(test_dir, "nonexistent.jld2")) === nothing
            
            # Test loading corrupted file
            bad_path = joinpath(test_dir, "corrupt.jld2")
            write(bad_path, "not a jld2 file")
            # JLD2 usually throws, our wrapper catches and returns nothing
            @test GraphMERT.load_model(bad_path) === nothing
        finally
            rm(test_dir, recursive=true, force=true)
        end
    end

    @testset "Legacy/Fallback Handling" begin
        # Test that passing a directory returns default/nothing as appropriate
        # (This exercises the legacy JSON fallback path)
        empty_dir = mktempdir()
        try
            @test GraphMERT.load_model(empty_dir) === nothing
        finally
            rm(empty_dir, recursive=true, force=true)
        end
    end
end
