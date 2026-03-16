using Test
using GraphMERT
using Flux
using JLD2

@testset "Model Persistence Tests" begin

    # Helper to create a small test model config
    function create_test_config()
        r_conf = GraphMERT.RoBERTaConfig(
            vocab_size = 100,
            hidden_size = 32,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            max_position_embeddings = 64
        )
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
        test_dir = mktempdir()
        test_path = joinpath(test_dir, "model_roundtrip.jld2")
        
        try
            # 1. Create model
            config = create_test_config()
            model = GraphMERT.GraphMERTModel(config)
            
            # Mutate a weight to verify restoration
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
            
        finally
            rm(test_dir, recursive=true, force=true)
        end
    end
end
