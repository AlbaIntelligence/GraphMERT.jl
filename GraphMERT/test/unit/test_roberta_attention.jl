using Test
using Flux
using GraphMERT
using GraphMERT: RoBERTaConfig, RoBERTaModel, RoBERTaEmbeddings

@testset "RoBERTa Attention Decay" begin
    config = RoBERTaConfig(
        hidden_size = 64,
        num_attention_heads = 4,
        num_hidden_layers = 1,
        max_position_embeddings = 128
    )
    model = RoBERTaModel(config)
    
    batch_size = 2
    seq_len = 10
    input_ids = rand(1:100, seq_len, batch_size)
    attention_mask = ones(Float32, batch_size, seq_len, seq_len)
    position_ids = zeros(Int, seq_len, batch_size)
    token_type_ids = zeros(Int, seq_len, batch_size)
    
    # 1. Standard forward pass
    out1, _ = model(input_ids, attention_mask, position_ids, token_type_ids)
    @test size(out1) == (batch_size, seq_len, 64)
    
    # 2. Forward pass with attention decay (should be different)
    # Currently, the API doesn't support passing the decay mask.
    # We expect this to fail or require signature update.
    
    # Create a decay mask (simple distance bias)
    decay_mask_matrix = zeros(Float32, seq_len, seq_len)
    for i in 1:seq_len, j in 1:seq_len
        # Create decay mask values in [0, 1] (exp space)
        # Because implementation takes log(mask), we simulate exp(-0.1 * dist)
        decay_mask_matrix[i, j] = exp(-0.1f0 * abs(i - j))
    end
    decay_mask = GraphMERT.AttentionDecayMask(decay_mask_matrix, exp, seq_len, 0.1f0)

    # Call with extra argument
    out2, _ = model(input_ids, attention_mask, position_ids, token_type_ids, decay_mask)
    @test out1 ≉ out2 # Should NOT be equal now that it is implemented
    @info "Model successfully processed decay mask argument and outputs differ"
end
