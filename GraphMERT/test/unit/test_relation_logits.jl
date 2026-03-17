using Test
using GraphMERT
using Flux

@testset "Compute Relation Logits" begin
    # 1. Setup config
    config = GraphMERT.GraphMERTConfig(
        roberta_config = GraphMERT.RoBERTaConfig(
            vocab_size = 20, hidden_size = 4, num_hidden_layers = 1, num_attention_heads = 1
        ),
        hgat_config = GraphMERT.HGATConfig(input_dim = 4, hidden_dim = 4),
        hidden_dim = 4,
        max_sequence_length = 32, # 4 roots * 8 (1+7)
        relation_types = ["REL1", "REL2"]
    )
    
    # Custom small chain config
    chain_config = GraphMERT.ChainGraphConfig(
        num_roots = 4,
        num_leaves_per_root = 2,
        pad_token_id = 0,
        max_sequence_length = 32
    )
    
    # 2. Create graph
    tokens = ["a", "b", "c", "d"]
    token_ids = [1, 2, 3, 4]
    graph = GraphMERT.create_empty_chain_graph(token_ids, tokens, chain_config)
    
    # 3. Create mock outputs
    batch_size = 2
    seq_len = 32
    hidden = 4
    hgat_output = rand(Float32, batch_size, seq_len, hidden)
    
    # 4. Create classifier
    # Input dim: 2 * hidden = 8
    # Output dim: num_relations = 2
    classifier = Dense(8, 2)
    
    # 5. Run compute_relation_logits
    logits = GraphMERT.compute_relation_logits(hgat_output, graph, classifier)
    
    # 6. Check output shape
    # (batch, num_edges, num_relations)
    # num_edges = 4 * 2 = 8
    @test size(logits) == (batch_size, 8, 2)
    println("Logits shape: ", size(logits))
    
    # 7. Check if forward pass works (Zygote)
    ps = Flux.params(classifier)
    grads = gradient(ps) do
        l = GraphMERT.compute_relation_logits(hgat_output, graph, classifier)
        return sum(l)
    end
    @test grads !== nothing
    println("Gradient check passed")
end
