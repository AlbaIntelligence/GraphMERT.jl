using Test
using GraphMERT
using Flux
using Random
using Dates

@testset "Real Training Loop Integration" begin
    # 1. Setup small model and config
    config = GraphMERT.GraphMERTConfig(
        roberta_config = GraphMERT.RoBERTaConfig(
            vocab_size = 20,      # Micro vocab
            hidden_size = 16,     # Micro hidden
            num_hidden_layers = 1, # Single layer
            num_attention_heads = 1, # Single head
            intermediate_size = 32,
            max_position_embeddings = 32
        ),
        hgat_config = GraphMERT.HGATConfig(
            input_dim = 16,
            hidden_dim = 16,
            num_heads = 1,
            num_layers = 1
        ),
        hidden_dim = 16,
        max_sequence_length = 32
    )
    
    mlm_config = GraphMERT.MLMConfig(
        vocab_size = 20,
        mask_probability = 0.15,
        mask_token_id = 19
    )
    
    mnm_config = GraphMERT.MNMConfig(
        vocab_size = 20,
        mask_probability = 0.15,
        mask_token_id = 19
    )
    
    # Tiny chain config for test speed
    chain_config = GraphMERT.ChainGraphConfig(
        num_roots = 4,
        num_leaves_per_root = 2,
        pad_token_id = 0
    )
    
    # Ensure domain is registered for validation
    if "biomedical" ∉ GraphMERT.list_domains()
        GraphMERT.register_domain!("biomedical", GraphMERT.BiomedicalDomain())
    end

    # 2. Create Model
    model = GraphMERT.create_graphmert_model(config)
    
    # Snapshot parameters before training
    ps_before = deepcopy(Flux.params(model))
    p1_before = copy(model.roberta.embeddings.word_embeddings.weight)
    
    # 3. Create Dummy Data
    train_texts = [
        "entity1 relates to entity2",
        "entity3 causes entity4",
        "entity1 treats entity3"
    ]
    
    # 4. Run Training Loop (1 epoch, minimal steps)
    # We use a temp directory for checkpoints to avoid clutter
    mktempdir() do tmp_dir
        trained_model = train_graphmert(
            train_texts,
            config;
            mlm_config=mlm_config,
            mnm_config=mnm_config,
            num_epochs=1,
            learning_rate=1e-3,
            checkpoint_dir=tmp_dir,
            save_checkpoints=true,
            max_steps_per_epoch=2,
            val_texts=["entity1 treats entity2"], # Trigger validation logic
            val_interval=1,
            chain_config=chain_config  # Pass tiny chain config
        )
        
        # 5. Assertions
        
        # Check that parameters changed
        p1_after = trained_model.roberta.embeddings.word_embeddings.weight
        @test p1_before != p1_after
        
        # Check checkpoints were created
        @test isfile(joinpath(tmp_dir, "latest.jld2"))
        
        # Check that model is still functional
        @test trained_model isa GraphMERTModel
    end
end
