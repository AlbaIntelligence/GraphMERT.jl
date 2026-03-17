using Test
using GraphMERT
using Flux
using Random

@testset "Training with Seed Injection Integration" begin
    # 1. Setup small model and config
    config = GraphMERT.GraphMERTConfig(
        roberta_config = GraphMERT.RoBERTaConfig(
            vocab_size = 20,      # Micro vocab
            hidden_size = 16,     # Micro hidden
            num_hidden_layers = 1, # Single layer
            num_attention_heads = 1, # Single head
            intermediate_size = 32,
            max_position_embeddings = 128
        ),
        hgat_config = GraphMERT.HGATConfig(
            input_dim = 16,
            hidden_dim = 16,
            num_heads = 1,
            num_layers = 1
        ),
        hidden_dim = 16,
        max_sequence_length = 128
    )
    
    # Tiny chain config
    chain_config = GraphMERT.ChainGraphConfig(
        num_roots = 4,
        num_leaves_per_root = 2,
        pad_token_id = 0,
        max_sequence_length = 128
    )
    
    # MLM config matching the model vocab
    mlm_config = GraphMERT.MLMConfig(
        vocab_size = 20,
        mask_token_id = 19, # Use last token as mask
        hidden_size = 16,
        max_length = 128
    )

    # MNM config matching the model vocab
    mnm_config = GraphMERT.MNMConfig(
        vocab_size = 20,
        mask_token_id = 19,
        hidden_size = 16
    )
    
    # 2. Prepare Injection Data
    # Seed triples with valid CUIs
    seed_kg = [
        GraphMERT.SemanticTriple("diabetes", "C0011849", "treats", "metformin", [1, 2], 0.9, "UMLS"),
        GraphMERT.SemanticTriple("diabetes", "C0011849", "associated_with", "obesity", [3, 4], 0.8, "UMLS"),
        GraphMERT.SemanticTriple("cancer", "C0006826", "causes", "death", [5, 6], 0.9, "UMLS")
    ]
    
    injection_config = GraphMERT.SeedInjectionConfig(
        0.5, 5, 10, 0.5, 10, 5, 0.2, 10
    )
    
    # 3. Create Training Text matching the seed KG
    train_texts = [
        "diabetes is a disease",  # Matches "diabetes"
        "cancer is bad",          # Matches "cancer"
        "nothing matches here"
    ]
    
    # 4. Run Training with Injection
    mktempdir() do tmp_dir
        println("Creating model...")
        # Create model explicitly and convert to Float64 to avoid Flux warnings on CPU
        model = GraphMERT.create_graphmert_model(config)
        model = Flux.f64(model)
        println("Model created.")

        # We just want to ensure it runs without error (no bounds errors in injection)
        println("Calling train_graphmert...")
        flush(stdout)
        trained_model = train_graphmert(
            train_texts,
            config;
            model=model,
            seed_kg=seed_kg,
            mlm_config=mlm_config,
            mnm_config=mnm_config,
            injection_config=injection_config,
            num_epochs=1,
            checkpoint_dir=tmp_dir,
            save_checkpoints=false,
            chain_config=chain_config,
            max_steps_per_epoch=3
        )
        
        @test model isa GraphMERTModel
        println("Training with injection completed successfully")
    end
end
