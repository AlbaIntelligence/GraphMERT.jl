using Test
using GraphMERT
using Flux
using Dates

@testset "Validation Loop Integration" begin
    # Create temp dir
    test_dir = mktempdir()
    checkpoint_dir = joinpath(test_dir, "checkpoints")
    
    try
        # Create small config matching model dimensions
        config = GraphMERT.GraphMERTConfig(
            roberta_config = GraphMERT.RoBERTaConfig(
                vocab_size=100, 
                hidden_size=32, 
                num_hidden_layers=1, 
                num_attention_heads=2,
                max_position_embeddings=64
            ),
            hgat_config = GraphMERT.HGATConfig(
                input_dim=32, 
                hidden_dim=32, 
                num_heads=2, 
                num_layers=1
            ),
            hidden_dim = 32,
            max_sequence_length=64
        )
        
        # Load and register biomedical domain
        # We need to manually register it because train_graphmert assumes it's available
        # if we pass domain="biomedical" (which is hardcoded in pipeline.jl for now)
        if GraphMERT.get_domain("biomedical") === nothing
            # Attempt to load it if not present.
            # Depending on how the test env is set up, we might need to include the file.
            # But GraphMERT.jl includes biomedical.jl.
            # So we just need to load it.
            bio = GraphMERT.BiomedicalDomain()
            GraphMERT.register_domain!("biomedical", bio)
        end
        
        train_texts = ["test sentence one", "test sentence two"]
        val_texts = ["validation sentence one", "validation sentence two"]
        
        mlm_config = GraphMERT.MLMConfig(vocab_size=100, mask_token_id=99)
        mnm_config = GraphMERT.MNMConfig(vocab_size=100, mask_token_id=99)
        
        chain_config = GraphMERT.ChainGraphConfig(
            num_roots=8,
            num_leaves_per_root=2,
            max_sequence_length=64,
            vocab_size=100
        )
        
        # Run training with validation
        model = train_graphmert(
            train_texts,
            config;
            mlm_config=mlm_config,
            mnm_config=mnm_config,
            chain_config=chain_config,
            num_epochs=2,
            max_steps_per_epoch=2,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=true,
            val_texts=val_texts,
            val_interval=1
        )
        
        # Verify best checkpoint creation
        best_path = joinpath(checkpoint_dir, "best.jld2")
        @test isfile(best_path)
        println("Found best checkpoint: $best_path")
        
        # Verify CSV log contains val_factscore column
        files = readdir(checkpoint_dir)
        csv_files = filter(f -> endswith(f, ".csv"), files)
        @test length(csv_files) == 1
        
        csv_path = joinpath(checkpoint_dir, csv_files[1])
        lines = readlines(csv_path)
        header = lines[1]
        @test contains(header, "val_factscore")
        
        # Check last line (epoch summary)
        # It should have a value for val_factscore (not NaN for epoch summary if it ran)
        # Step -1 lines are epoch summaries
        epoch_lines = filter(l -> contains(l, ",-1,"), lines)
        @test length(epoch_lines) >= 1
        
        last_epoch = split(epoch_lines[end], ",")
        val_score = parse(Float64, last_epoch[end])
        println("Validation score from log: $val_score")
        @test val_score >= 0.0 # Should be a valid number
        
    finally
        rm(test_dir, recursive=true, force=true)
    end
end
