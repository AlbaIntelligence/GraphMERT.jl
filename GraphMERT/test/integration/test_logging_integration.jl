using Test
using GraphMERT
using Flux
using Dates

@testset "Logging Integration" begin
    # Create temp dir
    test_dir = mktempdir()
    checkpoint_dir = joinpath(test_dir, "checkpoints")
    
    try
        # Create small config
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
        
        # Mock data
        train_texts = ["test sentence one", "test sentence two"]
        
        # Create mlm config with matching vocab size and valid mask token
        mlm_config = GraphMERT.MLMConfig(vocab_size=100, mask_token_id=99)
        
        # Create mnm config with matching vocab size and valid mask token
        mnm_config = GraphMERT.MNMConfig(vocab_size=100, mask_token_id=99)
        
        # Create compatible chain graph config
        chain_config = GraphMERT.ChainGraphConfig(
            num_roots=8,
            num_leaves_per_root=2,
            max_sequence_length=64, # Must match max_position_embeddings
            vocab_size=100
        )
        
        # Run training for 1 epoch, 2 steps
        # This will trigger log creation and writing
        model = train_graphmert(
            train_texts,
            config;
            mlm_config=mlm_config,
            mnm_config=mnm_config,
            chain_config=chain_config,
            num_epochs=1,
            max_steps_per_epoch=2,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=false
        )
        
        # Verify log file creation
        files = readdir(checkpoint_dir)
        csv_files = filter(f -> endswith(f, ".csv"), files)
        @test length(csv_files) == 1
        
        csv_path = joinpath(checkpoint_dir, csv_files[1])
        println("Found log file: $csv_path")
        
        # Verify CSV content
        lines = readlines(csv_path)
        @test length(lines) >= 3 # Header + 2 steps
        
        header = lines[1]
        @test header == "epoch,step,combined_loss,mnm_loss,mlm_loss,elapsed_seconds,learning_rate"
        
        # Parse first data line
        data = split(lines[2], ",")
        @test length(data) == 7
        @test parse(Int, data[1]) == 1 # epoch
        @test parse(Int, data[2]) == 1 # step
        
        println("Log file content verified successfully")
        
    finally
        rm(test_dir, recursive=true, force=true)
    end
end
