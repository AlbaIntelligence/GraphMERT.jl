"""
Training Pipeline for GraphMERT.jl

This module implements the complete training pipeline that combines:
- MLM (Masked Language Modeling) on syntactic space
- MNM (Masked Node Modeling) on semantic space
- Seed KG injection for vocabulary transfer
- Joint training with balanced loss weighting

The pipeline follows the paper's methodology for training GraphMERT models
with both syntactic and semantic objectives.
"""

using Random
using Statistics
using Dates

"""
    set_reproducible_seed(seed::Int=42)

Set random seed for reproducibility across Julia, Flux, and other libraries.

# Arguments
- `seed::Int`: Random seed value (default: 42)
"""
function set_reproducible_seed(seed::Int = 42)
    # Set Julia's global random seed
    Random.seed!(seed)

    # Set Flux's random seed if available
    if isdefined(Main, :Flux)
        try
            Flux.gpu(false)  # Ensure CPU mode for reproducibility
            # Flux doesn't have a global seed function, but we can set it per operation
        catch
            # Flux not available or error, continue
        end
    end

    # Set CUDA seed if available
    if isdefined(Main, :CUDA)
        try
            CUDA.seed!(seed)
        catch
            # CUDA not available or error, continue
        end
    end

    # Log the seed for debugging
    @info "Random seed set to $seed for reproducibility"
end

# Types and functions will be available from main module after all includes

# Training components are included in the same module namespace

function default_graphmert_config()
    roberta_config = GraphMERT.RoBERTaConfig(
        vocab_size = 30522,
        hidden_size = 512,
        num_hidden_layers = 12,
        num_attention_heads = 8,
        intermediate_size = 2048,
        max_position_embeddings = 1024,
        type_vocab_size = 2,
        layer_norm_eps = 1e-12,
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.1,
    )

    hgat_config = GraphMERT.HGATConfig(
        input_dim = 512,
        hidden_dim = 256,
        num_heads = 4,
        num_layers = 2,
        dropout_rate = 0.1,
        attention_dropout_rate = 0.1,
        layer_norm_eps = 1e-12,
        use_residual = true,
        use_layer_norm = true,
    )

    attention_config = GraphMERT.SpatialAttentionConfig(
        max_distance = 512,
        decay_lambda = 0.6,
        decay_p_init = 1.0,
        use_distance_bias = true,
        distance_bias_weight = 0.1,
    )

    return GraphMERT.GraphMERTConfig(
        roberta_config = roberta_config,
        hgat_config = hgat_config,
        attention_config = attention_config,
        max_sequence_length = 1024,
        hidden_dim = 512,
    )
end

"""
    prepare_training_data(texts::Vector{String}, seed_kg::Vector{SemanticTriple},
                         mlm_config::MLMConfig, mnm_config::MNMConfig,
                         injection_config::SeedInjectionConfig)

Prepare training data by creating leafy chain graphs and applying masking.

# Arguments
- `texts::Vector{String}`: Training text sequences
- `seed_kg::Vector{SemanticTriple}`: Seed knowledge graph for injection
- `mlm_config::MLMConfig`: MLM training configuration
- `mnm_config::MNMConfig`: MNM training configuration
- `injection_config::SeedInjectionConfig`: Seed injection configuration

# Returns
- `Tuple{Vector{LeafyChainGraph}, Vector{MNMBatch}, Vector{MLMBatch}}`: Prepared data
"""
function prepare_training_data(
texts::Vector{String},
seed_kg::Vector{GraphMERT.SemanticTriple},
mlm_config::GraphMERT.MLMConfig,
mnm_config::GraphMERT.MNMConfig,
injection_config::GraphMERT.SeedInjectionConfig,
)
# Step 1: Extract entities from texts (mock implementation)
all_triples = Vector{GraphMERT.SemanticTriple}()
    for text in texts
    # Simple entity extraction: split by spaces and find entities in seed KG
    words = split(lowercase(text), r"\W+")
    for word in words
    # Find triples where head matches this word
    matching_triples = filter(t -> occursin(lowercase(word), lowercase(t.head)), seed_kg)
    append!(all_triples, matching_triples)
end
    end

    # Remove duplicates
unique_triples = unique(all_triples)

# Step 2: Select triples for injection using our mock algorithm
selected_triples = GraphMERT.inject_seed_triples(unique_triples, injection_config.alpha_score_threshold)

# Step 3: Create leafy chain graphs
    graphs = Vector{LeafyChainGraph}()
for text in texts
# Simple tokenization
words = split(text, " ")
tokens = String[w for w in words[1:min(length(words), 128)]]  # Convert to String
token_ids = Int[hash(w) % 30522 for w in tokens]  # Mock tokenization, convert to Int

config = GraphMERT.default_chain_graph_config()
graph = GraphMERT.create_empty_chain_graph(token_ids, tokens, config)

    # Inject selected triples into graph (simplified - inject into first few roots)
        for (i, triple) in enumerate(selected_triples)
            if i <= config.num_roots
                root_idx = i - 1  # 0-based
                inject_triple!(graph, root_idx, 0, triple.tail_tokens, triple.tail, Symbol(triple.relation), triple.head)
            end
        end

        push!(graphs, graph)
    end

    # Step 4: Create MLM batch (simplified)
    batch_size = min(length(texts), 32)
    seq_len = mlm_config.max_length
    input_ids = zeros(Int, batch_size, seq_len)
    attention_mask = zeros(Int, batch_size, seq_len)
    labels = fill(-100, batch_size, seq_len)

    for i in 1:batch_size
    text = texts[i]
    words = split(text, " ")
    tokens = Int[hash(w) % mlm_config.vocab_size for w in words[1:min(length(words), seq_len)]]
    input_ids[i, 1:length(tokens)] = tokens
    attention_mask[i, 1:length(tokens)] .= 1
    end

    mlm_batch = GraphMERT.MLMBatch(input_ids, attention_mask, labels, Int[], Tuple{Int,Int}[])

    # Step 5: Create MNM batch (placeholder - would need proper implementation)
    mnm_batch = nothing  # TODO: Implement proper MNM batching

    return graphs, mnm_batch, mlm_batch
end

"""
    train_graphmert(train_texts::Vector{String}, config::GraphMERTConfig;
                   seed_kg::Union{Vector{SemanticTriple}, Nothing}=nothing,
                   mlm_config::MLMConfig=default_mlm_config(),
                   mnm_config::MNMConfig=default_mnm_config(),
                   injection_config::Union{SeedInjectionConfig, Nothing}=nothing,
                   num_epochs::Int=10, learning_rate::Float64=1e-4,
                   checkpoint_dir::String="checkpoints")::GraphMERTModel

Main training function for GraphMERT model.

Trains a GraphMERT model using joint MLM + MNM objectives with optional
seed KG injection for vocabulary transfer.

# Arguments
- `train_texts::Vector{String}`: Training text corpus
- `config::GraphMERTConfig`: Model configuration
- `seed_kg::Union{Vector{SemanticTriple}, Nothing}`: Optional seed KG for injection
- `mlm_config::MLMConfig`: MLM training configuration
- `mnm_config::MNMConfig`: MNM training configuration
- `injection_config::Union{SeedInjectionConfig, Nothing}`: Seed injection configuration
- `num_epochs::Int`: Number of training epochs
- `learning_rate::Float64`: Learning rate for optimizer
- `checkpoint_dir::String`: Directory for saving checkpoints

# Returns
- `GraphMERTModel`: Trained model

# Example
```julia
# Prepare training data
train_texts = load_pubmed_abstracts("diabetes_train.json")
seed_kg = load_umls_triples("seed_kg.json")

# Train model
model = train_graphmert(
    train_texts,
    default_graphmert_config(),
    seed_kg=seed_kg,
    num_epochs=10,
    checkpoint_dir="checkpoints/diabetes"
)
```
"""
function train_graphmert(
    train_texts::Vector{String},
    config::GraphMERTConfig;
    seed_kg::Union{Vector{GraphMERT.SemanticTriple},Nothing} = nothing,
    mlm_config::GraphMERT.MLMConfig = default_mlm_config(),
    mnm_config::GraphMERT.MNMConfig = default_mnm_config(),
    injection_config::Union{GraphMERT.SeedInjectionConfig,Nothing} = nothing,
    distillation_config::Union{Distillation.DistillationConfig,Nothing} = nothing,
    teacher_model::Union{GraphMERTModel,Nothing} = nothing,
    num_epochs::Int = 10,
    learning_rate::Float64 = 1e-4,
    checkpoint_dir::String = "checkpoints",
    random_seed::Int = 42,
    max_steps_per_epoch::Int = 10,
    chain_config::GraphMERT.ChainGraphConfig = GraphMERT.default_chain_graph_config(),
    μ::Float64 = 1.0,
    model::Union{GraphMERTModel,Nothing} = nothing,
    save_checkpoints::Bool = true,
    val_texts::Vector{String} = String[],
    val_interval::Int = 1,
)::GraphMERTModel

    set_reproducible_seed(random_seed)
    rng = Random.MersenneTwister(random_seed)

    if mlm_config.vocab_size != config.roberta_config.vocab_size
        @warn "MLM vocab_size != RoBERTa vocab_size; using RoBERTa vocab_size for tokenization" mlm_vocab=mlm_config.vocab_size roberta_vocab=config.roberta_config.vocab_size
    end

    if model === nothing
        model = create_graphmert_model(config)
    end

    optimizer = Flux.Adam(learning_rate)

    if seed_kg !== nothing || injection_config !== nothing
        @warn "Seed KG injection is not yet wired into the real training loop; training proceeds without injection" has_seed_kg = (seed_kg !== nothing) has_injection_config = (injection_config !== nothing)
    end

    if distillation_config !== nothing && distillation_config.enabled && teacher_model === nothing
         @warn "Distillation enabled in config but no teacher model provided. Distillation will be skipped."
    end

    logger = create_training_logger(checkpoint_dir)
    best_val_score = -1.0

    for epoch = 1:num_epochs
        println("Epoch $epoch/$num_epochs")

        shuffled_texts = shuffle(rng, train_texts)
        steps = min(max_steps_per_epoch, length(shuffled_texts))

        total_loss = 0.0
        total_mlm_loss = 0.0
        total_mnm_loss = 0.0

        for step in 1:steps
            text = shuffled_texts[step]
            toks = split(text)
            toks = toks[1:min(length(toks), chain_config.num_roots)]

            vocab_size = config.roberta_config.vocab_size
            token_ids = Int[]
            for t in toks
                if vocab_size > 4
                    push!(token_ids, 4 + Int(mod(hash(t), vocab_size - 4)))
                else
                    push!(token_ids, 1)
                end
            end

            graph = create_empty_chain_graph(token_ids, String[t for t in toks], chain_config)

            step_total, step_mlm, step_mnm, step_dist = train_joint_mlm_mnm_step(
                model,
                graph,
                mlm_config,
                mnm_config,
                optimizer,
                μ,
                rng;
                distillation_config=distillation_config,
                teacher_model=teacher_model
            )

            total_loss += step_total
            total_mlm_loss += step_mlm
            total_mnm_loss += step_mnm

            # Log metrics per step
            log_metrics(logger, epoch, step, step_total, step_mnm, step_mlm, step_dist, learning_rate)
        end

        avg_loss = 0.0
        avg_mlm = 0.0
        avg_mnm = 0.0
        
        if steps > 0
            avg_loss = total_loss / steps
            avg_mlm = total_mlm_loss / steps
            avg_mnm = total_mnm_loss / steps
        end
        
        # Validation
        val_score = NaN
        if !isempty(val_texts) && epoch % val_interval == 0
            # Use biomedical domain by default for now
            # TODO: make domain configurable or infer from config
            v_score, _ = validate_model(model, val_texts, config; domain="biomedical")
            val_score = v_score
            
            if val_score > best_val_score
                best_val_score = val_score
                if save_checkpoints && !isempty(checkpoint_dir)
                    path = joinpath(checkpoint_dir, "best.jld2")
                    save_training_checkpoint(model, path, optimizer)
                    @info "New best model saved (FActScore*: $best_val_score): $path"
                end
            end
        end
        
        checkpoint_path_str = ""
        if save_checkpoints && !isempty(checkpoint_dir) && epoch % 2 == 0
            path = joinpath(checkpoint_dir, "graphmert_epoch$(epoch).jld2")
            save_training_checkpoint(model, path, optimizer)
            checkpoint_path_str = path
        end
        
        log_epoch_summary(logger, epoch, avg_loss, avg_mnm, avg_mlm, checkpoint_path_str, val_score)
    end

    close_logger(logger)
    return model
end

"""
    create_training_configurations()::Tuple{GraphMERTConfig, MLMConfig, MNMConfig, SeedInjectionConfig}

Create default training configurations for GraphMERT.

# Returns
- `Tuple{GraphMERTConfig, MLMConfig, MNMConfig, SeedInjectionConfig}`: Default configurations
"""


function create_training_configurations()::Tuple{
    GraphMERTConfig,
    GraphMERT.MLMConfig,
    GraphMERT.MNMConfig,
    GraphMERT.SeedInjectionConfig,
    Distillation.DistillationConfig,
}
    # Model configuration (80M parameters for laptop deployment)
    roberta_config = RoBERTaConfig(
        vocab_size = 30522,
        hidden_size = 512,
        num_hidden_layers = 12,
        num_attention_heads = 8,
        intermediate_size = 2048,
        max_position_embeddings = 1024,
        type_vocab_size = 2,
        layer_norm_eps = 1e-12,
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.1,
    )

    hgat_config = HGATConfig(
        input_dim = 512,
        hidden_dim = 256,
        num_heads = 4,
        num_layers = 2,
        dropout_rate = 0.1,
        attention_dropout_rate = 0.1,
        layer_norm_eps = 1e-12,
        use_residual = true,
        use_layer_norm = true,
    )

    attention_config = SpatialAttentionConfig(
        max_distance = 512,
        decay_lambda = 0.6,
        decay_p_init = 1.0,
        use_distance_bias = true,
        distance_bias_weight = 0.1,
    )

    model_config = GraphMERTConfig(
        roberta_config = roberta_config,
        hgat_config = hgat_config,
        attention_config = attention_config,
        entity_types = ["DISEASE", "DRUG", "PROTEIN", "SYMPTOM", "BIOMARKER"],
        relation_types = ["TREATS", "CAUSES", "ASSOCIATED_WITH", "INDICATES", "PREVENTS"],
        max_sequence_length = 1024,
        hidden_dim = 512,
    )

    # MLM configuration
    mlm_config = default_mlm_config()

    # MNM configuration
    mnm_config = default_mnm_config()

    # Seed injection configuration
    injection_config = SeedInjectionConfig(
        0.5,  # entity_linking_threshold
        10,   # top_k_candidates
        40,   # top_n_triples_per_entity
        0.7,  # alpha_score_threshold
        10,   # score_bucket_size
        5,    # relation_bucket_size
        0.2,  # injection_ratio
        10,    # max_triples_per_sequence
    )
    
    distillation_config = Distillation.DistillationConfig(enabled=false)

    return model_config, mlm_config, mnm_config, injection_config, distillation_config
end

"""
    load_training_data(data_path::String)::Tuple{Vector{String}, Vector{SemanticTriple}}

Load training data and seed knowledge graph.

# Arguments
- `data_path::String`: Path to training data file

# Returns
- `Tuple{Vector{String}, Vector{SemanticTriple}}`: Training texts and seed KG
"""
function load_training_data(data_path::String)::Tuple{Vector{String},Vector{SemanticTriple}}
    # Placeholder implementation - would load actual data
    # For demo, return sample data
    train_texts = [
        "Diabetes mellitus is a chronic metabolic disorder.",
        "Metformin is commonly used to treat type 2 diabetes.",
        "Insulin resistance is a key feature of type 2 diabetes.",
    ]

    seed_kg = [
        SemanticTriple(
            "diabetes",
            "C0011849",
            "treats",
            "metformin",
            [2156, 23421],
            0.95,
            "UMLS",
        ),
        SemanticTriple(
            "metformin",
            "C0025598",
            "inhibits",
            "diabetes",
            [23421, 2156],
            0.92,
            "UMLS",
        ),
    ]

    return train_texts, seed_kg
end

"""
    save_training_checkpoint(model::GraphMERTModel, checkpoint_path::String, optimizer=nothing)

Save model checkpoint to disk, including optimizer state.
Creates versioned checkpoints and updates 'latest' symlink.

# Arguments
- `model::GraphMERTModel`: Model to save
- `checkpoint_path::String`: Path to save checkpoint
- `optimizer`: Optional optimizer state to save
"""
function save_training_checkpoint(model::GraphMERTModel, checkpoint_path::String, optimizer=nothing)
    # Create directory if it doesn't exist
    dir_path = dirname(checkpoint_path)
    mkpath(dir_path)

    # Save using persistence module
    success = GraphMERT.save_model(
        model, 
        checkpoint_path; 
        include_config=true, 
        optimizer=optimizer, 
        include_optimizer_state=(optimizer!==nothing)
    )

    if success
        println("Saved checkpoint: $checkpoint_path")
        
        # Create/Update 'latest.jld2' symlink/copy
        latest_path = joinpath(dir_path, "latest.jld2")
        try
            # On some systems symlinks might fail or be tricky, copy is safer for now
            # or just write a small file pointing to it?
            # Let's try to copy to keep it simple and robust
            cp(checkpoint_path, latest_path; force=true)
            println("Updated latest checkpoint: $latest_path")
        catch e
            @warn "Failed to update latest checkpoint symlink/copy: $e"
        end
    end
end



# Export functions for external use
export train_graphmert,
    prepare_training_data,
    create_training_configurations,
    load_training_data,
    save_training_checkpoint,
    set_reproducible_seed
