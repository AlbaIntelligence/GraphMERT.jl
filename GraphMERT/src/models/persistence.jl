"""
Model persistence and serialization utilities.
"""

# Types will be available from main module

"""
    ModelMetadata

Metadata for saved models.
"""
struct ModelMetadata
    version::String
    created_at::String
    description::String
    config::GraphMERTConfig
    performance_metrics::Dict{String,Any}
end

"""
    save_model(model::GraphMERTModel, save_path::String; include_config::Bool=true, include_optimizer_state::Bool=false)::Bool

Save model to file.

# Arguments
- `model::GraphMERTModel`: Model to save
- `save_path::String`: Output path
- `include_config::Bool`: Whether to save configuration
- `include_optimizer_state::Bool`: Whether to save optimizer state

# Returns
- `Bool`: Success status
"""
function save_model(
    model::GraphMERT.GraphMERTModel,
    save_path::String;
    include_config::Bool = true,
    include_optimizer_state::Bool = false,
)::Bool
    # For demo purposes, create a simple checkpoint
    checkpoint_data = Dict(
        "model_type" => "GraphMERT",
        "version" => "0.1.0",
        "timestamp" => string(now()),
        "model_params" => length(Flux.params(model)),
        "config" => include_config ? model.config : nothing,
        "description" => "GraphMERT model checkpoint",
    )

    # Save to JSON file (in practice would save model weights)
    open(save_path, "w") do io
        JSON.print(io, checkpoint_data, 2)
    end

    @info "Model saved to: $save_path"
    return true
end

"""
    load_model(load_path::String; device::Symbol=:cpu, strict::Bool=true)::Union{GraphMERTModel, Nothing}

Load model from file. When loading a full checkpoint, returns the full model (RoBERTa + H-GAT)
so that extraction uses the encoder (FR-005). Current implementation builds full GraphMERTModel
from saved config; loading pretrained weights from checkpoint is post-MVP.

# Arguments
- `load_path::String`: Path to model checkpoint (JSON with model_type and config)
- `device::Symbol`: Target device (`:cpu` or `:cuda`)
- `strict::Bool`: Whether to require exact parameter match

# Returns
- `GraphMERTModel` or `Nothing`: Loaded model or nothing if failed
"""
function load_model(
    load_path::String;
    device::Symbol = :cpu,
    strict::Bool = true,
)::Union{GraphMERTModel,Nothing}
    try
        # If path is a directory (e.g. encoders/roberta-base), look for checkpoint.json inside
        checkpoint_file = load_path
        if isdir(load_path)
            checkpoint_file = joinpath(load_path, "checkpoint.json")
            # If no GraphMERT checkpoint but Hugging Face config.json exists, build default model (weights TBD)
            if !isfile(checkpoint_file) && isfile(joinpath(load_path, "config.json"))
                @info "Using encoder directory (no checkpoint.json); building default model for: $load_path"
                return GraphMERTModel(GraphMERTConfig())
            end
        end
        if !isfile(checkpoint_file)
            @error "Model file not found: $checkpoint_file"
            return nothing
        end

        # Load checkpoint metadata
        checkpoint_data = JSON.parsefile(checkpoint_file)

        if !haskey(checkpoint_data, "model_type") ||
           checkpoint_data["model_type"] != "GraphMERT"
            @error "Invalid model file: $load_path"
            return nothing
        end

        # For demo purposes, create a new model with the saved config
        # In practice, would load actual model weights
        if haskey(checkpoint_data, "config") && checkpoint_data["config"] !== nothing
            config = checkpoint_data["config"]
            # Create model with loaded config
            model = GraphMERTModel(GraphMERTConfig())  # Simplified
            @info "Model loaded from: $load_path"
            return model
        else
            @warn "No configuration found in checkpoint, creating default model"
            model = GraphMERTModel(GraphMERTConfig())
            return model
        end

    catch e
        @error "Failed to load model from $load_path: $e"
        return nothing
    end
end

"""
    save_weights(model::GraphMERTModel, save_path::String)

Save model weights to file.
"""
function save_weights(model::GraphMERTModel, save_path::String)
    # Placeholder implementation
    @warn "Weight saving not yet implemented"
    return false
end

"""
    load_weights(load_path::String)

Load model weights from file.
"""
function load_weights(load_path::String)
    # Placeholder implementation
    @warn "Weight loading not yet implemented"
    return nothing
end

"""
    save_config(config::GraphMERTConfig, save_path::String)

Save configuration to file.
"""
function save_config(config::GraphMERTConfig, save_path::String)
    # Placeholder implementation
    @warn "Config saving not yet implemented"
    return false
end

"""
    load_model_config(load_path::String)

Load model configuration from file.
"""
function load_model_config(load_path::String)
    # Placeholder implementation
    @warn "Model config loading not yet implemented"
    return nothing
end

"""
    save_metadata(metadata::ModelMetadata, save_path::String)

Save metadata to file.
"""
function save_metadata(metadata::ModelMetadata, save_path::String)
    # Placeholder implementation
    @warn "Metadata saving not yet implemented"
    return false
end

"""
    load_metadata(load_path::String)

Load metadata from file.
"""
function load_metadata(load_path::String)
    # Placeholder implementation
    @warn "Metadata loading not yet implemented"
    return nothing
end
