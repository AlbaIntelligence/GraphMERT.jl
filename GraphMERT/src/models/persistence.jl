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

using JSON
using JLD2
using Flux
using Dates

"""
    save_model(model::GraphMERTModel, save_path::String; include_config::Bool=true, include_optimizer_state::Bool=false)::Bool

Save model weights and configuration to a JLD2 file.
Uses `Flux.state` to serialize parameters in a framework-compatible way.

# Arguments
- `model::GraphMERTModel`: Model to save
- `save_path::String`: Output path (should end in .jld2)
- `include_config::Bool`: Whether to save configuration (default: true)
- `include_optimizer_state::Bool`: (Not yet implemented)

# Returns
- `Bool`: Success status
"""
function save_model(
    model::GraphMERT.GraphMERTModel,
    save_path::String;
    include_config::Bool = true,
    include_optimizer_state::Bool = false,
)::Bool
    try
        # Ensure directory exists
        dir_path = dirname(save_path)
        if !isdir(dir_path)
            mkpath(dir_path)
        end

        # Get model state (parameters + buffers)
        # Note: Flux.state returns a named tuple structure that JLD2 can serialize directly
        model_state = Flux.state(model)

        # Prepare dictionary to save
        # We explicitely separate complex structures to avoid type inference issues
        data = Dict{String,Any}(
            "model_type" => "GraphMERT",
            "version" => "1.0"
        )

        if include_config
            data["config"] = model.config
        end

        # Save using JLD2.save which handles Dict{String, Any} correctly
        data["model_state"] = model_state
        JLD2.save(save_path, data)

        @info "Model saved successfully to: $save_path"
        return true
    catch e
        @error "Failed to save model to $save_path: $e"
        return false
    end
end

"""
    load_model(load_path::String; device::Symbol=:cpu, strict::Bool=true)::Union{GraphMERTModel, Nothing}

Load model from a JLD2 checkpoint.
Reconstructs the model structure from the saved config, then loads weights using `Flux.loadmodel!`.

# Arguments
- `load_path::String`: Path to .jld2 checkpoint
- `device::Symbol`: Target device (currently only :cpu supported)
- `strict::Bool`: Whether to fail on missing keys (passed to loadmodel!)

# Returns
- `GraphMERTModel` or `Nothing`
"""
function load_model(
    load_path::String;
    device::Symbol = :cpu,
    strict::Bool = true,
)::Union{GraphMERTModel,Nothing}
    # 1. Handle directory path (legacy/HF style) - fallback for non-JLD2 paths
    if isdir(load_path) || endswith(load_path, "json")
        # Existing logic for JSON checkpoints/directories (kept for backward compat)
        return _load_legacy_json_model(load_path)
    end

    # 2. JLD2 Loading
    try
        if !isfile(load_path)
            @error "Model file not found: $load_path"
            return nothing
        end

        # Load data from JLD2
        # JLD2.load returns a Dict{String, Any}
        data = JLD2.load(load_path)

        # Validate type
        if get(data, "model_type", "") != "GraphMERT"
            @warn "Checkpoint at $load_path does not claim to be 'GraphMERT'. Proceeding with caution."
        end

        # Reconstruct model from config
        if !haskey(data, "config")
            @error "Checkpoint missing 'config'. Cannot reconstruct architecture."
            return nothing
        end
        
        config = data["config"]
        model = GraphMERTModel(config)

        # Load weights
        if haskey(data, "model_state")
            model_state = data["model_state"]
            Flux.loadmodel!(model, model_state)
            @info "Model weights restored from $load_path"
        else
            @warn "Checkpoint missing 'model_state'. Returning uninitialized model."
        end

        if device == :gpu
            # model = gpu(model) # specific gpu logic if needed
        end

        return model

    catch e
        @error "Failed to load JLD2 model from $load_path: $e"
        return nothing
    end
end

# Internal helper for the old JSON logic
function _load_legacy_json_model(load_path::String)
    try
        checkpoint_file = load_path
        if isdir(load_path)
            checkpoint_file = joinpath(load_path, "checkpoint.json")
            if !isfile(checkpoint_file) && isfile(joinpath(load_path, "config.json"))
                @info "Using encoder directory (no checkpoint.json); building default model for: $load_path"
                return GraphMERTModel(GraphMERTConfig())
            end
        end
        
        if !isfile(checkpoint_file)
            @error "Legacy checkpoint not found: $checkpoint_file"
            return nothing
        end

        checkpoint_data = JSON.parsefile(checkpoint_file)
        if haskey(checkpoint_data, "config")
             # JSON parsing of structs is tricky; this is a simplified fallback
             # Ideally we convert dict -> struct here. For now returning default.
             @warn "JSON checkpoint loading is deprecated. Returning default initialized model."
             return GraphMERTModel(GraphMERTConfig())
        end
        return nothing
    catch e
         @error "Legacy load failed: $e"
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
