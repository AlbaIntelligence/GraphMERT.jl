"""
Performance monitoring for GraphMERT training

This module provides utilities for monitoring training progress,
logging metrics, and tracking performance during GraphMERT training.
"""

"""
    log_training_progress(epoch::Int, step::Int, total_loss::Float64,
                         mnm_loss::Float64, mlm_loss::Float64)

Log training progress during training.

# Arguments
- `epoch::Int`: Current epoch number
- `step::Int`: Current step within epoch
- `total_loss::Float64`: Combined MLM + MNM loss
- `mnm_loss::Float64`: MNM loss component
- `mlm_loss::Float64`: MLM loss component
"""
function log_training_progress(epoch::Int, step::Int, total_loss::Float64,
  mnm_loss::Float64, mlm_loss::Float64)
  @info "Epoch $epoch, Step $step: Loss = $total_loss (MLM: $mlm_loss, MNM: $mnm_loss)"
end

"""
    log_evaluation_metrics(metrics::Dict{String, Float64})

Log evaluation metrics after training epoch.

# Arguments
- `metrics::Dict{String, Float64}`: Dictionary of evaluation metrics
"""
function log_evaluation_metrics(metrics::Dict{String,Float64})
  @info "Evaluation Metrics:" metrics...
end

"""
    create_checkpoint_filename(epoch::Int, step::Int=0)::String

Create standardized checkpoint filename.

# Arguments
- `epoch::Int`: Epoch number
- `step::Int`: Step number (optional)

# Returns
- `String`: Checkpoint filename
"""
function create_checkpoint_filename(epoch::Int, step::Int=0)::String
  if step > 0
    return "graphmert_epoch$(epoch)_step$(step).jld2"
  else
    return "graphmert_epoch$(epoch).jld2"
  end
end