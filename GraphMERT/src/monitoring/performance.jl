"""
Performance monitoring for GraphMERT training

This module provides utilities for monitoring training progress,
logging metrics, and tracking performance during GraphMERT training.

Features:
- Training progress logging with detailed metrics
- Performance monitoring and profiling
- Memory usage tracking
- Training speed metrics
- Loss curve visualization data
- Checkpoint management
- Evaluation metrics logging
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
function log_training_progress(
    epoch::Int,
    step::Int,
    total_loss::Float64,
    mnm_loss::Float64,
    mlm_loss::Float64,
)
    @info "Epoch $epoch, Step $step: Loss = $total_loss (MLM: $mlm_loss, MNM: $mnm_loss)"
end

"""
    log_evaluation_metrics(metrics::Dict{String, Float64})

Log evaluation metrics after training epoch.

# Arguments
- `metrics::Dict{String, Float64}`: Dictionary of evaluation metrics
"""
function log_evaluation_metrics(metrics::Dict{String,Float64})
    @info "Evaluation Metrics" metrics = metrics
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
function create_checkpoint_filename(epoch::Int, step::Int = 0)::String
    if step > 0
        return "graphmert_epoch$(epoch)_step$(step).jld2"
    else
        return "graphmert_epoch$(epoch).jld2"
    end
end

"""
    TrainingMetrics

Structure to hold training metrics and performance data.
"""
mutable struct TrainingMetrics
    epoch::Int
    step::Int
    total_loss::Float64
    mnm_loss::Float64
    mlm_loss::Float64
    learning_rate::Float64
    memory_usage::Float64
    step_time::Float64
    tokens_processed::Int
    throughput::Float64
    timestamp::DateTime
end

"""
    PerformanceMonitor

Monitor for tracking training performance and metrics.
"""
mutable struct PerformanceMonitor
    metrics_history::Vector{TrainingMetrics}
    start_time::DateTime
    start_ns::UInt64
    current_epoch::Int
    current_step::Int
    total_steps::Int
    total_tokens::Int
    best_loss::Float64
    patience_counter::Int
    max_patience::Int
end

"""
    create_performance_monitor(total_steps::Int, max_patience::Int=10)::PerformanceMonitor

Create a new performance monitor.

# Arguments
- `total_steps::Int`: Total number of training steps
- `max_patience::Int`: Maximum patience for early stopping

# Returns
- `PerformanceMonitor`: New performance monitor instance
"""
function create_performance_monitor(
    total_steps::Int,
    max_patience::Int = 10,
)::PerformanceMonitor
    return PerformanceMonitor(
        TrainingMetrics[],
        now(),
        time_ns(),
        0,
        0,
        total_steps,
        0,
        Inf,
        0,
        max_patience,
    )
end

"""
    update_training_metrics(monitor::PerformanceMonitor, epoch::Int, step::Int,
                           total_loss::Float64, mnm_loss::Float64, mlm_loss::Float64,
                           learning_rate::Float64, step_time::Float64, tokens_processed::Int)

Update training metrics in the performance monitor.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor to update
- `epoch::Int`: Current epoch
- `step::Int`: Current step
- `total_loss::Float64`: Combined loss
- `mnm_loss::Float64`: MNM loss
- `mlm_loss::Float64`: MLM loss
- `learning_rate::Float64`: Current learning rate
- `step_time::Float64`: Time for this step in seconds
"""
function update_training_metrics(
    monitor::PerformanceMonitor,
    epoch::Int,
    step::Int,
    total_loss::Float64,
    mnm_loss::Float64,
    mlm_loss::Float64,
    learning_rate::Float64,
    step_time::Float64,
    tokens_processed::Int,
)
    # Calculate memory usage (simplified)
    memory_usage = Base.gc_live_bytes() / 1024^2  # MB

    # Calculate throughput from actual tokens and time (tokens per second)
    throughput = tokens_processed > 0 && step_time > 0 ? tokens_processed / step_time : 0.0

    # Create metrics entry
    metrics = TrainingMetrics(
        epoch,
        step,
        total_loss,
        mnm_loss,
        mlm_loss,
        learning_rate,
        memory_usage,
        step_time,
        tokens_processed,
        throughput,
        now(),
    )

    # Update monitor
    push!(monitor.metrics_history, metrics)
    monitor.current_epoch = epoch
    monitor.current_step = step
    monitor.total_tokens += max(tokens_processed, 0)

    # Update best loss
    if total_loss < monitor.best_loss
        monitor.best_loss = total_loss
        monitor.patience_counter = 0
    else
        monitor.patience_counter += 1
    end
end

"""
    log_detailed_progress(monitor::PerformanceMonitor)

Log detailed training progress with comprehensive metrics.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor
"""
function log_detailed_progress(monitor::PerformanceMonitor)
    if isempty(monitor.metrics_history)
        return
    end

    latest = monitor.metrics_history[end]
    # Monotonic elapsed time for stability
    elapsed_ns = time_ns() - monitor.start_ns

    progress_pct = round(latest.step / max(monitor.total_steps, 1) * 100, digits = 1)
    elapsed_sec = round(elapsed_ns / 1e9, digits = 3)
    total_loss = round(latest.total_loss, digits = 6)
    mnm_loss = round(latest.mnm_loss, digits = 6)
    mlm_loss = round(latest.mlm_loss, digits = 6)
    memory_mb = round(latest.memory_usage, digits = 3)
    step_time = round(latest.step_time, digits = 6)
    throughput = round(latest.throughput, digits = 3)
    best_loss = round(monitor.best_loss, digits = 6)
    patience_str = "$(monitor.patience_counter)/$(monitor.max_patience)"

    @info "Training Progress" epoch = latest.epoch step = latest.step total_steps =
        monitor.total_steps progress = "$progress_pct%" elapsed_time = "$elapsed_sec s" total_loss =
        total_loss mnm_loss = mnm_loss mlm_loss = mlm_loss learning_rate =
        latest.learning_rate memory_usage = "$(memory_mb) MB" step_time = "$(step_time) s" tokens_processed =
        latest.tokens_processed throughput = "$(throughput) tokens/s" best_loss = best_loss patience =
        patience_str
end

"""
    should_early_stop(monitor::PerformanceMonitor)::Bool

Check if training should stop early based on patience.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor

# Returns
- `Bool`: True if early stopping should be triggered
"""
function should_early_stop(monitor::PerformanceMonitor)::Bool
    return monitor.patience_counter >= monitor.max_patience
end

"""
    get_loss_curve(monitor::PerformanceMonitor)::Vector{Float64}

Get loss curve data for visualization.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor

# Returns
- `Vector{Float64}`: Loss values over time
"""
function get_loss_curve(monitor::PerformanceMonitor)::Vector{Float64}
    return [m.total_loss for m in monitor.metrics_history]
end

"""
    get_memory_usage(monitor::PerformanceMonitor)::Vector{Float64}

Get memory usage over time.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor

# Returns
- `Vector{Float64}`: Memory usage values over time
"""
function get_memory_usage(monitor::PerformanceMonitor)::Vector{Float64}
    return [m.memory_usage for m in monitor.metrics_history]
end

"""
    get_throughput(monitor::PerformanceMonitor)::Vector{Float64}

Get training throughput over time.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor

# Returns
- `Vector{Float64}`: Throughput values over time
"""
function get_throughput(monitor::PerformanceMonitor)::Vector{Float64}
    return [m.throughput for m in monitor.metrics_history]
end

"""
    log_training_summary(monitor::PerformanceMonitor)

Log comprehensive training summary.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor
"""
function log_training_summary(monitor::PerformanceMonitor)
    if isempty(monitor.metrics_history)
        @warn "No training metrics available"
        return
    end

    # Monotonic elapsed
    total_elapsed_sec = round((time_ns() - monitor.start_ns) / 1e9, digits = 3)
    loss_curve = get_loss_curve(monitor)
    memory_usage = get_memory_usage(monitor)
    throughput = get_throughput(monitor)

    final_loss = round(loss_curve[end], digits = 6)
    best_loss = round(monitor.best_loss, digits = 6)
    avg_memory = round(mean(memory_usage), digits = 3)
    peak_memory = round(maximum(memory_usage), digits = 3)
    avg_throughput = round(mean(throughput), digits = 3)
    peak_throughput = round(maximum(throughput), digits = 3)
    loss_improvement = round(
        (loss_curve[1] - loss_curve[end]) / max(loss_curve[1], eps()) * 100,
        digits = 3,
    )

    @info "Training Summary" total_epochs = monitor.current_epoch total_steps =
        monitor.current_step total_tokens = monitor.total_tokens total_time = "$(total_elapsed_sec) s" final_loss =
        final_loss best_loss = best_loss average_memory = "$(avg_memory) MB" peak_memory = "$(peak_memory) MB" average_throughput = "$(avg_throughput) tokens/s" peak_throughput = "$(peak_throughput) tokens/s" loss_improvement = "$(loss_improvement)%"
end

"""
    save_training_log(monitor::PerformanceMonitor, log_path::String)

Save training metrics to file.

# Arguments
- `monitor::PerformanceMonitor`: Performance monitor
- `log_path::String`: Path to save log file
"""
function save_training_log(monitor::PerformanceMonitor, log_path::String)
    mkpath(dirname(log_path))

    open(log_path, "w") do io
        write(io, "GraphMERT Training Log\n")
        write(io, "====================\n\n")
        write(io, "Training started: $(monitor.start_time)\n")
        write(io, "Total epochs: $(monitor.current_epoch)\n")
        write(io, "Total steps: $(monitor.current_step)\n")
        write(io, "Best loss: $(monitor.best_loss)\n\n")

        write(
            io,
            "Epoch,Step,Total_Loss,MNM_Loss,MLM_Loss,Learning_Rate,Memory_MB,Step_Time_s,Throughput_tokens_s\n",
        )

        for metrics in monitor.metrics_history
            write(
                io,
                "$(metrics.epoch),$(metrics.step),$(metrics.total_loss),$(metrics.mnm_loss),$(metrics.mlm_loss),$(metrics.learning_rate),$(metrics.memory_usage),$(metrics.step_time),$(metrics.throughput)\n",
            )
        end
    end

    @info "Training log saved to: $log_path"
end

# Export functions for external use
export TrainingMetrics,
    PerformanceMonitor,
    create_performance_monitor,
    update_training_metrics,
    log_detailed_progress,
    should_early_stop,
    get_loss_curve,
    get_memory_usage,
    get_throughput,
    log_training_summary,
    save_training_log,
    log_training_progress,
    log_evaluation_metrics,
    create_checkpoint_filename
