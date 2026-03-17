"""
Training logging utilities for GraphMERT.jl.
Handles CSV logging and console output.
"""

using Dates
using Printf

"""
    TrainingLogger

Handles logging of training metrics to console and CSV.
"""
struct TrainingLogger
    log_dir::String
    csv_path::String
    io::IOStream
    start_time::DateTime
end

"""
    create_training_logger(log_dir::String, experiment_name::String="train")

Create a new training logger that writes to a CSV file in the specified directory.
"""
function create_training_logger(log_dir::String, experiment_name::String="train")
    if !isdir(log_dir)
        mkpath(log_dir)
    end
    
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    csv_path = joinpath(log_dir, "$(experiment_name)_$(timestamp).csv")
    
    # Initialize CSV file with header
    io = open(csv_path, "w")
    write(io, "epoch,step,combined_loss,mnm_loss,mlm_loss,elapsed_seconds,learning_rate,val_factscore\n")
    flush(io)
    
    @info "Logging training metrics to: $csv_path"
    
    return TrainingLogger(
        log_dir,
        csv_path,
        io,
        now()
    )
end

"""
    log_metrics(logger::TrainingLogger, epoch::Int, step::Int, 
               combined_loss::Real, mnm_loss::Real, mlm_loss::Real, 
               learning_rate::Real=0.0)

Log metrics for a training step.
"""
function log_metrics(logger::TrainingLogger, epoch::Int, step::Int, 
                     combined_loss::Real, mnm_loss::Real, mlm_loss::Real,
                     learning_rate::Real=0.0)
    elapsed = (now() - logger.start_time).value / 1000.0
    
    # Log to CSV (val_factscore is NaN for step logs)
    write(logger.io, "$epoch,$step,$combined_loss,$mnm_loss,$mlm_loss,$elapsed,$learning_rate,NaN\n")
    flush(logger.io)
    
    # Log to console (brief)
    @printf("Epoch %d, Step %d: Loss = %.4f (MNM: %.4f, MLM: %.4f) [%.1fs]\n", 
            epoch, step, combined_loss, mnm_loss, mlm_loss, elapsed)
end

"""
    log_epoch_summary(logger::TrainingLogger, epoch::Int, avg_loss::Real, 
                      avg_mnm::Real, avg_mlm::Real, checkpoint_path::String="",
                      val_score::Real=NaN)

Log summary statistics for an epoch.
"""
function log_epoch_summary(logger::TrainingLogger, epoch::Int, avg_loss::Real, 
                           avg_mnm::Real, avg_mlm::Real, checkpoint_path::String="",
                           val_score::Real=NaN)
    elapsed = (now() - logger.start_time).value / 1000.0
    
    # Log epoch summary to CSV as step -1
    write(logger.io, "$epoch,-1,$avg_loss,$avg_mnm,$avg_mlm,$elapsed,0.0,$val_score\n")
    flush(logger.io)
    
    msg = @sprintf("Epoch %d Completed: Avg Loss = %.4f (MNM: %.4f, MLM: %.4f)", 
                   epoch, avg_loss, avg_mnm, avg_mlm)
    
    if !isnan(val_score)
        msg *= @sprintf(" | Val FActScore*: %.4f", val_score)
    end
    
    if !isempty(checkpoint_path)
        msg *= " | Checkpoint: $checkpoint_path"
    end
    
    @info msg
end

"""
    close_logger(logger::TrainingLogger)

Close the logger file handle.
"""
function close_logger(logger::TrainingLogger)
    if isopen(logger.io)
        close(logger.io)
    end
end
