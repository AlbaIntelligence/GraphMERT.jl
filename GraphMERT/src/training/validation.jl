"""
Validation logic for GraphMERT training.
"""

using Statistics

"""
    validate_model(model::GraphMERTModel, val_texts::Vector{String}, config::GraphMERTConfig;
                  domain::String="biomedical",
                  confidence_threshold::Float64=0.5)

Run validation on held-out texts by extracting a KG and computing FActScore*.

# Arguments
- `model::GraphMERTModel`: Model to evaluate
- `val_texts::Vector{String}`: Validation text corpus
- `config::GraphMERTConfig`: Model configuration
- `domain::String`: Domain for extraction (default: "biomedical")
- `confidence_threshold::Float64`: Confidence threshold for extraction (default: 0.5)

# Returns
- `Tuple{Float64, Any}`: (factscore, full_result)
"""
function validate_model(
    model::GraphMERTModel,
    val_texts::Vector{String},
    config::GraphMERTConfig;
    domain::String="biomedical",
    confidence_threshold::Float64=0.5
)
    if isempty(val_texts)
        return 0.0, nothing
    end

    # Combine validation texts
    # In a real scenario, we might want to process them in batches or document-by-document
    # But extract_knowledge_graph takes a single string.
    full_text = join(val_texts, "\n\n")
    
    # Create options
    options = GraphMERT.ProcessingOptions(
        domain=domain,
        confidence_threshold=confidence_threshold
    )
    
    # Run extraction
    # Note: We assume extract_knowledge_graph is available (late binding)
    @info "Running validation extraction on $(length(val_texts)) documents..."
    kg = GraphMERT.extract_knowledge_graph(full_text, model; options=options)
    
    # Compute FActScore
    @info "Computing FActScore* for validation set..."
    result = GraphMERT.evaluate_factscore(
        kg, 
        full_text; 
        confidence_threshold=confidence_threshold,
        include_domain_metrics=false # Skip expensive domain metrics for training loop
    )
    
    @info "Validation FActScore*: $(result.factscore)"
    
    return result.factscore, result
end
