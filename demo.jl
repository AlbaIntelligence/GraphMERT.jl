#!/usr/bin/env julia

"""
GraphMERT.jl End-to-End Demo
"""

using Pkg
Pkg.activate(".")

using GraphMERT
using Dates
using Logging

# Set logging to Info to see pipeline progress
global_logger(ConsoleLogger(stderr, Logging.Info))

println("=== GraphMERT.jl Demo ===")
println("Powered by GraphMERT (arXiv:2510.09580)")
println("Running in: $(pwd())")
println()

# 1. Initialize Model
println("1. Initializing GraphMERT Model...")
# Create a default configuration
config = GraphMERTConfig()
# Initialize model with random weights
model = GraphMERTModel(config)
println("   Model initialized with random weights (for demo purposes)")
println("   Architecture: RoBERTa (L=$(config.roberta_config.num_hidden_layers)) + H-GAT + Leafy Chain Graph")

# 2. Setup Biomedical Domain
println("\n2. Setting up Biomedical Domain...")
# Use mock mode for UMLS to avoid API key requirements for demo
umls_client = create_umls_client("mock"; mock_mode=true)
println("   UMLS Client: Mock Mode (Offline)")

# Initialize domain with mock clients
bio_domain = load_biomedical_domain(umls_client)
register_domain!("biomedical", bio_domain)
println("   Biomedical domain registered successfully")

# 3. Extraction Demo
println("\n3. Running Knowledge Graph Extraction...")
text = "Metformin is the first-line treatment for type 2 diabetes. It activates AMPK and inhibits gluconeogenesis."
println("   Input Text: \"$text\"")

options = ProcessingOptions(
    domain = "biomedical",
    confidence_threshold = 0.0, # Lower threshold to ensure we see output even with random weights
    use_umls = true,
    use_helper_llm = false # Disable LLM for basic demo to avoid API keys
)

println("   Extracting...")
try
    # Note: With random weights, the model won't predict meaningful relations,
    # but the pipeline will execute the entity extraction (heuristic/LLM) and graph construction.
    # Since LLM is disabled, entity extraction might fallback to regex/heuristics if implemented in domain.
    # In BiomedicalDomain, without LLM, it uses fallback heuristics.
    
    kg = extract_knowledge_graph(text, model; options=options)
    
    println("   Extraction Complete!")
    println("   Entities Found: $(length(kg.entities))")
    
    if !isempty(kg.entities)
        for (i, e) in enumerate(kg.entities)
            # Fallback to label if entity_type is not available in attributes/field
            etype = get(e.attributes, "entity_type", e.label)
            println("     $i. $(e.text) [$etype] (conf: $(round(e.confidence, digits=2)))")
        end
    else
        println("     (No entities found)")
    end
    
    # Create ID map for robust lookup
    id_to_entity = Dict(e.id => e for e in kg.entities)

    println("   Relations Found: $(length(kg.relations))")
    if !isempty(kg.relations)
        for (i, r) in enumerate(kg.relations)
            head = get(id_to_entity, r.head, nothing)
            tail = get(id_to_entity, r.tail, nothing)
            
            if head !== nothing && tail !== nothing
                println("     $i. $(head.text) --[$(r.relation_type)]--> $(tail.text)")
            else
                println("     $i. Invalid entity IDs: $(r.head) -> $(r.tail)")
            end
        end
    else
        println("     (No relations found)")
    end
    
catch e
    println("   Extraction failed: $e")
    # Base.showerror(stdout, e, catch_backtrace())
end

# 4. Persistence Demo
println("\n4. Testing Model Persistence...")
temp_path = "demo_checkpoint.jld2"
println("   Saving model to $temp_path...")
if save_model(model, temp_path)
    println("   Save successful.")
    
    println("   Loading model back...")
    loaded_model = load_model(temp_path)
    
    if loaded_model !== nothing
        println("   Model loaded successfully!")
        if model.config.hidden_dim == loaded_model.config.hidden_dim
             println("   Config matches: Yes")
        else
             println("   Config matches: No")
        end
    else
        println("   Failed to load model.")
    end
    
    # Cleanup
    if isfile(temp_path)
        rm(temp_path)
        println("   Cleaned up temporary checkpoint.")
    end
else
    println("   Failed to save model.")
end

println("\n=== Demo Complete ===")
println("Run 'julia --project=. GraphMERT/test/runtests.jl' for full test suite.")
