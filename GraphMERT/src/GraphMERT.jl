"""
GraphMERT.jl - Efficient and Scalable Knowledge Graph Construction

A Julia implementation of the GraphMERT algorithm for constructing biomedical
knowledge graphs using RoBERTa-based architecture with Hierarchical Graph Attention (H-GAT).

## Features

- RoBERTa-based encoder with H-GAT components
- Biomedical domain processing with UMLS integration
- Seed KG injection for training data preparation
- MLM + MNM training objectives
- Helper LLM integration for entity discovery
- Leafy chain graph structure for text representation

## Quick Start

```julia
using GraphMERT

# Load a pre-trained model
model = load_model("path/to/model.onnx")

# Extract knowledge graph from text
text = "Diabetes is a chronic condition that affects blood sugar levels."
graph = extract_knowledge_graph(text, model)

# Access entities and relations
println("Entities: ", length(graph.entities))
println("Relations: ", length(graph.relations))
```

## Architecture

The GraphMERT implementation follows the original paper's architecture:

1. **RoBERTa Encoder**: Base transformer for text understanding
2. **H-GAT**: Hierarchical Graph Attention for semantic relation encoding
3. **Leafy Chain Graph**: Structure for representing text with semantic nodes
4. **Domain Abstraction Layer**: Pluggable domain providers for domain-specific functionality
5. **Helper LLM**: External LLM for entity discovery and relation matching

## Performance Targets

- Process 5,000 tokens per second on laptop hardware
- Memory usage < 4GB for datasets up to 124.7M tokens
- FActScore within 5% of original paper results (69.8% target)
- ValidityScore within 5% of original paper results (68.8% target)

## Citation

If you use GraphMERT.jl in your research, please cite the original paper:

```bibtex
@article{graphmert2024,
  title={GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data},
  author={Authors},
  journal={arXiv preprint},
  year={2024}
}
```
"""

__precompile__(false)

module GraphMERT

using Dates
# using DocStringExtensions  # Temporarily disabled

# Core modules
include("types.jl")
include("exceptions.jl")
include("config.jl")
include("utils.jl")
include("testing/progressive.jl")

# Domain abstraction layer
include("domains/interface.jl")
include("domains/registry.jl")

# Domain modules (can be loaded conditionally)
# include("domains/biomedical/domain.jl")  # Loaded on demand or via explicit include

# Architecture components
include("architectures/roberta.jl")
include("architectures/hgat.jl")
include("architectures/attention.jl")

# Graph structures
include("graphs/leafy_chain.jl")
# Note: Domain-specific graph structures (e.g., graphs/biomedical.jl) should be
# included by domain modules, not here in the core module

# Models
include("models/graphmert.jl")
include("models/persistence.jl")

# Text processing (domain-agnostic)
include("text/tokenizer.jl")
# Note: Domain-specific text processing (e.g., text/pubmed.jl) should be
# included by domain modules, not here in the core module

# LLM integration
include("llm/helper.jl")

# Training
include("training/mlm.jl")
include("training/mnm.jl")
include("training/seed_injection.jl")
include("training/pipeline.jl")
include("training/span_masking.jl")
include("data/preparation.jl")

# Evaluation
# Note: Evaluation modules contain domain-specific code (e.g., UMLSClient for biomedical)
# These will be refactored when creating domain modules
include("evaluation/factscore.jl")
# include("evaluation/validity.jl")  # Temporarily disabled - requires UMLSClient which is domain-specific
include("evaluation/graphrag.jl")
# include("evaluation/diabetes.jl")  # Temporarily disabled - domain-specific

# Benchmarking
include("benchmarking/benchmarks.jl")
include("monitoring/performance.jl")

# API
include("api/extraction.jl")
include("api/batch.jl")
include("api/config.jl")
include("api/helpers.jl")
include("api/serialization.jl")

# Optimization
include("optimization/memory.jl")
include("optimization/speed.jl")

# Export main API functions
export extract_knowledge_graph, load_model, preprocess_text
export KnowledgeGraph, KnowledgeEntity, KnowledgeRelation, TextPosition
export GraphMERTModel, ProcessingOptions, GraphMERTConfig
export FActScore, ValidityScore, GraphRAG
export filter_knowledge_graph
export export_knowledge_graph, export_to_json, export_to_csv, export_to_rdf, export_to_ttl

# Export domain-related functions
export DomainProvider, DomainConfig
export register_domain!, get_domain, list_domains
export set_default_domain, get_default_domain, has_domain
export extract_entities, extract_relations
export validate_entity, validate_relation
export calculate_entity_confidence, calculate_relation_confidence

# Export helper LLM functions
export create_helper_llm_client, discover_entities, match_relations
export discover_entities_batch, match_relations_batch

# Note: Domain-specific exports (UMLS, PubMed, biomedical graph functions) should be
# exported by domain modules, not here in the core module

# Export MLM functions
export create_mlm_config, create_mlm_batch, train_mlm_step, evaluate_mlm
export calculate_mlm_loss, calculate_boundary_loss, calculate_total_mlm_loss

# Export utility functions
export preprocess_text, tokenize_text, normalize_text, softmax, sigmoid, relu
export cosine_similarity, shuffle_data, split_data, batch_data

# ============================================================================
# Main API Functions
# ============================================================================

"""
    extract_knowledge_graph(text::String; options::ProcessingOptions=default_processing_options())

Extract a knowledge graph from text using GraphMERT.

This function delegates to the domain-specific extraction logic via the domain provider.

# Arguments
- `text::String`: Input text to process
- `options::ProcessingOptions`: Processing options (must include domain field)

# Returns
- `KnowledgeGraph`: Extracted knowledge graph with entities and relations

# Example
```julia
using GraphMERT

# Set domain in options
options = ProcessingOptions(domain="biomedical")
text = "Alzheimer's disease is a neurodegenerative disorder."
graph = extract_knowledge_graph(text; options=options)
println("Found \$(length(graph.entities)) entities and \$(length(graph.relations)) relations")
```
"""
function extract_knowledge_graph(
  text::String;
  options::ProcessingOptions=default_processing_options(),
)
  # Delegate to the domain-specific extraction API
  # This function is kept here for backward compatibility but delegates to api/extraction.jl
  # Create a dummy model for now (in real usage, this would be provided)
  model = create_graphmert_model(GraphMERTConfig())
  
  # Use the domain-aware extraction function from api/extraction.jl
  return extract_knowledge_graph(text, model; options=options)
end

"""
    load_graphmert_model(model_path::String)

Load a pre-trained GraphMERT model.

# Arguments
- `model_path::String`: Path to the model file

# Returns
- `GraphMERTModel`: Loaded model

# Example
```julia
model = load_model("path/to/model.onnx")
```
"""
function load_model(model_path::String)
  # Basic implementation: create a GraphMERT model from scratch
  # In a full implementation, this would load pre-trained weights from model_path

  # Use default configuration for now
  config = GraphMERTConfig()

  # Use the existing create_graphmert_model function
  model = create_graphmert_model(config)

  return model
end

"""
    preprocess_text_for_graphmert(text::String; max_length::Int=512)

Preprocess text for GraphMERT processing.

# Arguments
- `text::String`: Input text
- `max_length::Int`: Maximum text length (default: 512)

# Returns
- `String`: Preprocessed text

# Example
```julia
processed = preprocess_text_for_graphmert("Alzheimer's disease is a neurodegenerative disorder.")
```
"""
function preprocess_text_for_graphmert(text::String; max_length::Int=512)
  # Basic text preprocessing
  text = strip(text)
  if length(text) > max_length
    text = text[1:max_length]
  end
  return String(text)  # Ensure we return a String, not SubString
end

# ============================================================================
# Domain Initialization (Optional)
# ============================================================================

"""
    initialize_default_domains()

Initialize and register default domains (biomedical and Wikipedia) for convenience.

This function loads both domain modules and registers them if they're available.
Call this after loading domain modules if you want them available by default.

# Example
```julia
using GraphMERT
include("GraphMERT/src/domains/biomedical.jl")
include("GraphMERT/src/domains/wikipedia.jl")
initialize_default_domains()
```
"""
function initialize_default_domains()
    # Try to load biomedical domain
    try
        if !has_domain("biomedical")
            include("domains/biomedical.jl")
            biomedical_domain = load_biomedical_domain()
            register_domain!("biomedical", biomedical_domain)
            @info "Initialized biomedical domain"
        end
    catch e
        @warn "Could not initialize biomedical domain: $e"
    end
    
    # Try to load Wikipedia domain
    try
        if !has_domain("wikipedia")
            include("domains/wikipedia.jl")
            wikipedia_domain = load_wikipedia_domain()
            register_domain!("wikipedia", wikipedia_domain)
            @info "Initialized Wikipedia domain"
        end
    catch e
        @warn "Could not initialize Wikipedia domain: $e"
    end
end

# Export
export initialize_default_domains

"""
    fallback_entity_recognition(text::String, domain::DomainProvider=nothing)

Fallback entity recognition using simple pattern matching when ML models are not available.
If domain is provided, uses domain provider's extract_entities method.

# Deprecated: Use domain provider's extract_entities method instead
"""
function fallback_entity_recognition(text::String, domain::Union{Any, Nothing}=nothing)
    if domain !== nothing
        # Use domain provider's extract_entities method
        options = ProcessingOptions(domain=get_domain_name(domain))
        entities = extract_entities(domain, text, options)
        return [e.text for e in entities]
    end
    
    # Basic fallback: simple noun phrase extraction
    # Split by whitespace and capitalize potential entities
    words = split(text)
    entities = String[]
    
    # Simple heuristic: capitalize potential proper nouns and noun phrases
    for (i, word) in enumerate(words)
        # Skip very short words
        if length(word) > 3 && isuppercase(word[1])
            push!(entities, word)
        end
    end
    
    return entities
end

"""
    fallback_relation_matching(entities::Vector{String}, text::String, domain::DomainProvider=nothing)

Fallback relation matching using simple co-occurrence and pattern matching.
If domain is provided, uses domain provider's extract_relations method.

# Deprecated: Use domain provider's extract_relations method instead
"""
function fallback_relation_matching(entities::Vector{String}, text::String, domain::Union{Any, Nothing}=nothing)
    if domain !== nothing
        # Use domain provider's extract_relations method
        # Convert entities to Entity objects
        entity_objects = Vector{Entity}()
        for (i, entity_text) in enumerate(entities)
            push!(entity_objects, Entity(
                "entity_$i",
                entity_text,
                entity_text,
                "UNKNOWN",
                get_domain_name(domain),
                Dict{String,Any}(),
                TextPosition(1, length(entity_text), 1, 1),
                0.5,
                text
            ))
        end
        
        options = ProcessingOptions(domain=get_domain_name(domain))
        relations = extract_relations(domain, entity_objects, text, options)
        
        # Convert to Dict format for backward compatibility
        result = Dict{String, Dict{String, Any}}()
        for rel in relations
            key = "$(rel.head)_$(rel.relation_type)_$(rel.tail)"
            result[key] = Dict(
                "entity1" => rel.head,
                "entity2" => rel.tail,
                "relation" => rel.relation_type,
                "confidence" => rel.confidence,
            )
        end
        return result
    end
    
    # Basic fallback: simple co-occurrence based relations
    relations = Dict{String, Dict{String, Any}}()

    # Simple co-occurrence based relations
    for i in 1:length(entities)
        for j in (i+1):length(entities)
            entity1 = entities[i]
            entity2 = entities[j]

            # Check if entities appear close to each other
            pos1 = findfirst(entity1, text)
            pos2 = findfirst(entity2, text)

            if pos1 !== nothing && pos2 !== nothing
                distance = abs(first(pos1) - first(pos2))

                # If entities are close (within 200 characters), create a relation
                if distance < 200
                    # Determine relation type based on simple heuristics
                    relation_type = "ASSOCIATED_WITH"

                    # Check for generic patterns (not domain-specific)
                    text_lower = lowercase(text)
                    if occursin("treat", text_lower) || occursin("treats", text_lower)
                        relation_type = "TREATS"
                    elseif occursin("cause", text_lower) || occursin("causes", text_lower)
                        relation_type = "CAUSES"
                    elseif occursin("prevent", text_lower) || occursin("prevents", text_lower)
                        relation_type = "PREVENTS"
                    elseif occursin("related", text_lower) || occursin("related to", text_lower)
                        relation_type = "RELATED_TO"
                    end

                    key = "$(entity1)_$(relation_type)_$(entity2)"
                    relations[key] = Dict(
                        "entity1" => entity1,
                        "entity2" => entity2,
                        "relation" => relation_type,
                        "confidence" => 0.5,
                        "distance" => distance,
                    )
                end
            end
        end
    end

    return relations
end

# ============================================================================
# Export Additional Functions from Submodules
# ============================================================================

# Export from API module
export extract_knowledge_graph,
  discover_head_entities,
  match_relations_for_entities,
  predict_tail_tokens,
  form_tail_from_tokens,
  filter_and_deduplicate_triples

# Export from Evaluation modules
export evaluate_factscore,
  evaluate_validity,
  evaluate_graphrag,
  calculate_factscore_confidence_interval,
  calculate_validity_confidence_interval

# Export from Config module
export default_processing_options

# Export from Training.MNM module
export select_leaves_to_mask,
  apply_mnm_masks, calculate_mnm_loss, create_mnm_batch, train_mnm_step, evaluate_mnm

# Export from Training.SeedInjection module
export link_entity_sapbert,
  select_triples_for_entity,
  inject_seed_kg,
  select_triples_for_injection,
  bucket_by_score,
  bucket_by_relation_frequency,
  validate_injected_triples

# Export from Training.Pipeline module
export train_graphmert,
  prepare_training_data, create_training_configurations, load_training_data

# Export from Graphs module
export default_chain_graph_config,
  create_empty_chain_graph,
  build_adjacency_matrix,
  floyd_warshall,
  inject_triple!,
  graph_to_sequence,
  create_attention_mask,
  create_leafy_chain_from_text

# Export from Types module (additional types)
export ChainGraphNode,
  ChainGraphConfig,
  LeafyChainGraph,
  MNMConfig,
  MNMBatch,
  SeedInjectionConfig,
  EntityLinkingResult,
  SemanticTriple

end # module GraphMERT
