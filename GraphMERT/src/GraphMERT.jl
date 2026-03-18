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

# Core modules first (types needed by other modules)
include("types.jl")
include("config.jl")
include("utils.jl")

# LLM integration
include("llm/local.jl")
include("llm/helper.jl")
include("llm/embeddings.jl")
# Local LLM (GGUF/llama-cpp) and helper LLM only; no Ollama

# Re-export from LocalLLM module
using .LocalLLM: LocalLLMConfig, LocalLLMClient, LocalModelMetadata, load_local_model

# Domain abstraction layer
include("domains/interface.jl")
include("domains/registry.jl")

# Domain modules
include("domains/biomedical.jl")
include("domains/wikipedia.jl")

# Architecture components
include("architectures/attention.jl")
include("architectures/roberta.jl")
include("architectures/hgat.jl")

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

# Training
include("training/logging.jl")
include("training/distillation.jl")
include("training/mlm.jl")
include("training/mnm.jl")
include("training/ontology_sources.jl")
include("training/ontology.jl")
include("training/seed_injection.jl")
include("training/pipeline.jl")
include("training/span_masking.jl")
include("data/preparation.jl")

# Evaluation
# Note: Evaluation modules contain domain-specific code (e.g., UMLSClient for biomedical)
# These will be refactored when creating domain modules
include("evaluation/factscore.jl")
include("evaluation/validity.jl")
include("evaluation/graphrag.jl")
# include("evaluation/diabetes.jl")  # Temporarily disabled - domain-specific

# Benchmarking
include("benchmarking/benchmarks.jl")
include("monitoring/performance.jl")

# API
include("api/extraction.jl")
include("training/validation.jl")
include("api/reliability.jl")
include("api/batch.jl")
include("api/config.jl")
include("api/helpers.jl")
include("api/serialization.jl")

# LLM integrations
include("llm/helper.jl")
include("llm/local.jl")

# Visualization
include("visualization/visualization.jl")

# Optimization
include("optimization/memory.jl")
include("optimization/speed.jl")

# Export main API functions
export extract_knowledge_graph, load_model, save_model, preprocess_text
export KnowledgeGraph, KnowledgeEntity, KnowledgeRelation, TextPosition, ProvenanceRecord
export ValidityReport, FactualityScore, CleaningPolicy
export get_provenance, validate_kg, clean_kg
export GraphMERTModel, ProcessingOptions, GraphMERTConfig
export FActScore, ValidityScore, GraphRAG
export filter_knowledge_graph
export export_knowledge_graph, export_to_json, export_to_csv, export_to_rdf, export_to_ttl

# Export EPIC 2 functions
export link_entities_sapbert, get_umls_triples, inject_seed_triples

# Export EPIC 5 functions (Attention Mechanisms)
export SpatialAttentionConfig, create_attention_decay_mask, create_graph_attention_mask

# Export domain-related functions
export DomainProvider, DomainConfig
export register_domain!, get_domain, list_domains
export set_default_domain, get_default_domain, get_default_domain_name, has_domain
export extract_entities, extract_relations
export validate_entity, validate_relation
export calculate_entity_confidence, calculate_relation_confidence
export load_wikipedia_domain, load_biomedical_domain

# Export helper LLM functions
export AbstractLLMClient, OpenAIClient, MockLLMClient, GeminiClient
export create_helper_llm_client, create_mock_llm_client, create_gemini_client
export discover_entities, match_relations, make_llm_request
export discover_entities_batch, match_relations_batch, form_tail_from_tokens
export create_entity_discovery_prompt, create_relation_matching_prompt, create_tail_formation_prompt
export parse_entity_response, parse_relation_response, parse_tail_formation_response

# Export local LLM functions
export LocalLLMConfig, LocalLLMClient, LocalModelMetadata, load_local_model

# Export embedding functions
export AbstractEmbeddingClient, GeminiEmbeddingClient, MockEmbeddingClient
export create_gemini_embedding_client, create_mock_embedding_client
export embed, embed_batch, cosine_similarity

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
  # Use default encoder (roberta-base) when available, else fall back to in-memory default
  model = load_model()
  if model === nothing
    model = create_graphmert_model(GraphMERTConfig())
  end
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
  # Delegate to the persistence-layer loader, which handles missing files
  # and returns `nothing` when loading fails.
  return load_model(model_path; device=:cpu, strict=true)
end

"""
    load_model()::Union{GraphMERTModel, Nothing}

Load the default encoder (roberta-base) from ~/.cache/llama-cpp/models/encoders/roberta-base.
If that path is missing or invalid, returns `nothing`. Override the root with
`GRAPHMERT_ENCODER_ROOT` (directory containing encoder subdirs).
"""
function load_model()::Union{GraphMERTModel,Nothing}
  return load_model(default_encoder_path())
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
load_wikipedia_domain(wikidata_client::Union{Any, Nothing} = nothing)

Load and register the Wikipedia domain.

# Arguments
- `wikidata_client::Union{Any, Nothing}`: Optional Wikidata client for entity linking

# Returns
- `WikipediaDomain`: The created domain instance

# Example
```julia
using GraphMERT
domain = load_wikipedia_domain()
register_domain!("wikipedia", domain)
```
"""
function load_wikipedia_domain(wikidata_client::Union{Any, Nothing} = nothing)
include(joinpath(@__DIR__, "domains", "wikipedia.jl"))
return Base.invokelatest(GraphMERT.WikipediaDomain, wikidata_client)
end

"""
load_biomedical_domain(umls_client::Union{Any, Nothing} = nothing)

Load and register the biomedical domain.

# Arguments
- `umls_client::Union{Any, Nothing}`: Optional UMLS client for entity linking

# Returns
- `BiomedicalDomain`: The created domain instance

# Example
```julia
using GraphMERT
domain = load_biomedical_domain()
register_domain!("biomedical", domain)
```
"""
function load_biomedical_domain(umls_client::Union{Any, Nothing} = nothing)
    include(joinpath(@__DIR__, "domains", "biomedical.jl"))
    return Base.invokelatest(GraphMERT.BiomedicalDomain, umls_client)
end

"""
    initialize_default_domains()

Initialize and register default domains (biomedical and Wikipedia) for convenience.

This function loads both domain modules and registers them if they're available.
Call this after loading domain modules if you want them available by default.

# Example
```julia
using GraphMERT
initialize_default_domains()
```
"""
function initialize_default_domains()
    # Try to load biomedical domain (not implemented yet)
    try
        if !has_domain("biomedical")
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

    # Basic fallback: simple noun phrase / keyword extraction
    # Split by whitespace and capture medically relevant tokens even if lowercase
    words = split(text)
    entities = String[]

    # Heuristic: keep reasonably long tokens that look like content words
    for (i, word) in enumerate(words)
        clean = strip(replace(word, r"[^\p{L}]" => ""))
        # Skip very short or empty tokens
        if length(clean) ≤ 3
            continue
        end

        lower = lowercase(clean)
        # Always keep if it starts with uppercase (proper nouns)
        if isuppercase(clean[1])
            push!(entities, clean)
        # Also keep common biomedical keywords regardless of casing
        elseif occursin(r"(diabet|metformin|insulin|glucos|cancer|tumor|protein|gene)", lower)
            push!(entities, clean)
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
export default_processing_options, default_encoder_path

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

# Export from Visualization module
export kg_to_graphs_format, visualize_graph, plot_knowledge_graph
export filter_by_confidence, filter_by_entity_type, filter_by_relation_type
export simplify_graph, cluster_entities, create_visualization_summary
export validate_visualization_input

# ============================================================================
# Visualization Functions (delegating to Visualization submodule)
# ============================================================================

"""
    kg_to_graphs_format(kg::KnowledgeGraph; validate::Bool=true)

Convert a KnowledgeGraph to Graphs.jl/MetaGraphs.jl format.

Delegates to Visualization.kg_to_graphs_format.
"""
kg_to_graphs_format(kg::KnowledgeGraph; validate::Bool=true) =
    Visualization.kg_to_graphs_format(kg, validate=validate)

"""
    visualize_graph(kg::KnowledgeGraph; kwargs...)

Create a static visualization of a knowledge graph.

Delegates to Visualization.visualize_graph.
"""
visualize_graph(kg::KnowledgeGraph; kwargs...) =
    Visualization.visualize_graph(kg; kwargs...)

"""
    plot_knowledge_graph(kg::KnowledgeGraph; kwargs...)

Alias for visualize_graph.

Delegates to Visualization.plot_knowledge_graph.
"""
plot_knowledge_graph(kg::KnowledgeGraph; kwargs...) =
    Visualization.plot_knowledge_graph(kg; kwargs...)

"""
    filter_by_confidence(kg::KnowledgeGraph, min_confidence::Float64)

Filter knowledge graph by confidence.

Delegates to Visualization.filter_by_confidence.
"""
filter_by_confidence(kg::KnowledgeGraph, min_confidence::Float64) =
    Visualization.filter_by_confidence(kg, min_confidence)

"""
    filter_by_entity_type(kg::KnowledgeGraph, entity_types::Vector{String})

Filter knowledge graph by entity types.

Delegates to Visualization.filter_by_entity_type.
"""
filter_by_entity_type(kg::KnowledgeGraph, entity_types::Vector{String}) =
    Visualization.filter_by_entity_type(kg, entity_types)

"""
    filter_by_relation_type(kg::KnowledgeGraph, relation_types::Vector{String})

Filter knowledge graph by relation types.

Delegates to Visualization.filter_by_relation_type.
"""
filter_by_relation_type(kg::KnowledgeGraph, relation_types::Vector{String}) =
    Visualization.filter_by_relation_type(kg, relation_types)

"""
    simplify_graph(kg::KnowledgeGraph; kwargs...)

Simplify a knowledge graph for visualization.

Delegates to Visualization.simplify_graph.
"""
simplify_graph(kg::KnowledgeGraph; kwargs...) =
    Visualization.simplify_graph(kg; kwargs...)

"""
    cluster_entities(kg::KnowledgeGraph, method::Symbol=:entity_type)

Cluster entities in a knowledge graph.

Delegates to Visualization.cluster_entities.
"""
cluster_entities(kg::KnowledgeGraph, method::Symbol=:entity_type) =
    Visualization.cluster_entities(kg, method)

"""
    create_visualization_summary(kg::KnowledgeGraph)

Create a summary of knowledge graph statistics for visualization.

Delegates to Visualization.create_visualization_summary.
"""
create_visualization_summary(kg::KnowledgeGraph) =
    Visualization.create_visualization_summary(kg)

"""
    validate_visualization_input(kg::KnowledgeGraph)

Validate that a knowledge graph is suitable for visualization.

Delegates to Visualization.validate_visualization_input.
"""
validate_visualization_input(kg::KnowledgeGraph) =
    Visualization.validate_visualization_input(kg)

end # module GraphMERT
