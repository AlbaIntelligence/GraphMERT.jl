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
4. **UMLS Integration**: Biomedical knowledge base for entity linking
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

module GraphMERT

using Dates
using DocStringExtensions

# Core modules
include("types.jl")
include("exceptions.jl")
include("config.jl")
include("utils.jl")
include("testing/progressive.jl")

# Architecture components
include("architectures/roberta.jl")
include("architectures/hgat.jl")
include("architectures/attention.jl")

# Graph structures
include("graphs/leafy_chain.jl")
include("graphs/biomedical.jl")

# Models
include("models/graphmert.jl")
include("models/persistence.jl")

# Biomedical domain
include("biomedical/umls.jl")
include("biomedical/entities.jl")
include("biomedical/relations.jl")
include("text/pubmed.jl")

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
include("evaluation/factscore.jl")
include("evaluation/validity.jl")
include("evaluation/graphrag.jl")
include("evaluation/diabetes.jl")

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

# Export biomedical types and functions
export BiomedicalEntityType, BiomedicalRelationType
export DISEASE,
  DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM, PROCEDURE, ORGANISM, CHEMICAL, CELL_TYPE
export TREATS,
  CAUSES, ASSOCIATED_WITH, PREVENTS, INHIBITS, ACTIVATES, BINDS_TO, INTERACTS_WITH
export REGULATES, EXPRESSES, LOCATED_IN, PART_OF, DERIVED_FROM, SYNONYMOUS_WITH
export CONTRAINDICATED_WITH, INDICATES, MANIFESTS_AS, ADMINISTERED_FOR, TARGETS
export METABOLIZED_BY, TRANSPORTED_BY, SECRETED_BY, PRODUCED_BY, CONTAINS, COMPONENT_OF
export UNKNOWN, UNKNOWN_RELATION

# Export biomedical functions
export extract_entities_from_text, classify_entity, validate_biomedical_entity
export classify_relation, validate_biomedical_relation, calculate_entity_confidence
export calculate_relation_confidence, normalize_entity_text, get_entity_type_name
export get_relation_type_name, get_supported_entity_types, get_supported_relation_types

# Export UMLS functions
export create_umls_client, search_concepts, get_concept_details, link_entity
export fallback_entity_recognition, fallback_relation_matching
export get_entity_cui, get_entity_semantic_types, link_entities_batch

# Export helper LLM functions
export create_helper_llm_client, discover_entities, match_relations
export discover_entities_batch, match_relations_batch

# Export PubMed functions
export create_pubmed_client, search_pubmed, fetch_pubmed_articles, process_pubmed_article

# Export knowledge graph functions
export build_biomedical_graph, analyze_biomedical_graph, calculate_graph_metrics
export find_connected_components, filter_by_confidence, filter_by_entity_type
export export_to_json, get_entity_by_id, get_relations_by_entity

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
    extract_knowledge_graph(text::String; options::ProcessingOptions=ProcessingOptions())

Extract a knowledge graph from text using GraphMERT.

# Arguments
- `text::String`: Input text to process
- `options::ProcessingOptions`: Processing options (optional)

# Returns
- `KnowledgeGraph`: Extracted knowledge graph with entities and relations

# Example
```julia
text = "Alzheimer's disease is a neurodegenerative disorder."
graph = extract_knowledge_graph(text)
println("Found \$(length(graph.entities)) entities and \$(length(graph.relations)) relations")
```
"""
function extract_knowledge_graph(
  text::String;
  options::ProcessingOptions=ProcessingOptions(),
)
  # Preprocess text
  processed_text = preprocess_text_for_graphmert(text; max_length=options.max_length)

  # Extract entities using fallback method
  entities = fallback_entity_recognition(processed_text)

  # Convert to KnowledgeEntity objects
  knowledge_entities = Vector{KnowledgeEntity}()
  for (i, entity_text) in enumerate(entities)
    entity = KnowledgeEntity(
      "entity_$i",
      entity_text,
      "UNKNOWN",
      0.5,
      TextPosition(1, length(entity_text), 1, 1),
      Dict{String,Any}(),
      Dates.now(),
    )
    push!(knowledge_entities, entity)
  end

  # Extract relations using fallback method
  relations = fallback_relation_matching(entities, processed_text)

  # Convert to KnowledgeRelation objects
  knowledge_relations = Vector{KnowledgeRelation}()
  for (key, rel_data) in relations
    relation = KnowledgeRelation(
      head=rel_data["entity1"],
      tail=rel_data["entity2"],
      relation_type=rel_data["relation"],
      confidence=0.5,
      attributes=Dict{String,Any}("context" => processed_text),
      created_at=Dates.now(),
    )
    push!(knowledge_relations, relation)
  end

  # Create knowledge graph
  metadata = Dict{String,Any}(
    "total_entities" => length(knowledge_entities),
    "total_relations" => length(knowledge_relations),
    "confidence_threshold" => options.confidence_threshold,
    "processing_time" => Dates.now(),
  )

  return KnowledgeGraph(knowledge_entities, knowledge_relations, metadata)
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
model = load_graphmert_model("path/to/model.onnx")
```
"""
function load_graphmert_model(model_path::String)
  # Placeholder implementation
  # TODO: Implement actual model loading
  config = GraphMERTConfig(model_path=model_path)
  return GraphMERTModel(config)
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
