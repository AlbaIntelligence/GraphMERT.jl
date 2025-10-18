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

# Core modules
include("types.jl")
include("exceptions.jl")
include("config.jl")
include("utils.jl")

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
export KnowledgeGraph, BiomedicalEntity, BiomedicalRelation
export GraphMERTModel, ProcessingOptions, GraphMERTConfig
export FActScore, ValidityScore, GraphRAG

end # module GraphMERT
