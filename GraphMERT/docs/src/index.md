# GraphMERT.jl

[![Build Status](https://github.com/alba-intelligence/GraphMERT.jl/workflows/CI/badge.svg)](https://github.com/alba-intelligence/GraphMERT.jl/actions)
[![Coverage](https://codecov.io/gh/alba-intelligence/GraphMERT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alba-intelligence/GraphMERT.jl)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://alba-intelligence.github.io/GraphMERT.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://alba-intelligence.github.io/GraphMERT.jl/dev)

**Efficient and Scalable Knowledge Graph Construction from Unstructured Text**

> **Implementation status (March 2026)**: Architecture and MLM training are structurally complete.
> Real training, tail prediction, and model persistence are not yet implemented.
> See [`reports/PARITY_PLAN.md`](../../reports/PARITY_PLAN.md) for details.

GraphMERT.jl is a Julia implementation of the GraphMERT algorithm for constructing knowledge
graphs using RoBERTa-based architecture with Hierarchical Graph Attention (H-GAT).

## Features

- 🔗 **Reliability pipeline**: Provenance (`get_provenance`), ontology validation (`validate_kg` → ValidityReport), factuality (`evaluate_factscore(kg, reference)` → FactualityScore), and KG cleaning (`clean_kg(kg; policy)`).
- 🧬 **Domain-Agnostic Architecture**: Pluggable domain system (biomedical + Wikipedia included)
- 🚀 **Performance target**: 5,000+ tokens/second (pending real training)
- 🧠 **Advanced Architecture**: RoBERTa encoder with Hierarchical Graph Attention
- 📊 **Evaluation Metrics**: FActScore*, ValidityScore, and GraphRAG
- 🎯 **Seed KG Injection**: Prior knowledge integration algorithm (UMLS-backed)
- 🤖 **Helper LLM Support**: External LLM for entity discovery, relation matching, and tail formation

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

## Performance Targets

- **Throughput**: 5,000+ tokens per second
- **Memory**: <4GB for datasets up to 124.7M tokens
- **Accuracy**: FActScore within 5% of original paper (69.8% target)
- **Validity**: ValidityScore within 5% of original paper (68.8% target)

## Architecture

The GraphMERT implementation follows the original paper's architecture:

1. **RoBERTa Encoder**: Base transformer for text understanding
2. **H-GAT**: Hierarchical Graph Attention for semantic relation encoding
3. **Leafy Chain Graph**: Structure for representing text with semantic nodes
4. **UMLS Integration**: Biomedical knowledge base for entity linking
5. **Helper LLM**: External LLM for entity discovery and relation matching

## Scientific Validation

> ⚠️ **Not yet validated**: The paper targets below have not been reproduced from a real trained
> model. The training pipeline currently uses mock losses. Validation is pending completion of
> [`reports/TASK_LIST.md`](../../reports/TASK_LIST.md) streams A–D.

Paper targets (arXiv:2510.09580):

- FActScore target: **69.8%** (diabetes domain)
- ValidityScore target: **68.8%**
- Training: 90 GPU-hours on PubMed diabetes abstracts

## Installation

```julia
using Pkg
Pkg.add("GraphMERT")
```

## Documentation

- 📖 [Getting Started](getting_started/installation.md)
- 🧬 [User Guide](user_guide/core_concepts.md)
- 📚 [API Reference](api/core.md)
- 🛠️ [Troubleshooting](troubleshooting.md)

## Citation

If you use GraphMERT.jl in your research, please cite:

```bibtex
@article{graphmert2024,
  title={GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/alba-intelligence/GraphMERT.jl/blob/main/LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/alba-intelligence/GraphMERT.jl) for details.

## Support

- 📧 Email: alba.intelligence@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/alba-intelligence/GraphMERT.jl/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/alba-intelligence/GraphMERT.jl/discussions)
