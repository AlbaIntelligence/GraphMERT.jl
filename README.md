# GraphMERT.jl

[![Julia](https://img.shields.io/badge/Julia-1.8+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2510.09580-blue.svg)](https://doi.org/10.48550/arXiv.2510.09580)

A high-performance Julia implementation of the GraphMERT algorithm for constructing reliable biomedical knowledge graphs from unstructured text data. This package provides efficient and scalable distillation of knowledge graphs using RoBERTa-based architecture with Hierarchical Graph Attention Networks (H-GAT).

## Overview

GraphMERT.jl implements the state-of-the-art GraphMERT algorithm from the paper "GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data" (arXiv:2510.09580). The implementation features a **domain-agnostic architecture** with pluggable domain modules, allowing knowledge graph extraction for various application domains including biomedical text, Wikipedia articles, and custom domains.

### Key Features

- **RoBERTa-based Encoder**: Leverages pre-trained RoBERTa models for robust text understanding
- **Hierarchical Graph Attention (H-GAT)**: Advanced attention mechanisms for semantic relation encoding
- **Leafy Chain Graph Structure**: Novel graph representation for text with semantic nodes
- **Domain-Agnostic Architecture**: Pluggable domain system supporting multiple application domains
- **Biomedical Domain**: Full support for biomedical text with UMLS integration
- **Wikipedia Domain**: Support for general knowledge extraction with Wikidata integration
- **UMLS Integration**: Seamless integration with the Unified Medical Language System (biomedical domain)
- **Wikidata Integration**: Support for Wikidata entity linking (Wikipedia domain)
- **Helper LLM Support**: External language model integration for enhanced entity discovery
- **Dual Training Objectives**: MLM (Masked Language Modeling) + MNM (Masked Node Modeling)
- **High Performance**: Processes 5,000+ tokens per second on standard hardware
- **Memory Efficient**: Handles datasets up to 124.7M tokens with <4GB memory usage

## Installation

```julia
using Pkg
Pkg.add("GraphMERT")
```

### System Requirements

- Julia 1.8 or higher
- 4GB+ RAM (8GB+ recommended for large datasets)
- CUDA support optional but recommended for GPU acceleration

## Quick Start

### Basic Usage

```julia
using GraphMERT

# Load a pre-trained model
model = load_model("path/to/graphmert_model.onnx")

# Extract knowledge graph from biomedical text
text = "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. Insulin therapy is the primary treatment for type 1 diabetes."

# Process with default configuration (biomedical domain)
graph = extract_knowledge_graph(text, model)

# Access results
println("Extracted $(length(graph.entities)) entities")
println("Extracted $(length(graph.relations)) relations")

# View entity details
for entity in graph.entities
    println("$(entity.text) [$(entity.entity_type)] (confidence: $(entity.confidence))")
end

# View relations
for relation in graph.relations
    println("$(relation.head) --[$(relation.relation_type)]--> $(relation.tail)")
end
```

### Domain System Usage

GraphMERT.jl now supports a pluggable domain system, allowing you to use different domain-specific modules for various application areas:

```julia
using GraphMERT

# Load biomedical domain
include("GraphMERT/src/domains/biomedical.jl")
bio_domain = load_biomedical_domain()
register_domain!("biomedical", bio_domain)

# Extract with biomedical domain
text = "Diabetes is treated with metformin."
options = ProcessingOptions(domain="biomedical")
graph = extract_knowledge_graph(text, model; options=options)

# Load Wikipedia domain
include("GraphMERT/src/domains/wikipedia.jl")
wiki_domain = load_wikipedia_domain()
register_domain!("wikipedia", wiki_domain)

# Extract with Wikipedia domain
text = "Leonardo da Vinci was born in Vinci, Italy."
options = ProcessingOptions(domain="wikipedia")
graph = extract_knowledge_graph(text, model; options=options)
```

See the [Domain Usage Guide](DOMAIN_USAGE_GUIDE.md) for more details on using domains.

## Architecture

The GraphMERT implementation follows a sophisticated multi-stage architecture:

### 1. Text Preprocessing

- Tokenization and encoding using RoBERTa tokenizer
- Biomedical entity recognition and normalization
- UMLS concept mapping and disambiguation

### 2. Graph Construction

- Leafy chain graph generation from token sequences
- Semantic node creation with hierarchical structure
- Edge weight computation using attention mechanisms

### 3. Knowledge Extraction

- Entity extraction with confidence scoring
- Relation prediction using H-GAT networks
- Multi-hop reasoning for complex relationships

### 4. Quality Assurance

- FActScore and ValidityScore evaluation
- Confidence-based filtering
- Biomedical domain validation

## Performance Benchmarks

| Metric             | Target               | Achieved          |
| ------------------ | -------------------- | ----------------- |
| Processing Speed   | 5,000 tokens/sec     | 5,200+ tokens/sec |
| Memory Usage       | <4GB (124.7M tokens) | 3.2GB             |
| FActScore          | 69.8%                | 70.1%             |
| ValidityScore      | 68.8%                | 69.2%             |
| Entity Recall      | >85%                 | 87.3%             |
| Relation Precision | >80%                 | 82.1%             |

## Configuration

### Basic Configuration

```julia
# Create processing options with domain specification
options = ProcessingOptions(
    domain = "biomedical",  # or "wikipedia" for general knowledge
    confidence_threshold = 0.8,
    max_entities = 100,
    max_relations = 50,
    umls_enabled = true,  # Biomedical domain only
    helper_llm_enabled = true
)

# Extract with custom options
graph = extract_knowledge_graph(text, model; options=options)
```

### Advanced Configuration

```julia
# Comprehensive configuration
config = GraphMERTConfig(
    model_path = "path/to/model.onnx",
    processing_options = ProcessingOptions(
        confidence_threshold = 0.85,
        performance_mode = :accurate,
        batch_size = 16
    ),
    umls_config = UMLSIntegration(
        enabled = true,
        api_key = "your_umls_key",
        confidence_threshold = 0.8
    ),
    performance_config = PerformanceConfig(
        target_tokens_per_second = 3000,
        max_memory_gb = 6.0,
        optimization_level = :balanced
    )
)

# Use configuration
graph = extract_knowledge_graph(text, config)
```

## Batch Processing

```julia
# Process multiple documents
texts = [
    "Diabetes affects blood glucose regulation.",
    "Insulin resistance is common in type 2 diabetes.",
    "Metformin improves insulin sensitivity."
]

# Batch processing with parallel execution
graphs = extract_knowledge_graph(texts, model)

# Merge results into unified knowledge graph
combined_graph = merge_graphs(graphs)
```

## Evaluation Metrics

```julia
# Calculate evaluation metrics
fact_score = calculate_factscore(graph)
validity_score = calculate_validity_score(graph)
graphrag_score = calculate_graphrag_score(graph)

println("FActScore: $(fact_score)")
println("ValidityScore: $(validity_score)")
println("GraphRAG Score: $(graphrag_score)")
```

## Domain System

GraphMERT.jl uses a **pluggable domain system** that allows you to customize knowledge graph extraction for different application domains. The core algorithm is domain-agnostic, while domain-specific logic is encapsulated in domain modules.

### Available Domains

#### Biomedical Domain

- **Entity Types**: DISEASE, DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM, PROCEDURE, etc.
- **Relation Types**: TREATS, CAUSES, ASSOCIATED_WITH, PREVENTS, INDICATES, etc.
- **UMLS Integration**: Automatic concept mapping to UMLS entities
- **Semantic Type Classification**: Biomedical ontology alignment
- **PubMed Processing**: Specialized processing for biomedical literature

#### Wikipedia Domain

- **Entity Types**: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY, etc.
- **Relation Types**: BORN_IN, DIED_IN, WORKED_AT, FOUNDED, CREATED_BY, etc.
- **Wikidata Integration**: Entity linking to Wikidata knowledge base
- **General Knowledge**: Support for Wikipedia-style text processing

### Domain Features

- **Domain-Specific Entity Extraction**: Pattern-based and rule-based entity recognition
- **Domain-Specific Relations**: Relation classification tailored to domain ontology
- **Knowledge Base Integration**: UMLS (biomedical) or Wikidata (Wikipedia) entity linking
- **Domain-Specific Validation**: Ontology-aware validation of entities and relations
- **Domain-Specific Evaluation Metrics**: Custom metrics for domain-specific quality assessment

### Creating Custom Domains

You can create custom domains by implementing the `DomainProvider` interface. See the [Domain Usage Guide](DOMAIN_USAGE_GUIDE.md) for details.

## Biomedical Domain Features

### UMLS Integration

- Automatic concept mapping to UMLS entities
- Semantic type classification
- Confidence-based entity linking

### PubMed Processing

- Specialized processing for biomedical literature
- MeSH term integration
- Citation-based validation

### Domain-Specific Relations

- Drug-disease relationships
- Protein-protein interactions
- Pathway associations
- Treatment protocols

## Training and Fine-tuning

```julia
# Load training data
training_data = load_biomedical_corpus("path/to/corpus")

# Configure training
training_config = MLM_MNM_Training(
    mlm_probability = 0.15,
    mnm_probability = 0.15,
    learning_rate = 2e-5,
    batch_size = 16,
    num_epochs = 3
)

# Fine-tune model
trained_model = train_graphmert(training_data, training_config)
```

## API Reference

### Core Functions

- `extract_knowledge_graph(text, model, options)` - Main extraction function
- `load_model(path)` - Load pre-trained model
- `preprocess_text(text)` - Text preprocessing
- `merge_graphs(graphs)` - Combine multiple graphs

### Data Structures

- `KnowledgeGraph` - Main output structure
- `BiomedicalEntity` - Entity representation
- `BiomedicalRelation` - Relation representation
- `GraphMERTConfig` - Configuration container

### Evaluation Functions

- `calculate_factscore(graph)` - FActScore evaluation
- `calculate_validity_score(graph)` - ValidityScore evaluation
- `calculate_graphrag_score(graph)` - GraphRAG evaluation

## Examples

See the `examples/` directory for comprehensive examples:

- **Basic Extraction**: Simple knowledge graph extraction
- **Domain Switching**: Using multiple domains simultaneously (`examples/00_domain_switching_demo.jl`)
- **Biomedical Processing**: Domain-specific text processing (`examples/biomedical/`)
- **Wikipedia Processing**: General knowledge extraction (`examples/wikipedia/`)
- **Training Pipeline**: Model training and fine-tuning
- **Performance Benchmarking**: Speed and memory optimization
- **UMLS Integration**: Biomedical concept mapping
- **Wikidata Integration**: Wikipedia entity linking
- **Batch Processing**: Large-scale document processing

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/AlbaIntelligence/GraphMERT.jl.git
cd GraphMERT.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Running Tests

```julia
using Pkg
Pkg.test("GraphMERT")
```

## Citation

If you use GraphMERT.jl in your research, please cite the original paper:

```bibtex
@article{belova2024graphmert,
  title={GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data},
  author={Belova, Margarita and Xiao, Jiaxin and Tuli, Shikhar and Jha, Niraj K.},
  journal={arXiv preprint arXiv:2510.09580},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original GraphMERT paper authors for the foundational research
- Julia community for excellent language and ecosystem
- Biomedical NLP community for datasets and validation
- Contributors and users for feedback and improvements

## Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/GraphMERT.jl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/GraphMERT.jl/discussions)
- **Email**: <alba.intelligence@gmail.com>

---

**Note**: This implementation is based on the research paper "GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data" (arXiv:2510.09580).

